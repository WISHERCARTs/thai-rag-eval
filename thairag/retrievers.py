"""Keyword, vector and hybrid retrievers over the same chunk set."""
from typing import Protocol, Sequence

import numpy as np
from rank_bm25 import BM25Okapi

from thairag.chunking import Chunk
from thairag.embeddings import Embedder
from thairag.tokenize import tokenize

Hit = tuple[str, float]  # (chunk id, score), best first


class Retriever(Protocol):
    def search(self, query: str, k: int) -> list[Hit]: ...


def _top_k(ids: Sequence[str], scores: np.ndarray, k: int) -> list[Hit]:
    k = min(k, len(ids))
    order = np.argsort(-scores, kind="stable")[:k]
    return [(ids[i], float(scores[i])) for i in order]


class BM25Retriever:
    def __init__(self, chunks: Sequence[Chunk]):
        self.ids = [c.id for c in chunks]
        self.bm25 = BM25Okapi([tokenize(c.index_text) for c in chunks])

    def search(self, query: str, k: int) -> list[Hit]:
        tokens = tokenize(query)
        if not tokens:
            return []
        return _top_k(self.ids, self.bm25.get_scores(tokens), k)


class VectorRetriever:
    """Cosine similarity search. `matrix` rows must be L2-normalised."""

    def __init__(self, ids: Sequence[str], matrix: np.ndarray, embedder: Embedder):
        if len(ids) != matrix.shape[0]:
            raise ValueError(f"{len(ids)} ids but {matrix.shape[0]} embeddings")
        self.ids = list(ids)
        self.matrix = matrix
        self.embedder = embedder

    def search(self, query: str, k: int) -> list[Hit]:
        query_vec = self.embedder.embed_query(query)
        return _top_k(self.ids, self.matrix @ query_vec, k)


class HybridRetriever:
    """Reciprocal rank fusion: score = sum over retrievers of 1 / (rrf_k + rank).

    RRF uses ranks only, so BM25 scores and cosine similarities never need to be
    put on the same scale.
    """

    def __init__(self, retrievers: Sequence[Retriever], depth: int = 20, rrf_k: int = 60):
        self.retrievers = list(retrievers)
        self.depth = depth
        self.rrf_k = rrf_k

    def search(self, query: str, k: int) -> list[Hit]:
        fused: dict[str, float] = {}
        for retriever in self.retrievers:
            for rank, (chunk_id, _) in enumerate(retriever.search(query, self.depth), start=1):
                fused[chunk_id] = fused.get(chunk_id, 0.0) + 1.0 / (self.rrf_k + rank)
        ranked = sorted(fused.items(), key=lambda item: item[1], reverse=True)
        return ranked[:k]
