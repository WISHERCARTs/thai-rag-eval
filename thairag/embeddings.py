"""Embedder interface shared by the Gemini embedder and the test fakes."""
from typing import Protocol

import numpy as np


class Embedder(Protocol):
    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_query(self, text: str) -> np.ndarray: ...


class QueryCache:
    """Wraps an embedder so queries embedded in one batch are not embedded again.

    Evaluation asks every question against several retrievers; prefilling the
    cache turns one API call per question into one call per batch.
    """

    def __init__(self, embedder):
        self.embedder = embedder
        self.cache: dict[str, np.ndarray] = {}

    def prefill(self, queries: list[str]) -> None:
        missing = [q for q in dict.fromkeys(queries) if q not in self.cache]
        if not missing:
            return
        if hasattr(self.embedder, "embed_queries"):
            vectors = self.embedder.embed_queries(missing)
        else:
            vectors = [self.embedder.embed_query(q) for q in missing]
        self.cache.update(zip(missing, vectors))

    def embed_query(self, text: str) -> np.ndarray:
        if text not in self.cache:
            self.cache[text] = self.embedder.embed_query(text)
        return self.cache[text]

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.embedder.embed_documents(texts)


def normalize(matrix: np.ndarray) -> np.ndarray:
    """L2-normalise rows (or a single vector) so a dot product is cosine similarity."""
    norms = np.linalg.norm(matrix, axis=-1, keepdims=True)
    return matrix / np.where(norms == 0, 1, norms)
