import numpy as np
import pytest

from tests.fakes import FakeEmbedder
from thairag.embeddings import QueryCache, normalize
from thairag.index import build_index, load_index
from thairag.retrievers import BM25Retriever, HybridRetriever, VectorRetriever
from thairag.tokenize import tokenize


class StaticRetriever:
    def __init__(self, ids):
        self.ids = ids

    def search(self, query, k):
        return [(i, 1.0) for i in self.ids[:k]]


def test_tokenize_segments_thai_and_drops_stopwords_and_punctuation():
    tokens = tokenize("วัดภูมินทร์ ตั้งอยู่ที่จังหวัดน่าน (ภาคเหนือ)")
    assert "วัด" in tokens or "วัดภูมินทร์" in tokens
    assert "น่าน" in tokens
    assert "(" not in tokens and ")" not in tokens
    assert "ที่" not in tokens  # stopword


def test_bm25_finds_the_matching_chunk(chunks):
    hits = BM25Retriever(chunks).search("อุทยานแห่งชาติแห่งแรกของไทยคือที่ไหน", 2)
    assert hits[0][0] == "khaoyai#0"


def test_bm25_returns_nothing_for_a_query_with_no_words(chunks):
    assert BM25Retriever(chunks).search("?!", 3) == []


def test_vector_retriever_ranks_by_cosine(chunks):
    embedder = FakeEmbedder()
    matrix = embedder.embed_documents([c.index_text for c in chunks])
    hits = VectorRetriever([c.id for c in chunks], matrix, embedder).search("รถไฟฟ้าบีทีเอสในกรุงเทพมหานคร", 4)

    assert hits[0][0] == "bkk#0"
    assert [s for _, s in hits] == sorted([s for _, s in hits], reverse=True)


def test_vector_retriever_rejects_mismatched_index(chunks):
    with pytest.raises(ValueError):
        VectorRetriever(["a"], np.zeros((2, 4)), FakeEmbedder())


def test_k_larger_than_corpus_returns_everything(chunks):
    assert len(BM25Retriever(chunks).search("ประเทศไทย", 50)) == len(chunks)


def test_hybrid_uses_reciprocal_rank_fusion():
    a = StaticRetriever(["x", "y", "z"])
    b = StaticRetriever(["z", "y", "w"])
    hits = HybridRetriever([a, b], rrf_k=60).search("q", 4)
    scores = dict(hits)

    assert scores["y"] == pytest.approx(2 / 62)
    assert scores["z"] == pytest.approx(1 / 63 + 1 / 61)
    assert scores["x"] == pytest.approx(1 / 61)
    # Found by both retrievers beats found by one, even at the top of one list
    assert {h[0] for h in hits[:2]} == {"y", "z"}
    assert hits[-1][0] == "w"


def test_query_cache_embeds_each_query_once():
    embedder = FakeEmbedder()
    cache = QueryCache(embedder)
    cache.prefill(["ก", "ข", "ก"])
    cache.embed_query("ก")
    cache.embed_query("ข")

    assert embedder.query_calls == 2


def test_normalize_handles_zero_vectors():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out, [[0.6, 0.8], [0.0, 0.0]])


def test_index_round_trip(chunks, tmp_path):
    embedder = FakeEmbedder()
    matrix = build_index(chunks, embedder, tmp_path, meta={"embed_model": "fake"})
    loaded_chunks, loaded_matrix, meta = load_index(tmp_path)

    assert loaded_chunks == chunks
    assert np.allclose(loaded_matrix, matrix)
    assert meta == {"chunks": 4, "dim": 64, "embed_model": "fake"}
