"""Thai word segmentation for keyword (BM25) search.

Thai is written without spaces between words, so whitespace splitting would
turn a whole sentence into one token. PyThaiNLP's dictionary-based `newmm`
segmenter splits it into words first.
"""
from functools import lru_cache

from pythainlp.corpus import thai_stopwords
from pythainlp.tokenize import word_tokenize


@lru_cache(maxsize=1)
def _stopwords() -> frozenset[str]:
    return frozenset(thai_stopwords())


def tokenize(text: str) -> list[str]:
    stop = _stopwords()
    tokens = word_tokenize(text.lower(), engine="newmm", keep_whitespace=False)
    return [t for t in tokens if t not in stop and any(ch.isalnum() for ch in t)]
