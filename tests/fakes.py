"""Deterministic stand-ins for the Gemini embedder and LLM."""
import hashlib
from typing import Callable

import numpy as np

from thairag.embeddings import normalize
from thairag.tokenize import tokenize


class FakeEmbedder:
    """Hashed bag-of-words vectors: texts sharing words get similar vectors."""

    def __init__(self, dim: int = 64):
        self.dim = dim
        self.query_calls = 0

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dim, dtype=np.float32)
        for token in tokenize(text):
            vec[int(hashlib.md5(token.encode()).hexdigest(), 16) % self.dim] += 1.0
        return normalize(vec)

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._vector(t) for t in texts])

    def embed_query(self, text: str) -> np.ndarray:
        self.query_calls += 1
        return self._vector(text)


class FakeLLM:
    """Returns `reply(prompt)` and records every call."""

    def __init__(self, reply: Callable[[str], str] | str):
        self._reply = reply if callable(reply) else (lambda _prompt: reply)
        self.calls: list[dict] = []

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        self.calls.append({"prompt": prompt, "system": system, "json_mode": json_mode})
        return self._reply(prompt)
