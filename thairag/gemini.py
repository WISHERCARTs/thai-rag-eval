"""Gemini API access: embeddings and text generation, with retry on rate limits."""
import logging
import os
import time
from typing import Callable, TypeVar

import numpy as np
from dotenv import load_dotenv

from thairag.embeddings import normalize

logger = logging.getLogger(__name__)
T = TypeVar("T")

DEFAULT_EMBED_MODEL = "gemini-embedding-001"
DEFAULT_CHAT_MODEL = "gemini-2.5-flash"


def make_client():
    from google import genai

    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Set GEMINI_API_KEY in the environment or in .env")
    return genai.Client(api_key=api_key)


def with_retry(call: Callable[[], T], attempts: int = 6, base_delay: float = 2.0) -> T:
    """Retry transient API failures (rate limits, 5xx) with exponential backoff."""
    for attempt in range(1, attempts + 1):
        try:
            return call()
        except Exception as exc:  # the SDK raises several error types
            if attempt == attempts:
                raise
            delay = base_delay * 2 ** (attempt - 1)
            logger.warning("Gemini call failed (%s); retry %d in %.0fs", exc, attempt, delay)
            time.sleep(delay)
    raise AssertionError("unreachable")


class GeminiEmbedder:
    def __init__(self, client, model: str | None = None, dim: int = 768, batch_size: int = 50):
        self.client = client
        self.model = model or os.getenv("EMBED_MODEL", DEFAULT_EMBED_MODEL)
        self.dim = dim
        self.batch_size = batch_size

    def _embed(self, texts: list[str], task_type: str) -> np.ndarray:
        from google.genai import types

        config = types.EmbedContentConfig(task_type=task_type, output_dimensionality=self.dim)
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            batch = texts[start : start + self.batch_size]
            response = with_retry(
                lambda: self.client.models.embed_content(model=self.model, contents=batch, config=config)
            )
            vectors.extend(e.values for e in response.embeddings)
        # Truncated Gemini embeddings are not unit length, so normalise
        return normalize(np.asarray(vectors, dtype=np.float32))

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_DOCUMENT")

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self._embed(texts, "RETRIEVAL_QUERY")

    def embed_query(self, text: str) -> np.ndarray:
        return self.embed_queries([text])[0]


class GeminiLLM:
    def __init__(self, client, model: str | None = None, temperature: float = 0.2):
        self.client = client
        self.model = model or os.getenv("CHAT_MODEL", DEFAULT_CHAT_MODEL)
        self.temperature = temperature

    def generate(self, prompt: str, system: str | None = None, json_mode: bool = False) -> str:
        from google.genai import types

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=self.temperature,
            response_mime_type="application/json" if json_mode else None,
        )
        response = with_retry(
            lambda: self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        )
        return response.text or ""
