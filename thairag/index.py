"""Build, save and load the chunk + embedding index."""
import json
from pathlib import Path

import numpy as np

from thairag.chunking import Chunk, chunk_document
from thairag.embeddings import Embedder


def load_corpus(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def chunk_corpus(docs: list[dict], max_chars: int = 700) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        chunks.extend(chunk_document(str(doc["doc_id"]), doc["title"], doc["text"], max_chars=max_chars))
    return chunks


def build_index(chunks: list[Chunk], embedder: Embedder, out_dir: Path, meta: dict | None = None) -> np.ndarray:
    out_dir.mkdir(parents=True, exist_ok=True)
    matrix = embedder.embed_documents([c.index_text for c in chunks])
    np.save(out_dir / "embeddings.npy", matrix)
    with open(out_dir / "chunks.jsonl", "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk.to_dict(), ensure_ascii=False) + "\n")
    info = {"chunks": len(chunks), "dim": int(matrix.shape[1]), **(meta or {})}
    (out_dir / "meta.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return matrix


def load_index(index_dir: Path) -> tuple[list[Chunk], np.ndarray, dict]:
    with open(index_dir / "chunks.jsonl", encoding="utf-8") as f:
        chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
    matrix = np.load(index_dir / "embeddings.npy")
    meta = json.loads((index_dir / "meta.json").read_text(encoding="utf-8"))
    if len(chunks) != matrix.shape[0]:
        raise ValueError(f"Index is inconsistent: {len(chunks)} chunks, {matrix.shape[0]} embeddings")
    return chunks, matrix, meta
