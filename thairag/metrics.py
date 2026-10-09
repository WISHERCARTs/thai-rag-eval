"""Retrieval metrics. `ranked` is best-first; `relevant` is the set of correct ids."""
from typing import Iterable, Sequence


def hit_at_k(ranked: Sequence[str], relevant: Iterable[str], k: int) -> float:
    relevant = set(relevant)
    return float(any(item in relevant for item in ranked[:k]))


def reciprocal_rank(ranked: Sequence[str], relevant: Iterable[str]) -> float:
    relevant = set(relevant)
    for rank, item in enumerate(ranked, start=1):
        if item in relevant:
            return 1.0 / rank
    return 0.0


def dedupe(items: Iterable[str]) -> list[str]:
    """Keep the first occurrence of each item, preserving order."""
    return list(dict.fromkeys(items))


def mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0
