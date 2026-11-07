"""Ranking metrics. Pure functions over ranked id lists, easy to verify by hand.

Conventions: ranks are 1-based, relevance is binary, and every metric is computed
at a cutoff. Queries with no relevant items are the caller's problem: these
functions return 0.0 for them rather than raising, and the harness never feeds them
one because every bundled question has evidence.
"""

from __future__ import annotations

import math
from collections.abc import Collection, Sequence


def recall_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(set(ranked[:k]) & set(relevant)) / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: Collection[str], k: int) -> float:
    if k <= 0:
        return 0.0
    return sum(1 for r in ranked[:k] if r in relevant) / k


def reciprocal_rank(ranked: Sequence[str], relevant: Collection[str], k: int | None = None) -> float:
    for i, r in enumerate(ranked[:k] if k else ranked, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Collection[str], k: int, n_relevant: int | None = None) -> float:
    """Binary-gain nDCG.

    ``n_relevant`` is how many relevant items exist in the whole index (defaults to
    ``len(relevant)``). The ideal ranking puts min(k, n_relevant) of them on top.
    """
    total = len(relevant) if n_relevant is None else n_relevant
    if total == 0:
        return 0.0
    dcg = sum(1.0 / math.log2(i + 1) for i, r in enumerate(ranked[:k], start=1) if r in relevant)
    idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(k, total) + 1))
    return dcg / idcg


def mean(values: Sequence[float | None]) -> float | None:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None
