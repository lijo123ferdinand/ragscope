"""Reciprocal Rank Fusion (Cormack, Clarke & Buettcher, SIGIR 2009).

    rrf(d) = sum over rankings r of  w_r / (k + rank_r(d))      (rank is 1-based)

Why rank fusion instead of adding scores: BM25 scores are unbounded and
query-dependent, cosine similarities live in [-1, 1]. Any linear blend needs
per-corpus score normalisation and a tuned weight, and silently breaks when either
distribution shifts. RRF only uses ranks, so it needs no calibration. k=60 is the
constant from the paper; it damps the advantage of the very top ranks so that a
document ranked 3rd by both systems beats one ranked 1st by only one.
"""

from __future__ import annotations

from collections.abc import Hashable, Sequence
from typing import TypeVar

K = TypeVar("K", bound=Hashable)


def reciprocal_rank_fusion(
    rankings: Sequence[Sequence[K]], k: int = 60, weights: Sequence[float] | None = None
) -> list[tuple[K, float]]:
    if k < 0:
        raise ValueError("k must be >= 0")
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must align with rankings")
    scores: dict[K, float] = {}
    first_seen: dict[K, int] = {}
    order = 0
    for r, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[r]
        for rank, key in enumerate(ranking, start=1):
            scores[key] = scores.get(key, 0.0) + w / (k + rank)
            if key not in first_seen:
                first_seen[key] = order
                order += 1
    # Deterministic tie-break: earlier appearance wins.
    return sorted(scores.items(), key=lambda kv: (-kv[1], first_seen[kv[0]]))
