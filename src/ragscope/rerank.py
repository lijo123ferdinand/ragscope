"""Second-stage rerankers.

The retrieve-then-rerank split exists because first-stage retrieval has to be
cheap over the whole corpus, while a reranker can afford to look closely at ~20-50
candidates. In production that second look is a cross-encoder or an LLM. Here the
interface is the point; :class:`OverlapReranker` is a transparent, deterministic
baseline and :class:`CrossEncoderReranker` plugs in a real model when installed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

from .text import stem, tokenize
from .types import ScoredChunk


class Reranker(ABC):
    name: str = "reranker"

    @abstractmethod
    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]: ...


class IdentityReranker(Reranker):
    """Keep first-stage order. The control arm for measuring a real reranker."""

    name = "none"

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        return list(candidates[:k])


class OverlapReranker(Reranker):
    """Score = query-term coverage (stemmed) + bonus for query bigrams in order.

    Coverage rewards chunks that mention *all* the query's concepts, which is the
    main thing a first-stage ranker gets wrong (BM25 happily ranks a chunk that
    repeats one rare term). The bigram bonus approximates proximity. The original
    first-stage rank breaks ties so the reranker never shuffles without a reason.
    """

    name = "overlap"

    def __init__(self, bigram_weight: float = 0.5) -> None:
        self.bigram_weight = bigram_weight

    def _score(self, q_terms: list[str], q_bigrams: set[tuple[str, str]], text: str) -> float:
        terms = [stem(t) for t in tokenize(text)]
        tset = set(terms)
        coverage = sum(t in tset for t in q_terms) / len(q_terms)
        if not q_bigrams:
            return coverage
        bigrams = set(zip(terms, terms[1:]))
        return coverage + self.bigram_weight * len(q_bigrams & bigrams) / len(q_bigrams)

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        q_terms = list(dict.fromkeys(stem(t) for t in tokenize(query)))
        if not q_terms:
            return list(candidates[:k])
        q_bigrams = set(zip(q_terms, q_terms[1:]))
        scored = [
            (self._score(q_terms, q_bigrams, sc.chunk.text), -i, sc) for i, sc in enumerate(candidates)
        ]
        scored.sort(key=lambda t: (t[0], t[1]), reverse=True)
        return [ScoredChunk(sc.chunk, s, "rerank") for s, _, sc in scored[:k]]


class CrossEncoderReranker(Reranker):  # pragma: no cover - optional dependency
    """sentence-transformers CrossEncoder. Optional extra: ``ragscope[st]``."""

    name = "cross-encoder"

    def __init__(self, model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2") -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:
            raise ImportError("CrossEncoderReranker needs: pip install 'ragscope[st]'") from exc
        self._model = CrossEncoder(model)

    def rerank(self, query: str, candidates: Sequence[ScoredChunk], k: int) -> list[ScoredChunk]:
        if not candidates:
            return []
        scores = self._model.predict([(query, c.chunk.text) for c in candidates])
        order = sorted(range(len(candidates)), key=lambda i: (-float(scores[i]), i))
        return [ScoredChunk(candidates[i].chunk, float(scores[i]), "rerank") for i in order[:k]]


def make_reranker(name: str) -> Reranker:
    if name == "none":
        return IdentityReranker()
    if name == "overlap":
        return OverlapReranker()
    if name == "cross-encoder":
        return CrossEncoderReranker()
    raise ValueError(f"unknown reranker {name!r}; choose none, overlap or cross-encoder")
