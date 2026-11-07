"""Plug in your own embedder and reranker, then score them with the harness.

Anything with ``name``, ``fit(texts)`` and ``embed(texts) -> (n, d) unit-norm array``
is an embedder; anything subclassing ``Reranker`` is a reranker.

    python examples/custom_components.py
"""

import numpy as np

from ragscope import DenseRetriever, HashingEmbedder, RAGPipeline
from ragscope.dataset import contains_evidence, load_corpus, load_qa
from ragscope.metrics import mean, reciprocal_rank
from ragscope.rerank import Reranker
from ragscope.types import ScoredChunk


class WordOnlyEmbedder(HashingEmbedder):
    """Ablation: the default embedder with character n-grams switched off."""

    name = "hashing-words-only"

    def __init__(self) -> None:
        super().__init__(char_weight=0.0)


class ShortestFirstReranker(Reranker):
    """A deliberately silly reranker, to show the harness catches regressions."""

    name = "shortest"

    def rerank(self, query, candidates, k):
        return sorted(candidates, key=lambda sc: len(sc.chunk.text))[:k]


def mrr(pipe: RAGPipeline, qa) -> float:
    scores = []
    for item in qa:
        hits = pipe.retrieve(item.question, 10)
        relevant = {h.chunk.chunk_id for h in hits if any(contains_evidence(h.chunk, e) for e in item.evidence)}
        scores.append(reciprocal_rank([h.chunk.chunk_id for h in hits], relevant))
    return mean(scores)


docs, qa = load_corpus(), load_qa()
for label, pipe in [
    ("hashing (words + char n-grams)", RAGPipeline(retriever=DenseRetriever(HashingEmbedder()))),
    ("hashing (words only)", RAGPipeline(retriever=DenseRetriever(WordOnlyEmbedder()))),
    ("hashing + shortest-first rerank", RAGPipeline(retriever=DenseRetriever(), reranker=ShortestFirstReranker())),
]:
    pipe.index(docs)
    print(f"{label:<34} MRR@10 = {mrr(pipe, qa):.3f}")

assert isinstance(HashingEmbedder().embed(["x"]), np.ndarray)
