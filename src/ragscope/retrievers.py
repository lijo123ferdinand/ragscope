"""Retrievers: BM25, dense, and hybrid (RRF over both).

All retrievers share one tiny interface, ``index(chunks)`` then
``search(query, k)``, so the evaluation harness can treat them uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence

import numpy as np

from .bm25 import BM25Index
from .embedders import Embedder, HashingEmbedder
from .fusion import reciprocal_rank_fusion
from .types import Chunk, ScoredChunk


class Retriever(ABC):
    name: str = "retriever"

    def __init__(self) -> None:
        self.chunks: list[Chunk] = []

    @abstractmethod
    def index(self, chunks: Sequence[Chunk]) -> None: ...

    @abstractmethod
    def search(self, query: str, k: int = 5) -> list[ScoredChunk]: ...

    def _require_index(self) -> None:
        if not self.chunks:
            raise RuntimeError(f"{type(self).__name__}.search called before index()")


class BM25Retriever(Retriever):
    name = "bm25"

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        super().__init__()
        self.bm25 = BM25Index(k1=k1, b=b)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.bm25.fit([c.text for c in self.chunks])

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        self._require_index()
        return [ScoredChunk(self.chunks[i], s, "bm25") for i, s in self.bm25.top_k(query, k)]


class DenseRetriever(Retriever):
    """Exact (brute-force) cosine search over an in-memory matrix.

    Exact search is the right call below ~1e5 chunks: a matrix-vector product is a
    few milliseconds and has perfect recall, so ANN (HNSW/IVF) would add a
    dependency and an accuracy knob for no benefit at this scale.
    """

    name = "dense"

    def __init__(self, embedder: Embedder | None = None) -> None:
        super().__init__()
        self.embedder = embedder or HashingEmbedder()
        self.matrix = np.zeros((0, 0), dtype=np.float32)

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        texts = [c.text for c in self.chunks]
        self.embedder.fit(texts)
        self.matrix = self.embedder.embed(texts)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        self._require_index()
        q = self.embedder.embed([query])[0]
        sims = self.matrix @ q  # rows are unit-norm, so dot product == cosine
        k = min(k, len(sims))
        # Stable sort on -sims: equal scores keep index order, so runs are reproducible.
        order = np.argsort(-sims, kind="stable")[:k]
        return [ScoredChunk(self.chunks[i], float(sims[i]), "dense") for i in order]


class HybridRetriever(Retriever):
    """Run BM25 and dense independently, fuse their rankings with RRF.

    ``depth`` is how many candidates each side contributes. It must be larger than
    k: a chunk ranked 12th by BM25 and 2nd by dense should be able to win, which is
    impossible if each side only hands over its top 5.
    """

    name = "hybrid"

    def __init__(
        self,
        sparse: Retriever | None = None,
        dense: Retriever | None = None,
        depth: int = 50,
        rrf_k: int = 60,
        weights: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        self.sparse = sparse or BM25Retriever()
        self.dense = dense or DenseRetriever()
        self.depth, self.rrf_k, self.weights = depth, rrf_k, weights

    def index(self, chunks: Sequence[Chunk]) -> None:
        self.chunks = list(chunks)
        self.sparse.index(self.chunks)
        self.dense.index(self.chunks)

    def search(self, query: str, k: int = 5) -> list[ScoredChunk]:
        self._require_index()
        depth = max(self.depth, k)
        by_id = {c.chunk_id: c for c in self.chunks}
        rankings = [
            [sc.chunk.chunk_id for sc in self.sparse.search(query, depth)],
            [sc.chunk.chunk_id for sc in self.dense.search(query, depth)],
        ]
        fused = reciprocal_rank_fusion(rankings, k=self.rrf_k, weights=self.weights)
        return [ScoredChunk(by_id[cid], s, "rrf") for cid, s in fused[:k]]


def make_retriever(name: str, embedder: Embedder | None = None) -> Retriever:
    if name == "bm25":
        return BM25Retriever()
    if name == "dense":
        return DenseRetriever(embedder)
    if name == "hybrid":
        return HybridRetriever(dense=DenseRetriever(embedder))
    raise ValueError(f"unknown retriever {name!r}; choose bm25, dense or hybrid")
