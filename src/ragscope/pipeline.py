"""The RAG pipeline: chunk -> index -> retrieve -> rerank -> generate.

Each stage is an injected object, so an experiment is a different constructor
call, not a different code path. That is also what makes the evaluation grid fair:
every configuration runs the identical orchestration.
"""

from __future__ import annotations

from collections.abc import Sequence

from .chunking import Chunker, SentenceChunker
from .generate import Answer, generate_answer
from .llm import LLMProvider, MockProvider
from .rerank import IdentityReranker, Reranker
from .retrievers import HybridRetriever, Retriever
from .types import Chunk, Document, ScoredChunk


class RAGPipeline:
    def __init__(
        self,
        chunker: Chunker | None = None,
        retriever: Retriever | None = None,
        reranker: Reranker | None = None,
        provider: LLMProvider | None = None,
        candidates: int = 20,
    ) -> None:
        self.chunker = chunker or SentenceChunker()
        self.retriever = retriever or HybridRetriever()
        self.reranker = reranker or IdentityReranker()
        self.provider = provider or MockProvider()
        self.candidates = candidates
        self.chunks: list[Chunk] = []
        self.documents: dict[str, Document] = {}

    def index(self, docs: Sequence[Document]) -> "RAGPipeline":
        self.documents = {d.doc_id: d for d in docs}
        if len(self.documents) != len(docs):
            raise ValueError("duplicate doc_id in corpus")
        self.chunks = self.chunker.chunk_all(docs)
        if not self.chunks:
            raise ValueError("corpus produced no chunks")
        self.retriever.index(self.chunks)
        return self

    def retrieve(self, query: str, k: int = 5) -> list[ScoredChunk]:
        if isinstance(self.reranker, IdentityReranker):
            return self.retriever.search(query, k)
        # Over-fetch so the reranker has something to reorder.
        pool = self.retriever.search(query, max(self.candidates, k))
        return self.reranker.rerank(query, pool, k)

    def answer(self, question: str, k: int = 3) -> Answer:
        return generate_answer(question, self.retrieve(question, k), self.provider)
