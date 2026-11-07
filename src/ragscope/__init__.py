"""ragscope: a production-shaped RAG pipeline with an evaluation harness.

The public surface is deliberately small; everything else is importable from its
module for people who want to swap one stage out.
"""

from .chunking import Chunker, FixedSizeChunker, RecursiveChunker, SentenceChunker
from .embedders import Embedder, HashingEmbedder
from .generate import Answer, Citation, generate_answer
from .llm import AnthropicProvider, LLMProvider, MockProvider, OpenAICompatProvider
from .pipeline import RAGPipeline
from .rerank import IdentityReranker, OverlapReranker, Reranker
from .retrievers import BM25Retriever, DenseRetriever, HybridRetriever, Retriever
from .types import Chunk, Document, ScoredChunk

__version__ = "0.1.0"

__all__ = [
    "Answer", "AnthropicProvider", "BM25Retriever", "Chunk", "Chunker", "Citation",
    "DenseRetriever", "Document", "Embedder", "FixedSizeChunker", "HashingEmbedder",
    "HybridRetriever", "IdentityReranker", "LLMProvider", "MockProvider",
    "OpenAICompatProvider", "OverlapReranker", "RAGPipeline", "RecursiveChunker",
    "Reranker", "Retriever", "ScoredChunk", "SentenceChunker", "generate_answer",
]
