"""Core value types.

Chunks carry character offsets into their source document rather than only a copy
of the text. That is what makes a citation auditable: an answer that cites ``[2]``
can be traced to ``doc.text[start:end]`` and highlighted in a UI, and the test suite
can assert ``chunk.text == doc.text[chunk.start:chunk.end]`` for every chunker.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Document:
    doc_id: str
    text: str
    title: str = ""
    metadata: dict = field(default_factory=dict, compare=False, hash=False)


@dataclass(frozen=True, slots=True)
class Chunk:
    chunk_id: str  # "<doc_id>#<ordinal>", stable across runs for a given chunker config
    doc_id: str
    text: str
    start: int  # inclusive char offset into Document.text
    end: int  # exclusive


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk: Chunk
    score: float
    # Which stage produced the score ("bm25", "dense", "rrf", "rerank"). Scores from
    # different stages are not comparable, so we keep the provenance next to them.
    source: str = ""
