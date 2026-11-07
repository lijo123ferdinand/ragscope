"""Answer generation with verifiable citations.

The model is asked to cite sources as ``[n]``. We then *check* the citations instead
of trusting them: markers that point at a source number that was never provided are
reported in ``invalid_citations``. Hallucinated citations are one of the most
common and most damaging RAG failures, because they make a wrong answer look sourced.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass, field

from . import prompts
from .llm import IDK, LLMProvider
from .types import Chunk, ScoredChunk

_CITATION_RE = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


@dataclass(frozen=True)
class Citation:
    n: int  # the marker number used in the answer text
    chunk: Chunk  # where it points; chunk.start/end locate it in the source document


@dataclass
class Answer:
    question: str
    text: str
    contexts: list[ScoredChunk]
    citations: list[Citation] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def is_refusal(self) -> bool:
        return self.text.strip() == IDK

    @property
    def cited_doc_ids(self) -> list[str]:
        return list(dict.fromkeys(c.chunk.doc_id for c in self.citations))


def parse_citations(text: str) -> list[int]:
    """Citation numbers in order of first appearance; handles ``[1][2]`` and ``[1, 2]``."""
    seen: dict[int, None] = {}
    for group in _CITATION_RE.findall(text):
        for n in group.split(","):
            seen.setdefault(int(n), None)
    return list(seen)


def generate_answer(
    question: str, contexts: Sequence[ScoredChunk], provider: LLMProvider, *, max_tokens: int = 400
) -> Answer:
    contexts = list(contexts)
    if not contexts:
        # Do not spend an LLM call to learn that nothing was retrieved.
        return Answer(question, IDK, contexts)
    sources = [(sc.chunk.doc_id, sc.chunk.text) for sc in contexts]
    text = provider.complete(prompts.ANSWER_SYSTEM, prompts.answer_prompt(question, sources), max_tokens=max_tokens)
    text = text.strip()
    citations, invalid = [], []
    for n in parse_citations(text):
        if 1 <= n <= len(contexts):
            citations.append(Citation(n, contexts[n - 1].chunk))
        else:
            invalid.append(n)
    return Answer(question, text, contexts, citations, invalid)
