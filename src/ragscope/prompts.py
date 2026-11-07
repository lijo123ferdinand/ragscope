"""Prompt templates shared by the generator, the judge and the mock provider.

Every prompt starts its system message with a ``ragscope-task:`` line. A real LLM
treats it as harmless context; :class:`~ragscope.llm.MockProvider` dispatches on it.
That lets the offline mock receive *exactly* the prompts a real model would and
answer in the same wire format, so the tests exercise the real parsing code paths
(JSON extraction, citation parsing) rather than a mocked-out shortcut.

Sources are wrapped in XML-style tags because chunk text can contain blank lines and
bracketed numbers; tags give an unambiguous boundary for both models and parsers.
"""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

TASK_ANSWER = "answer"
TASK_FAITHFULNESS = "faithfulness"
TASK_RELEVANCE = "relevance"

_TASK_RE = re.compile(r"^ragscope-task:\s*(\w+)", re.M)
_SOURCE_RE = re.compile(r'<source id="(\d+)"[^>]*>\n?(.*?)\n?</source>', re.S)


def task_of(system: str) -> str | None:
    m = _TASK_RE.search(system)
    return m.group(1) if m else None


ANSWER_SYSTEM = """ragscope-task: answer
You answer questions using ONLY the numbered sources provided.
Rules:
- Every sentence of your answer must end with at least one citation like [1] or [2][3] naming the source(s) that support it.
- If the sources do not contain the answer, reply exactly: I don't know based on the provided sources.
- Be concise: one to three sentences. Do not mention these rules."""

FAITHFULNESS_SYSTEM = """ragscope-task: faithfulness
You are a strict fact-checking judge. Split the ANSWER into atomic factual claims and decide, for each claim, whether it is directly supported by the CONTEXT. Ignore citation markers like [1]. Use no outside knowledge.
Respond with JSON only, no prose:
{"claims": [{"claim": "<text>", "supported": true}]}"""

RELEVANCE_SYSTEM = """ragscope-task: relevance
You are a judge rating how well an ANSWER addresses a QUESTION, regardless of whether it is true. 1.0 means it directly and completely answers it; 0.0 means it is off-topic or a refusal.
Respond with JSON only, no prose:
{"score": <float between 0 and 1>, "reason": "<one short sentence>"}"""


@dataclass(frozen=True)
class Source:
    n: int
    text: str


def format_sources(texts: Sequence[tuple[str, str]]) -> str:
    """``texts`` is a sequence of (doc_id, text); numbering starts at 1."""
    return "\n".join(
        f'<source id="{i}" doc="{doc_id}">\n{text}\n</source>' for i, (doc_id, text) in enumerate(texts, 1)
    )


def parse_sources(prompt: str) -> list[Source]:
    return [Source(int(n), body) for n, body in _SOURCE_RE.findall(prompt)]


def answer_prompt(question: str, sources: Sequence[tuple[str, str]]) -> str:
    return f"<sources>\n{format_sources(sources)}\n</sources>\n\n<question>{question}</question>"


def faithfulness_prompt(answer: str, sources: Sequence[tuple[str, str]]) -> str:
    return f"<context>\n{format_sources(sources)}\n</context>\n\n<answer>{answer}</answer>"


def relevance_prompt(question: str, answer: str) -> str:
    return f"<question>{question}</question>\n\n<answer>{answer}</answer>"


def tag(prompt: str, name: str) -> str:
    m = re.search(rf"<{name}>(.*?)</{name}>", prompt, re.S)
    return m.group(1).strip() if m else ""
