"""LLM-as-judge metrics: faithfulness and answer relevance.

* Faithfulness = supported claims / total claims, judged against the retrieved
  context only. It catches answers that are fluent and on-topic but invented.
* Answer relevance = how directly the answer addresses the question, independent
  of truth. It catches answers that are faithful but evasive or off-target.

You need both: a model that copies an irrelevant sentence scores 1.0 on
faithfulness and ~0 on relevance; a model that confidently answers from its own
memory scores the reverse.

Judge output is parsed defensively. Models wrap JSON in prose or code fences, so we
extract the outermost object; if that still fails we return ``None`` and the
harness reports it as a judge failure rather than silently scoring 0.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from . import prompts
from .generate import Answer
from .llm import LLMProvider

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text: str) -> dict | None:
    m = _FENCE_RE.search(text)
    if m:
        text = m.group(1)
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        obj = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


@dataclass
class FaithfulnessResult:
    score: float | None  # None: refusal (no claims) or unparseable judge output
    claims: list[dict] = field(default_factory=list)
    error: str | None = None


def faithfulness(answer: Answer, provider: LLMProvider) -> FaithfulnessResult:
    if answer.is_refusal or not answer.contexts:
        # A refusal makes no claims. Scoring it 1.0 would reward refusing everything,
        # scoring it 0.0 would punish honesty, so it is excluded from the average.
        return FaithfulnessResult(None)
    sources = [(sc.chunk.doc_id, sc.chunk.text) for sc in answer.contexts]
    raw = provider.complete(prompts.FAITHFULNESS_SYSTEM, prompts.faithfulness_prompt(answer.text, sources))
    obj = extract_json(raw)
    if obj is None or not isinstance(obj.get("claims"), list):
        return FaithfulnessResult(None, error=f"unparseable judge output: {raw[:120]!r}")
    claims = [c for c in obj["claims"] if isinstance(c, dict)]
    if not claims:
        return FaithfulnessResult(None, claims)
    supported = sum(bool(c.get("supported")) for c in claims)
    return FaithfulnessResult(supported / len(claims), claims)


def answer_relevance(answer: Answer, provider: LLMProvider) -> float | None:
    raw = provider.complete(prompts.RELEVANCE_SYSTEM, prompts.relevance_prompt(answer.question, answer.text))
    obj = extract_json(raw)
    if obj is None:
        return None
    try:
        score = float(obj["score"])
    except (KeyError, TypeError, ValueError):
        return None
    return min(1.0, max(0.0, score))  # clamp: judges occasionally answer 1.2 or -0
