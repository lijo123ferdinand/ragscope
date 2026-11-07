"""LLM providers behind one small interface.

``complete(system, prompt) -> str`` is intentionally the whole contract: generation
and judging are single-turn, and a narrow interface is trivial to implement for a
new backend or to fake in a test.

* :class:`MockProvider`: deterministic, offline, the default everywhere.
* :class:`AnthropicProvider`: optional ``anthropic`` extra, key from ANTHROPIC_API_KEY.
* :class:`OpenAICompatProvider`: any ``/v1/chat/completions`` server (OpenAI,
  Ollama, vLLM, llama.cpp, LM Studio) via urllib, so no extra dependency.
"""

from __future__ import annotations

import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from typing import Any, Protocol, runtime_checkable

from . import prompts
from .text import sentence_spans, stem, tokenize


class ProviderError(RuntimeError):
    pass


@runtime_checkable
class LLMProvider(Protocol):
    name: str

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> str: ...


IDK = "I don't know based on the provided sources."
_CITE_RE = re.compile(r"\[\d+(?:\s*,\s*\d+)*\]")


def _terms(text: str) -> set[str]:
    return {stem(t) for t in tokenize(_CITE_RE.sub(" ", text))}


class MockProvider:
    """A deterministic stand-in for an LLM, built from transparent heuristics.

    * answer: extractive. Picks the source sentence(s) covering the most
      IDF-weighted question terms and cites them. Because it copies sentences verbatim, its faithfulness is
      ~1.0 by construction; the useful signals it produces are answer relevance and
      reference-answer hit rate, which depend entirely on retrieval quality.
    * faithfulness: a claim is supported if >= ``support_threshold`` of its terms
      occur in the context.
    * relevance: fraction of question terms the answer covers.

    It is a measuring instrument for the retrieval half of the pipeline, not a claim
    about LLM quality. Use a real provider for generation-quality numbers.
    """

    name = "mock"

    def __init__(self, support_threshold: float = 0.8, max_sentences: int = 2) -> None:
        self.support_threshold = support_threshold
        self.max_sentences = max_sentences
        self.calls = 0

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> str:
        self.calls += 1
        task = prompts.task_of(system)
        if task == prompts.TASK_ANSWER:
            return self._answer(prompt)
        if task == prompts.TASK_FAITHFULNESS:
            return self._faithfulness(prompt)
        if task == prompts.TASK_RELEVANCE:
            return self._relevance(prompt)
        raise ProviderError(f"MockProvider does not understand task {task!r}")

    def _answer(self, prompt: str) -> str:
        q = _terms(prompts.tag(prompt, "question"))
        if not q:
            return IDK
        sents: list[tuple[int, int, str, set[str]]] = []
        for src in prompts.parse_sources(prompt):
            for j, (s, e) in enumerate(sentence_spans(src.text)):
                sent = src.text[s:e]
                if not sent.lstrip().startswith("#"):  # skip markdown headings
                    sents.append((src.n, j, sent, _terms(sent)))
        if not sents:
            return IDK
        # IDF over the provided sentences, a stand-in for attention: a question term
        # that appears everywhere in the context ("tree" in an LSM-tree document)
        # tells us little about which sentence answers it.
        n = len(sents)
        weight = {t: math.log(1 + n / (1 + sum(t in s[3] for s in sents))) for t in q}
        total = sum(weight.values())
        scored = [(sum(weight[t] for t in q & terms) / total, sn, j, sent) for sn, j, sent, terms in sents]
        # Highest score first; earlier source and earlier sentence break ties.
        scored.sort(key=lambda c: (-c[0], c[1], c[2]))
        best = scored[0][0]
        if best == 0:
            return IDK
        # Overlapping chunks repeat sentences. Say each sentence once and cite every
        # source that contains it, which is what a careful writer would do.
        picked: dict[str, list[int]] = {}
        for score, sn, _, sent in scored:
            if score < 0.75 * best:
                break
            key = " ".join(sent.split())
            if key in picked:
                if sn not in picked[key]:
                    picked[key].append(sn)
            elif len(picked) < self.max_sentences:
                picked[key] = [sn]
        # Citation goes before the final period, the way people (and LLMs) write it.
        return " ".join(
            f"{sent.rstrip('.')} {''.join(f'[{n}]' for n in sorted(ns))}." for sent, ns in picked.items()
        )

    def _faithfulness(self, prompt: str) -> str:
        context = _terms(" ".join(s.text for s in prompts.parse_sources(prompt)))
        answer = prompts.tag(prompt, "answer")
        claims = []
        if answer and answer != IDK:
            clean = _CITE_RE.sub("", answer)
            for s, e in sentence_spans(clean):
                claim = clean[s:e].strip()
                terms = _terms(claim)
                if not terms:
                    continue
                ratio = len(terms & context) / len(terms)
                claims.append({"claim": claim, "supported": ratio >= self.support_threshold})
        return json.dumps({"claims": claims})

    def _relevance(self, prompt: str) -> str:
        q = _terms(prompts.tag(prompt, "question"))
        answer = prompts.tag(prompt, "answer")
        if not q or not answer or answer == IDK:
            return json.dumps({"score": 0.0, "reason": "refusal or empty"})
        score = round(len(q & _terms(answer)) / len(q), 4)
        return json.dumps({"score": score, "reason": "question-term coverage"})


class AnthropicProvider:
    """Claude via the official SDK. ``pip install 'ragscope[anthropic]'``."""

    name = "anthropic"

    def __init__(self, model: str = "claude-sonnet-5-5", api_key: str | None = None, client: Any = None) -> None:
        self.model = model
        if client is not None:  # dependency injection for tests / custom transports
            self._client = client
            return
        try:
            import anthropic
        except ImportError as exc:
            raise ImportError("AnthropicProvider needs: pip install 'ragscope[anthropic]'") from exc
        key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ProviderError("set ANTHROPIC_API_KEY or pass api_key=")
        # The SDK already retries 429/5xx with backoff; no second retry layer here.
        self._client = anthropic.Anthropic(api_key=key)

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> str:
        resp = self._client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")


class OpenAICompatProvider:
    """Any OpenAI-compatible chat completions endpoint, stdlib only.

    Defaults target a local Ollama server, which needs no key. For OpenAI set
    ``base_url="https://api.openai.com/v1"`` and OPENAI_API_KEY.
    """

    name = "openai"

    def __init__(
        self,
        model: str = "llama3.1",
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 3,
    ) -> None:
        self.model = model
        self.base_url = (base_url or os.environ.get("OPENAI_BASE_URL") or "http://localhost:11434/v1").rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("OPENAI_API_KEY", "")
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(self, system: str, prompt: str, *, max_tokens: int = 512) -> str:
        body = json.dumps(
            {
                "model": self.model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": prompt}],
                "temperature": 0,  # judging and benchmarking want repeatability
                "max_tokens": max_tokens,
            }
        ).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = f"{self.base_url}/chat/completions"
        for attempt in range(self.max_retries + 1):
            req = urllib.request.Request(url, data=body, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    payload = json.load(resp)
                return payload["choices"][0]["message"]["content"] or ""
            except urllib.error.HTTPError as exc:
                # Retry only what is plausibly transient; a 400/401 will not fix itself.
                if exc.code in (429, 500, 502, 503, 504) and attempt < self.max_retries:
                    time.sleep(min(2**attempt * 0.5, 8))
                    continue
                detail = exc.read()[:300].decode(errors="replace")
                raise ProviderError(f"HTTP {exc.code} from {url}: {detail}") from exc
            except urllib.error.URLError as exc:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt * 0.5, 8))
                    continue
                raise ProviderError(f"cannot reach {url}: {exc.reason}") from exc
            except (KeyError, IndexError, json.JSONDecodeError) as exc:
                raise ProviderError(f"unexpected response shape from {url}") from exc
        raise ProviderError("unreachable")  # pragma: no cover


def make_provider(name: str, model: str | None = None) -> LLMProvider:
    if name == "mock":
        return MockProvider()
    if name == "anthropic":
        return AnthropicProvider(model=model) if model else AnthropicProvider()
    if name in ("openai", "ollama"):
        return OpenAICompatProvider(model=model) if model else OpenAICompatProvider()
    raise ValueError(f"unknown provider {name!r}; choose mock, anthropic or openai")
