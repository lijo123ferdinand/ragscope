"""Tokenisation and sentence splitting shared by every stage.

One tokenizer for BM25, the hashing embedder, the reranker and the mock LLM keeps
comparisons honest: if BM25 and dense retrieval disagree, it is because of the
scoring model, not because they saw different tokens.
"""

from __future__ import annotations

import re
from collections.abc import Iterator

_TOKEN_RE = re.compile(r"[a-z0-9]+")

# A compact English stopword list. Large lists (NLTK's ~180 words) start removing
# terms that carry meaning in technical text ("not", "no", "all", "before"), which
# matters for questions like "Can a Bloom filter return false negatives?".
STOPWORDS = frozenset(
    """a an and are as at be been but by can could did do does for from had has have
    how i if in into is it its of on or so such than that the their them then there
    these they this those to was we were what when where which while who why will
    with would you your""".split()
)


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    tokens = _TOKEN_RE.findall(text.lower())
    if keep_stopwords:
        return tokens
    return [t for t in tokens if t not in STOPWORDS]


_SUFFIXES = ("ations", "ation", "ings", "ing", "ers", "ed", "es", "er", "s")


def stem(token: str) -> str:
    """Crude, deterministic stemmer: strip one common suffix, then truncate to 6.

    Only the mock LLM/judge and the overlap reranker use this, to treat
    "elect"/"elects" and "replicated"/"replication" as one concept. BM25
    deliberately does NOT stem, so the benchmark shows the classic lexical-mismatch
    weakness that sub-word dense features are meant to cover.
    """
    for suf in _SUFFIXES:
        if token.endswith(suf) and len(token) - len(suf) >= 3:
            token = token[: -len(suf)]
            break
    return token[:6]


# Abbreviations whose trailing period must not end a sentence.
_ABBREVIATIONS = ("e.g.", "i.e.", "etc.", "vs.", "Dr.", "Mr.", "Mrs.", "Fig.", "No.")
_SENT_END_RE = re.compile(r"[.!?]+[\"')\]]*(?=\s+|$)")


def sentence_spans(text: str) -> Iterator[tuple[int, int]]:
    """Yield (start, end) spans of sentences, trimmed of surrounding whitespace.

    A regex splitter rather than a model: it is deterministic, dependency-free and
    good enough for well-formed prose. Paragraph breaks always end a sentence, so a
    heading without a period does not swallow the following paragraph.
    """
    for block in re.finditer(r"[^\n]+(?:\n(?!\s*\n)[^\n]+)*", text):
        b_start, b_end = block.span()
        pos = b_start
        for m in _SENT_END_RE.finditer(text, b_start, b_end):
            end = m.end()
            if any(text[max(b_start, end - len(a)) : end] == a for a in _ABBREVIATIONS):
                continue
            span = _trim(text, pos, end)
            if span:
                yield span
            pos = end
        span = _trim(text, pos, b_end)
        if span:
            yield span


def _trim(text: str, start: int, end: int) -> tuple[int, int] | None:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return (start, end) if start < end else None


def normalize(text: str) -> str:
    """Lowercase and collapse whitespace; used for evidence matching."""
    return " ".join(text.lower().split())
