"""Chunkers: how documents are cut into retrievable units.

Chunking is the most under-tested decision in most RAG systems. Every chunker here
returns spans (start, end) into the original text, so chunk text is always an exact
slice of the source and citations stay auditable. Sizes are in characters: it is
model-agnostic and cheap, and a ~4 chars/token rule of thumb converts it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence

from .text import _trim, sentence_spans
from .types import Chunk, Document


class Chunker(ABC):
    name: str = "chunker"

    @abstractmethod
    def spans(self, text: str) -> list[tuple[int, int]]:
        """Return (start, end) spans in document order."""

    def chunk(self, doc: Document) -> list[Chunk]:
        return [
            Chunk(f"{doc.doc_id}#{i}", doc.doc_id, doc.text[s:e], s, e)
            for i, (s, e) in enumerate(self.spans(doc.text))
        ]

    def chunk_all(self, docs: Iterable[Document]) -> list[Chunk]:
        return [c for d in docs for c in self.chunk(d)]


class FixedSizeChunker(Chunker):
    """Sliding window of ``size`` characters advancing by ``size - overlap``.

    The naive baseline. It ignores structure entirely, so it happily cuts words and
    sentences in half; overlap is the usual band-aid so that a fact cut at one
    boundary is whole in the neighbouring window.
    """

    name = "fixed"

    def __init__(self, size: int = 400, overlap: int = 80) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if not 0 <= overlap < size:
            # overlap >= size would never advance the window: an infinite loop.
            raise ValueError("overlap must satisfy 0 <= overlap < size")
        self.size, self.overlap = size, overlap

    def spans(self, text: str) -> list[tuple[int, int]]:
        out: list[tuple[int, int]] = []
        step = self.size - self.overlap
        start = 0
        while start < len(text):
            end = min(start + self.size, len(text))
            span = _trim(text, start, end)
            if span:
                out.append(span)
            if end == len(text):
                break
            start += step
        return out


class RecursiveChunker(Chunker):
    """Split on the coarsest separator that works, recursing to finer ones.

    Paragraphs first, then lines, then sentence ends, then spaces, then a hard cut.
    Adjacent small pieces are merged back up to ``size`` so we do not emit a chunk
    per short sentence. This is the strategy LangChain popularised; it is
    re-implemented here over spans so offsets survive.
    """

    name = "recursive"

    def __init__(self, size: int = 400, separators: Sequence[str] = ("\n\n", "\n", ". ", " ")) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        self.size = size
        self.separators = tuple(separators)

    def spans(self, text: str) -> list[tuple[int, int]]:
        pieces = self._split(text, 0, len(text), 0)
        return self._merge(text, pieces)

    def _split(self, text: str, start: int, end: int, level: int) -> list[tuple[int, int]]:
        if end - start <= self.size:
            return [(start, end)]
        if level >= len(self.separators):
            # No structure left to exploit: hard cut. Only reached by pathological
            # input such as a 2 KB token with no spaces.
            return [(i, min(i + self.size, end)) for i in range(start, end, self.size)]
        sep = self.separators[level]
        if sep not in text[start:end]:
            return self._split(text, start, end, level + 1)
        out: list[tuple[int, int]] = []
        pos = start
        while pos < end:
            idx = text.find(sep, pos, end)
            # Keep the separator on the left piece so the merged text is contiguous.
            cut = end if idx == -1 else idx + len(sep)
            out.extend(self._split(text, pos, cut, level + 1))
            pos = cut
        return out

    def _merge(self, text: str, pieces: list[tuple[int, int]]) -> list[tuple[int, int]]:
        merged: list[tuple[int, int]] = []
        cur: tuple[int, int] | None = None
        for s, e in pieces:
            if cur is None:
                cur = (s, e)
            elif e - cur[0] <= self.size:
                cur = (cur[0], e)  # pieces are contiguous, so merging is just widening
            else:
                merged.append(cur)
                cur = (s, e)
        if cur is not None:
            merged.append(cur)
        return [t for s, e in merged if (t := _trim(text, s, e))]


class SentenceChunker(Chunker):
    """Pack whole sentences up to ``size`` characters, overlapping by sentences.

    Never cuts a sentence, which is what most answers live in. A single sentence
    longer than ``size`` becomes its own (oversized) chunk rather than being cut:
    breaking it would destroy exactly the unit this chunker exists to preserve.
    """

    name = "sentence"

    def __init__(self, size: int = 400, overlap_sentences: int = 1) -> None:
        if size <= 0:
            raise ValueError("size must be positive")
        if overlap_sentences < 0:
            raise ValueError("overlap_sentences must be >= 0")
        self.size, self.overlap = size, overlap_sentences

    def spans(self, text: str) -> list[tuple[int, int]]:
        sents = list(sentence_spans(text))
        out: list[tuple[int, int]] = []
        i = 0
        while i < len(sents):
            j = i + 1
            while j < len(sents) and sents[j][1] - sents[i][0] <= self.size:
                j += 1
            out.append((sents[i][0], sents[j - 1][1]))
            if j >= len(sents):
                break
            # Step back by the overlap, but always advance at least one sentence or a
            # chunk holding a single sentence would loop forever.
            i = max(j - self.overlap, i + 1)
        return out


CHUNKERS: dict[str, type[Chunker]] = {
    c.name: c for c in (FixedSizeChunker, RecursiveChunker, SentenceChunker)
}


def make_chunker(name: str, size: int = 400) -> Chunker:
    try:
        return CHUNKERS[name](size=size)
    except KeyError:
        raise ValueError(f"unknown chunker {name!r}; choose from {sorted(CHUNKERS)}") from None
