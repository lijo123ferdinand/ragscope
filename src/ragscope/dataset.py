"""Corpus and QA-set loading, plus the definition of a relevant chunk.

Relevance is defined by *evidence spans*, not by document id. Each question lists
the exact sentence(s) that answer it; a chunk is relevant if it contains a whole
evidence span. Document-level labels would make every chunker look identical (any
chunk of the right document would count), which hides exactly the failure that
matters: a fixed-size window that cut the answer sentence in half.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from importlib import resources
from pathlib import Path

from .text import normalize
from .types import Chunk, Document


@dataclass(frozen=True)
class Evidence:
    doc_id: str
    text: str


@dataclass(frozen=True)
class QAItem:
    qid: str
    question: str
    answer: str  # short reference answer, used for the reference-hit metric
    evidence: tuple[Evidence, ...]
    # Free-form labels. The bundled set uses "lexical" (question reuses the source's
    # words) and "paraphrase" (deliberate vocabulary mismatch) so the two failure
    # regimes can be measured separately with ``ragscope eval --tag``.
    tags: tuple[str, ...] = ()


def _doc_from_text(doc_id: str, text: str) -> Document:
    first = text.lstrip().splitlines()[0] if text.strip() else ""
    title = first.lstrip("#").strip() if first.startswith("#") else doc_id
    return Document(doc_id, text, title)


def load_corpus(directory: str | Path | None = None) -> list[Document]:
    """Load ``*.md`` / ``*.txt`` files; doc_id is the file stem. Sorted for determinism."""
    if directory is None:
        root = resources.files("ragscope") / "data" / "corpus"
        files = sorted((p for p in root.iterdir() if p.name.endswith(".md")), key=lambda p: p.name)
        return [_doc_from_text(p.name.rsplit(".", 1)[0], p.read_text(encoding="utf-8")) for p in files]
    path = Path(directory)
    files = sorted(p for p in path.iterdir() if p.suffix in (".md", ".txt"))
    if not files:
        raise FileNotFoundError(f"no .md or .txt files in {path}")
    return [_doc_from_text(p.stem, p.read_text(encoding="utf-8")) for p in files]


def load_qa(path: str | Path | None = None) -> list[QAItem]:
    """Load a JSON list (or JSONL) of {id, question, answer, evidence: [{doc_id, text}]}."""
    if path is None:
        raw = (resources.files("ragscope") / "data" / "qa.json").read_text(encoding="utf-8")
    else:
        raw = Path(path).read_text(encoding="utf-8")
    stripped = raw.lstrip()
    rows = json.loads(raw) if stripped.startswith("[") else [json.loads(l) for l in raw.splitlines() if l.strip()]
    return [
        QAItem(
            str(r["id"]),
            r["question"],
            r.get("answer", ""),
            tuple(Evidence(e["doc_id"], e["text"]) for e in r["evidence"]),
            tuple(r.get("tags", ())),
        )
        for r in rows
    ]


def filter_tag(qa: Iterable[QAItem], tag: str | None) -> list[QAItem]:
    return [q for q in qa if tag is None or tag in q.tags]


def validate(docs: Iterable[Document], qa: Iterable[QAItem]) -> list[str]:
    """Return problems (evidence not found verbatim in its document). Empty = OK."""
    by_id = {d.doc_id: normalize(d.text) for d in docs}
    problems = []
    for item in qa:
        for ev in item.evidence:
            if ev.doc_id not in by_id:
                problems.append(f"{item.qid}: unknown doc {ev.doc_id!r}")
            elif normalize(ev.text) not in by_id[ev.doc_id]:
                problems.append(f"{item.qid}: evidence not found in {ev.doc_id!r}")
    return problems


def contains_evidence(chunk: Chunk, ev: Evidence) -> bool:
    return chunk.doc_id == ev.doc_id and normalize(ev.text) in normalize(chunk.text)
