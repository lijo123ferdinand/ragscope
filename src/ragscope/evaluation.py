"""The evaluation harness: run a grid of pipeline configs over a QA set.

Retrieval metrics (per question, then macro-averaged):
* Recall@k  fraction of the question's evidence spans contained in some top-k chunk.
            Counted per *span*, not per chunk, so overlapping chunkers that place
            the same sentence in two chunks are not double-credited or penalised.
* MRR@10    1 / rank of the first relevant chunk.
* nDCG@k    binary-gain nDCG; the ideal assumes every relevant chunk in the index
            (overlap can make there be more than one) could fill the top ranks.

Generation metrics (on top-``gen_k`` contexts):
* Faith     LLM-judge faithfulness, refusals excluded.
* AnsRel    LLM-judge answer relevance.
* RefHit    fraction of answers containing the short reference answer verbatim
            (case-insensitive). Crude, but it needs no judge and cannot be gamed by
            a lenient one.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field

from .chunking import make_chunker
from .dataset import QAItem, contains_evidence
from .embedders import Embedder, make_embedder
from .judge import answer_relevance, faithfulness
from .llm import LLMProvider, MockProvider
from .metrics import mean, ndcg_at_k, reciprocal_rank
from .pipeline import RAGPipeline
from .rerank import make_reranker
from .retrievers import make_retriever
from .text import normalize
from .types import Document

MRR_DEPTH = 10


@dataclass(frozen=True)
class EvalConfig:
    chunker: str
    retriever: str
    reranker: str = "none"

    @property
    def label(self) -> str:
        base = f"{self.retriever}/{self.chunker}"
        return base if self.reranker == "none" else f"{base}+{self.reranker}"


@dataclass
class ConfigResult:
    config: EvalConfig
    n_chunks: int
    recall_1: float
    recall_k: float
    mrr: float
    ndcg_k: float
    faithfulness: float | None = None
    answer_relevance: float | None = None
    ref_hit: float | None = None
    judge_failures: int = 0
    invalid_citations: int = 0
    ms_per_query: float = 0.0
    per_question: list[dict] = field(default_factory=list, repr=False)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["config"] = {**asdict(self.config), "label": self.config.label}
        return d


def default_grid(
    chunkers: Sequence[str] = ("fixed", "recursive", "sentence"),
    retrievers: Sequence[str] = ("bm25", "dense", "hybrid"),
    rerankers: Sequence[str] = ("none",),
) -> list[EvalConfig]:
    return [EvalConfig(c, r, rr) for rr in rerankers for r in retrievers for c in chunkers]


def evaluate_config(
    config: EvalConfig,
    docs: Sequence[Document],
    qa: Sequence[QAItem],
    *,
    k: int = 5,
    chunk_size: int = 400,
    provider: LLMProvider | None = None,
    generate: bool = True,
    gen_k: int = 3,
    embedder_factory: Callable[[], Embedder] = lambda: make_embedder("hashing"),
) -> ConfigResult:
    # Fresh components per config: the embedder learns idf from *this* chunking.
    pipe = RAGPipeline(
        chunker=make_chunker(config.chunker, chunk_size),
        retriever=make_retriever(config.retriever, embedder_factory()),
        reranker=make_reranker(config.reranker),
        provider=provider or MockProvider(),
    ).index(docs)

    depth = max(k, MRR_DEPTH)
    rows: list[dict] = []
    elapsed = 0.0
    for item in qa:
        relevant_all = {c.chunk_id for c in pipe.chunks if any(contains_evidence(c, e) for e in item.evidence)}
        t0 = time.perf_counter()
        hits = pipe.retrieve(item.question, depth)
        elapsed += time.perf_counter() - t0
        ranked = [h.chunk.chunk_id for h in hits]

        def span_recall(cut: int) -> float:
            top = [h.chunk for h in hits[:cut]]
            return sum(any(contains_evidence(c, e) for c in top) for e in item.evidence) / len(item.evidence)

        row = {
            "qid": item.qid,
            "recall_1": span_recall(1),
            "recall_k": span_recall(k),
            "mrr": reciprocal_rank(ranked, relevant_all, MRR_DEPTH),
            "ndcg_k": ndcg_at_k(ranked, relevant_all, k),
            "top": ranked[:k],
        }
        if generate:
            ans = pipe.answer(item.question, gen_k)
            faith = faithfulness(ans, pipe.provider)
            rel = answer_relevance(ans, pipe.provider)
            row.update(
                answer=ans.text,
                faithfulness=faith.score,
                answer_relevance=rel,
                ref_hit=float(normalize(item.answer) in normalize(ans.text)) if item.answer else None,
                judge_failed=int(faith.error is not None) + int(rel is None),
                invalid_citations=len(ans.invalid_citations),
            )
        rows.append(row)

    def avg(key: str) -> float | None:
        return mean([r.get(key) for r in rows])

    return ConfigResult(
        config=config,
        n_chunks=len(pipe.chunks),
        recall_1=avg("recall_1") or 0.0,
        recall_k=avg("recall_k") or 0.0,
        mrr=avg("mrr") or 0.0,
        ndcg_k=avg("ndcg_k") or 0.0,
        faithfulness=avg("faithfulness") if generate else None,
        answer_relevance=avg("answer_relevance") if generate else None,
        ref_hit=avg("ref_hit") if generate else None,
        judge_failures=sum(r.get("judge_failed", 0) for r in rows),
        invalid_citations=sum(r.get("invalid_citations", 0) for r in rows),
        ms_per_query=1000 * elapsed / max(1, len(qa)),
        per_question=rows,
    )


def run_eval(
    docs: Sequence[Document], qa: Sequence[QAItem], configs: Sequence[EvalConfig], **kwargs
) -> list[ConfigResult]:
    return [evaluate_config(c, docs, qa, **kwargs) for c in configs]


def _fmt(v: float | None) -> str:
    return "  -  " if v is None else f"{v:.3f}"


def format_table(results: Sequence[ConfigResult], k: int = 5, fmt: str = "text") -> str:
    headers = ["config", "chunks", "R@1", f"R@{k}", "MRR@10", f"nDCG@{k}", "Faith", "AnsRel", "RefHit", "ms/q"]
    rows = [
        [
            r.config.label,
            str(r.n_chunks),
            _fmt(r.recall_1),
            _fmt(r.recall_k),
            _fmt(r.mrr),
            _fmt(r.ndcg_k),
            _fmt(r.faithfulness),
            _fmt(r.answer_relevance),
            _fmt(r.ref_hit),
            f"{r.ms_per_query:.2f}",
        ]
        for r in results
    ]
    if fmt == "markdown":
        lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|"]
        lines += ["| " + " | ".join(row) + " |" for row in rows]
        return "\n".join(lines)
    widths = [max(len(h), *(len(row[i]) for row in rows)) for i, h in enumerate(headers)]
    line = lambda cells: "  ".join(c.ljust(w) if i == 0 else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
    out = [line(headers), "  ".join("-" * w for w in widths)]
    out += [line(row) for row in rows]
    return "\n".join(out)


def to_json(results: Sequence[ConfigResult], include_per_question: bool = False) -> str:
    payload = []
    for r in results:
        d = r.to_dict()
        if not include_per_question:
            d.pop("per_question")
        payload.append(d)
    return json.dumps(payload, indent=2)
