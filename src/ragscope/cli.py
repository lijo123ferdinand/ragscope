"""``ragscope`` command line: ``eval`` (the comparison table), ``ask`` and ``chunks``."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence

from .chunking import CHUNKERS, make_chunker
from .dataset import filter_tag, load_corpus, load_qa, validate
from .embedders import make_embedder
from .evaluation import default_grid, format_table, run_eval, to_json
from .llm import make_provider
from .pipeline import RAGPipeline
from .rerank import make_reranker
from .retrievers import make_retriever


def _csv(value: str) -> list[str]:
    return [v.strip() for v in value.split(",") if v.strip()]


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--corpus", help="directory of .md/.txt documents (default: bundled corpus)")
    p.add_argument("--chunk-size", type=int, default=400, help="chunk size in characters (default 400)")
    p.add_argument("--embedder", default="hashing", choices=["hashing", "st"], help="dense embedder")
    p.add_argument("--provider", default="mock", choices=["mock", "anthropic", "openai", "ollama"])
    p.add_argument("--model", help="model name for the provider")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragscope", description="RAG pipeline + evaluation harness")
    sub = parser.add_subparsers(dest="command", required=True)

    ev = sub.add_parser("eval", help="compare chunkers x retrievers x rerankers on a QA set")
    _add_common(ev)
    ev.add_argument("--qa", help="QA set (JSON list or JSONL); default: bundled")
    ev.add_argument("--tag", help="only evaluate questions with this tag (bundled: lexical, paraphrase)")
    ev.add_argument("--k", type=int, default=5, help="cutoff for Recall@k and nDCG@k")
    ev.add_argument("--gen-k", type=int, default=3, help="contexts passed to the generator")
    ev.add_argument("--chunkers", type=_csv, default=["fixed", "recursive", "sentence"])
    ev.add_argument("--retrievers", type=_csv, default=["bm25", "dense", "hybrid"])
    ev.add_argument("--rerankers", type=_csv, default=["none"], help="e.g. none,overlap")
    ev.add_argument("--no-generate", action="store_true", help="retrieval metrics only")
    ev.add_argument("--format", choices=["text", "markdown", "json"], default="text")
    ev.add_argument("--out", help="also write full per-question JSON results here")

    ask = sub.add_parser("ask", help="answer one question with citations")
    _add_common(ask)
    ask.add_argument("question")
    ask.add_argument("--chunker", default="sentence", choices=sorted(CHUNKERS))
    ask.add_argument("--retriever", default="hybrid", choices=["bm25", "dense", "hybrid"])
    ask.add_argument("--reranker", default="none", choices=["none", "overlap", "cross-encoder"])
    ask.add_argument("--k", type=int, default=3)

    ch = sub.add_parser("chunks", help="show how a chunker splits one document")
    ch.add_argument("doc_id")
    ch.add_argument("--chunker", default="sentence", choices=sorted(CHUNKERS))
    ch.add_argument("--chunk-size", type=int, default=400)
    ch.add_argument("--corpus")
    return parser


def cmd_eval(args: argparse.Namespace) -> int:
    docs = load_corpus(args.corpus)
    qa = filter_tag(load_qa(args.qa), args.tag)
    if not qa:
        print(f"no questions with tag {args.tag!r}", file=sys.stderr)
        return 2
    problems = validate(docs, qa)
    if problems:
        print("QA set does not match corpus:\n  " + "\n  ".join(problems[:20]), file=sys.stderr)
        return 2
    configs = default_grid(args.chunkers, args.retrievers, args.rerankers)
    provider = make_provider(args.provider, args.model)
    results = run_eval(
        docs,
        qa,
        configs,
        k=args.k,
        chunk_size=args.chunk_size,
        provider=provider,
        generate=not args.no_generate,
        gen_k=args.gen_k,
        embedder_factory=lambda: make_embedder(args.embedder),
    )
    if args.format == "json":
        print(to_json(results))
    else:
        print(f"ragscope eval: {len(docs)} docs, {len(qa)} questions{f' (tag={args.tag})' if args.tag else ''}, provider={provider.name}, "
              f"embedder={args.embedder}, chunk_size={args.chunk_size}\n")
        print(format_table(results, k=args.k, fmt=args.format))
        best = max(results, key=lambda r: (r.mrr, r.recall_k))
        print(f"\nbest by MRR: {best.config.label} (MRR {best.mrr:.3f}, R@{args.k} {best.recall_k:.3f})")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            fh.write(to_json(results, include_per_question=True))
    return 0


def cmd_ask(args: argparse.Namespace) -> int:
    pipe = RAGPipeline(
        chunker=make_chunker(args.chunker, args.chunk_size),
        retriever=make_retriever(args.retriever, make_embedder(args.embedder)),
        reranker=make_reranker(args.reranker),
        provider=make_provider(args.provider, args.model),
    ).index(load_corpus(args.corpus))
    ans = pipe.answer(args.question, args.k)
    print(ans.text)
    print()
    for c in ans.citations:
        print(f"[{c.n}] {c.chunk.doc_id} chars {c.chunk.start}-{c.chunk.end}")
    if ans.invalid_citations:
        print(f"warning: answer cited non-existent sources {ans.invalid_citations}", file=sys.stderr)
    return 0


def cmd_chunks(args: argparse.Namespace) -> int:
    docs = {d.doc_id: d for d in load_corpus(args.corpus)}
    if args.doc_id not in docs:
        print(f"unknown doc {args.doc_id!r}; available: {', '.join(sorted(docs))}", file=sys.stderr)
        return 2
    for c in make_chunker(args.chunker, args.chunk_size).chunk(docs[args.doc_id]):
        print(f"--- {c.chunk_id} [{c.start}:{c.end}] {len(c.text)} chars")
        print(c.text)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = {"eval": cmd_eval, "ask": cmd_ask, "chunks": cmd_chunks}[args.command]
    return handler(args)


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
