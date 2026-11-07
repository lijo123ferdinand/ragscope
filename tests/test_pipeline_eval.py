import json

import pytest

from ragscope import BM25Retriever, OverlapReranker, RAGPipeline, ScoredChunk, Chunk
from ragscope.cli import main
from ragscope.dataset import contains_evidence, filter_tag, load_corpus, load_qa, validate, Evidence
from ragscope.evaluation import EvalConfig, default_grid, evaluate_config, format_table, run_eval, to_json

DOCS = load_corpus()
QA = load_qa()


def test_bundled_dataset_is_consistent():
    assert len(DOCS) == 14 and len(QA) == 70
    assert validate(DOCS, QA) == []
    assert {t for q in QA for t in q.tags} == {"lexical", "paraphrase"}
    assert len(filter_tag(QA, "paraphrase")) == 20
    assert len({q.qid for q in QA}) == len(QA)


def test_validate_flags_bad_evidence():
    bad = [type(QA[0])("x", "q", "a", (Evidence("raft", "this sentence is not in the doc"),))]
    assert validate(DOCS, bad) == ["x: evidence not found in 'raft'"]


def test_contains_evidence_is_whitespace_and_case_insensitive():
    c = Chunk("raft#0", "raft", "A  Cluster of\nfive servers", 0, 1)
    assert contains_evidence(c, Evidence("raft", "a cluster of five servers"))
    assert not contains_evidence(c, Evidence("other", "a cluster of five servers"))


def test_pipeline_answers_with_traceable_citation():
    pipe = RAGPipeline().index(DOCS)
    ans = pipe.answer("What is the in-memory buffer of an LSM tree called?")
    assert "memtable" in ans.text
    cited = ans.citations[0].chunk
    assert cited.doc_id == "lsm_tree"
    assert pipe.documents[cited.doc_id].text[cited.start : cited.end] == cited.text


def test_pipeline_rejects_duplicate_doc_ids():
    with pytest.raises(ValueError):
        RAGPipeline().index([DOCS[0], DOCS[0]])


def test_reranker_reorders_over_fetched_pool():
    pipe = RAGPipeline(retriever=BM25Retriever(), reranker=OverlapReranker(), candidates=15).index(DOCS)
    hits = pipe.retrieve("Which PostgreSQL process cleans up dead tuples?", 3)
    assert len(hits) == 3 and all(h.source == "rerank" for h in hits)
    assert hits[0].chunk.doc_id == "mvcc"


def test_overlap_reranker_prefers_full_coverage():
    a = ScoredChunk(Chunk("a", "d", "raft raft raft", 0, 1), 9.0)
    b = ScoredChunk(Chunk("b", "d", "raft leader election", 0, 1), 1.0)
    assert OverlapReranker().rerank("raft leader election", [a, b], 2)[0].chunk.chunk_id == "b"


def test_eval_retrieval_metrics_are_sane_and_deterministic():
    cfg = EvalConfig("sentence", "hybrid")
    r1 = evaluate_config(cfg, DOCS, QA, k=5)
    r2 = evaluate_config(cfg, DOCS, QA, k=5)
    for attr in ("recall_1", "recall_k", "mrr", "ndcg_k", "faithfulness", "answer_relevance", "ref_hit"):
        v = getattr(r1, attr)
        assert 0.0 <= v <= 1.0
        assert v == getattr(r2, attr)
    assert r1.recall_1 <= r1.recall_k
    assert r1.recall_k > 0.8  # regression guard on the bundled benchmark
    assert r1.judge_failures == 0 and r1.invalid_citations == 0


def test_lexical_questions_are_easier_than_paraphrases():
    cfg = EvalConfig("recursive", "bm25")
    lex = evaluate_config(cfg, DOCS, filter_tag(QA, "lexical"), generate=False)
    para = evaluate_config(cfg, DOCS, filter_tag(QA, "paraphrase"), generate=False)
    assert lex.mrr > para.mrr


def test_grid_table_and_json():
    configs = default_grid(("fixed",), ("bm25", "dense"), ("none", "overlap"))
    assert [c.label for c in configs] == ["bm25/fixed", "dense/fixed", "bm25/fixed+overlap", "dense/fixed+overlap"]
    results = run_eval(DOCS, QA[:5], configs, generate=False)
    table = format_table(results)
    assert "bm25/fixed+overlap" in table and "R@5" in table
    md = format_table(results, fmt="markdown")
    assert md.splitlines()[1].startswith("|---|")
    payload = json.loads(to_json(results))
    assert payload[0]["config"]["label"] == "bm25/fixed" and "per_question" not in payload[0]


def test_cli_eval_ask_chunks(capsys, tmp_path):
    out = tmp_path / "r.json"
    assert main(["eval", "--chunkers", "sentence", "--retrievers", "bm25", "--tag", "lexical", "--out", str(out)]) == 0
    text = capsys.readouterr().out
    assert "bm25/sentence" in text and "best by MRR" in text
    assert json.loads(out.read_text())[0]["per_question"]

    assert main(["ask", "Which HTTP status code signals that a client is being throttled?"]) == 0
    text = capsys.readouterr().out
    assert "429" in text and "rate_limiting chars" in text

    assert main(["chunks", "raft", "--chunker", "fixed"]) == 0
    assert "--- raft#0" in capsys.readouterr().out
    assert main(["chunks", "nope"]) == 2
    assert main(["eval", "--tag", "nosuchtag"]) == 2
