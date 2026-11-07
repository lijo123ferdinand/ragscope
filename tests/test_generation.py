import json

from ragscope import Chunk, MockProvider, ScoredChunk, generate_answer
from ragscope.generate import parse_citations
from ragscope.judge import answer_relevance, extract_json, faithfulness
from ragscope.llm import IDK
from ragscope import prompts


def ctx(*texts):
    return [ScoredChunk(Chunk(f"c{i}", f"d{i}", t, 0, len(t)), 1.0) for i, t in enumerate(texts)]


class Scripted:
    """A provider that returns canned text, to test parsing of real-LLM-shaped output."""

    name = "scripted"

    def __init__(self, *replies):
        self.replies = list(replies)
        self.prompts = []

    def complete(self, system, prompt, *, max_tokens=512):
        self.prompts.append((system, prompt))
        return self.replies.pop(0)


def test_parse_citations_formats():
    assert parse_citations("A [2]. B [1][2]. C [3, 1].") == [2, 1, 3]
    assert parse_citations("no cites") == []


def test_mock_answer_is_extractive_and_cited():
    contexts = ctx("Bananas are yellow. Apples are red.", "Raft elects a leader by majority vote.")
    ans = generate_answer("How does Raft elect a leader?", contexts, MockProvider())
    assert "Raft elects a leader by majority vote" in ans.text
    assert [c.n for c in ans.citations] == [2]
    assert ans.citations[0].chunk.doc_id == "d1"
    assert ans.invalid_citations == []


def test_mock_is_deterministic():
    contexts = ctx("Alpha beta gamma. Delta epsilon.", "Gamma delta.")
    a = generate_answer("gamma delta?", contexts, MockProvider()).text
    b = generate_answer("gamma delta?", contexts, MockProvider()).text
    assert a == b


def test_mock_refuses_when_nothing_overlaps():
    ans = generate_answer("quantum chromodynamics?", ctx("Bananas are yellow."), MockProvider())
    assert ans.is_refusal and ans.citations == []


def test_no_contexts_skips_llm_call():
    p = MockProvider()
    assert generate_answer("q", [], p).is_refusal
    assert p.calls == 0


def test_hallucinated_citation_is_flagged():
    ans = generate_answer("q", ctx("one", "two"), Scripted("It is so [1][7]."))
    assert [c.n for c in ans.citations] == [1]
    assert ans.invalid_citations == [7]


def test_prompt_wraps_sources_and_roundtrips():
    p = Scripted("x [1].")
    generate_answer("What?", ctx("text with [3] brackets\n\nand blank lines"), p)
    system, prompt = p.prompts[0]
    assert prompts.task_of(system) == "answer"
    srcs = prompts.parse_sources(prompt)
    assert srcs[0].n == 1 and "blank lines" in srcs[0].text


def test_faithfulness_mock_detects_unsupported_claim():
    contexts = ctx("Raft uses a majority vote to elect a leader.")
    good = generate_answer("How does Raft elect a leader?", contexts, MockProvider())
    assert faithfulness(good, MockProvider()).score == 1.0
    bad = generate_answer("q", contexts, Scripted("Raft uses a majority vote [1]. Bananas orbit Jupiter daily [1]."))
    res = faithfulness(bad, MockProvider())
    assert res.score == 0.5
    assert [c["supported"] for c in res.claims] == [True, False]


def test_faithfulness_refusal_is_excluded_not_scored():
    ans = generate_answer("zzz?", ctx("Bananas."), MockProvider())
    assert faithfulness(ans, MockProvider()).score is None


def test_faithfulness_parses_fenced_llm_json():
    ans = generate_answer("q", ctx("a b c"), Scripted("A [1]."))
    judge = Scripted('Sure!\n```json\n{"claims": [{"claim": "A", "supported": true}, {"claim": "B", "supported": false}]}\n```')
    assert faithfulness(ans, judge).score == 0.5


def test_faithfulness_reports_unparseable_judge():
    ans = generate_answer("q", ctx("a b c"), Scripted("A [1]."))
    res = faithfulness(ans, Scripted("I think it is fine."))
    assert res.score is None and res.error


def test_answer_relevance_mock_and_clamping():
    ans = generate_answer("How does Raft elect a leader?", ctx("Raft elects a leader by vote."), MockProvider())
    assert answer_relevance(ans, MockProvider()) == 1.0
    assert answer_relevance(ans, Scripted('{"score": 1.7}')) == 1.0
    assert answer_relevance(ans, Scripted('{"score": "high"}')) is None
    refusal = generate_answer("zzz?", ctx("Bananas."), MockProvider())
    assert refusal.text == IDK
    assert answer_relevance(refusal, MockProvider()) == 0.0


def test_extract_json_edge_cases():
    assert extract_json('noise {"a": 1} trailing') == {"a": 1}
    assert extract_json("[1, 2]") is None
    assert extract_json("{broken") is None
    assert json.loads(MockProvider().complete(prompts.RELEVANCE_SYSTEM, prompts.relevance_prompt("q", "")))["score"] == 0.0


def test_mock_dedupes_sentences_from_overlapping_chunks():
    same = "Kafka keeps records with the same key in one partition."
    ans = generate_answer("Which partition gets records with the same key?", ctx(same, "Other. " + same), MockProvider())
    assert ans.text.count("Kafka keeps") == 1
    assert [c.n for c in ans.citations] == [1, 2]
