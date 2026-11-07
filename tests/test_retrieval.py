import math

import numpy as np
import pytest

from ragscope import BM25Retriever, Chunk, DenseRetriever, HashingEmbedder, HybridRetriever
from ragscope.bm25 import BM25Index
from ragscope.fusion import reciprocal_rank_fusion

TEXTS = [
    "the cat sat on the mat",
    "dogs chase cats in the park",
    "raft elects a leader with a majority vote",
    "kafka partitions preserve ordering per key",
]
CHUNKS = [Chunk(f"c{i}", f"d{i}", t, 0, len(t)) for i, t in enumerate(TEXTS)]


def test_bm25_matches_hand_computation():
    idx = BM25Index(k1=1.5, b=0.75).fit(["apple banana", "apple apple cherry", "cherry"])
    # docs (after tokenize): [apple, banana] [apple, apple, cherry] [cherry]; avgdl = 2
    n, df = 3, 2
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    d0 = idf * 1 * 2.5 / (1 + 1.5 * (1 - 0.75 + 0.75 * 2 / 2))
    d1 = idf * 2 * 2.5 / (2 + 1.5 * (1 - 0.75 + 0.75 * 3 / 2))
    scores = idx.scores("apple")
    assert scores[0] == pytest.approx(d0)
    assert scores[1] == pytest.approx(d1)
    assert 2 not in scores  # sparse: non-matching docs are absent
    assert scores[1] > scores[0]  # more occurrences wins despite longer doc


def test_bm25_idf_never_negative_for_ubiquitous_terms():
    idx = BM25Index().fit(["common a", "common b", "common c"])
    assert idx.idf["common"] > 0


def test_bm25_repeated_query_terms_count_once():
    idx = BM25Index().fit(TEXTS)
    assert idx.scores("raft") == idx.scores("raft raft raft")


def test_bm25_b_zero_ignores_length():
    idx = BM25Index(b=0.0).fit(["x", "x y z w v"])
    s = idx.scores("x")
    assert s[0] == pytest.approx(s[1])


def test_bm25_tie_break_is_deterministic():
    idx = BM25Index().fit(["same text", "same text", "same text"])
    assert [d for d, _ in idx.top_k("same", 3)] == [0, 1, 2]


def test_bm25_retriever_finds_lexical_match_and_requires_index():
    r = BM25Retriever()
    with pytest.raises(RuntimeError):
        r.search("x")
    r.index(CHUNKS)
    assert r.search("leader majority", 1)[0].chunk.chunk_id == "c2"
    assert r.search("zzz unknownterm", 3) == []


def test_hashing_embedder_is_deterministic_and_normalised():
    a, b = HashingEmbedder(dim=512), HashingEmbedder(dim=512)
    a.fit(TEXTS), b.fit(TEXTS)
    va, vb = a.embed(TEXTS), b.embed(TEXTS)
    assert np.array_equal(va, vb)
    assert np.allclose(np.linalg.norm(va, axis=1), 1.0, atol=1e-5)
    assert va.dtype == np.float32 and va.shape == (4, 512)


def test_hashing_embedder_char_ngrams_bridge_morphology():
    emb = HashingEmbedder(dim=2048)
    emb.fit(["replication of logs", "bananas are yellow"])
    q, close, far = emb.embed(["replicated", "replication of logs", "bananas are yellow"])
    assert float(q @ close) > float(q @ far)
    assert float(q @ close) > 0.2  # whole-word matching would give exactly 0


def test_hashing_embedder_empty_text_is_zero_not_nan():
    emb = HashingEmbedder(dim=64)
    v = emb.embed([""])
    assert not np.isnan(v).any() and not v.any()


def test_dense_retriever_ranks_by_cosine():
    r = DenseRetriever(HashingEmbedder(dim=1024))
    r.index(CHUNKS)
    hits = r.search("partitioned kafka ordering", 2)
    assert hits[0].chunk.chunk_id == "c3"
    assert hits[0].score >= hits[1].score
    assert len(r.search("anything", 99)) == len(CHUNKS)


def test_rrf_matches_formula_and_breaks_ties_by_first_seen():
    fused = dict(reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60))
    assert fused["a"] == pytest.approx(1 / 61 + 1 / 62)
    assert fused["d"] == pytest.approx(1 / 63)
    order = [key for key, _ in reciprocal_rank_fusion([["a", "b"], ["b", "a"]])]
    assert order == ["a", "b"]


def test_rrf_consensus_beats_single_top_rank():
    # ranked 2nd by both beats ranked 1st by one and absent from the other
    order = [k for k, _ in reciprocal_rank_fusion([["x", "both"], ["y", "both"]])]
    assert order[0] == "both"


def test_rrf_weights_and_validation():
    order = [k for k, _ in reciprocal_rank_fusion([["a"], ["b"]], weights=[1.0, 2.0])]
    assert order == ["b", "a"]
    with pytest.raises(ValueError):
        reciprocal_rank_fusion([["a"]], weights=[1.0, 2.0])


def test_hybrid_returns_union_of_signals():
    h = HybridRetriever(dense=DenseRetriever(HashingEmbedder(dim=1024)))
    h.index(CHUNKS)
    hits = h.search("cat leader", 4)
    ids = [x.chunk.chunk_id for x in hits]
    assert {"c0", "c2"} <= set(ids[:3])
    assert all(x.source == "rrf" for x in hits)
