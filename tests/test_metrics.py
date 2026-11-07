import math

import pytest

from ragscope.metrics import mean, ndcg_at_k, precision_at_k, recall_at_k, reciprocal_rank


def test_recall_and_precision():
    ranked = ["a", "b", "c", "d"]
    assert recall_at_k(ranked, {"b", "z"}, 2) == 0.5
    assert recall_at_k(ranked, set(), 2) == 0.0
    assert precision_at_k(ranked, {"a", "c"}, 4) == 0.5


def test_reciprocal_rank_with_cutoff():
    assert reciprocal_rank(["x", "y", "a"], {"a"}) == pytest.approx(1 / 3)
    assert reciprocal_rank(["x", "y", "a"], {"a"}, k=2) == 0.0


def test_ndcg_hand_computed():
    # relevant at ranks 2 and 3, two relevant overall
    got = ndcg_at_k(["x", "a", "b"], {"a", "b"}, 3)
    dcg = 1 / math.log2(3) + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3)
    assert got == pytest.approx(dcg / idcg)
    assert ndcg_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_ndcg_uses_index_wide_relevant_count():
    # one relevant retrieved at top but 3 exist: ideal would fill ranks 1-3
    assert ndcg_at_k(["a", "x", "y"], {"a"}, 3, n_relevant=3) < 1.0


def test_mean_skips_none():
    assert mean([1.0, None, 0.0]) == 0.5
    assert mean([None]) is None
