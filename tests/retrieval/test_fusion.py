import pytest

from src.retrieval.fusion import rrf


def test_rrf_hand_computed():
    vector = ["a", "b", "c"]
    bm25 = ["b", "d"]
    fused = dict(rrf([vector, bm25], k=60))

    assert fused["a"] == pytest.approx(1 / 61)             # only in vector, rank 1
    assert fused["b"] == pytest.approx(1 / 62 + 1 / 61)    # vector rank 2 + bm25 rank 1
    assert fused["c"] == pytest.approx(1 / 63)
    assert fused["d"] == pytest.approx(1 / 62)


def test_rrf_order_rewards_agreement():
    # "b" is 2nd in one list and 1st in the other: found by both methods -> wins.
    order = [item for item, _ in rrf([["a", "b", "c"], ["b", "d"]])]
    assert order == ["b", "a", "d", "c"]


def test_item_in_one_list_is_kept():
    # e.g. a Bangla chunk found by vector search only (BM25 can't match across languages)
    order = [item for item, _ in rrf([["bangla"], []])]
    assert order == ["bangla"]


def test_ties_broken_by_id():
    order = [item for item, _ in rrf([["z"], ["a"]])]  # both 1/61
    assert order == ["a", "z"]
