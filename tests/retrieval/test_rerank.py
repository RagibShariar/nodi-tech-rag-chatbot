"""Integration tests with the real bge-reranker-v2-m3 (skipped if PyTorch or the model isn't available)."""

import importlib.util

import pytest

from src import config


def _reranker_available() -> bool:
    # Runs at collection time, even when these tests are deselected, so it must be cheap:
    # check that the package is installed WITHOUT importing it (importing loads PyTorch, ~30s).
    if importlib.util.find_spec("sentence_transformers") is None:
        return False
    from huggingface_hub import try_to_load_from_cache
    # Only run if the model is already downloaded: tests must never trigger a 2.3 GB download.
    return isinstance(try_to_load_from_cache(config.RERANK_MODEL, "config.json"), str)


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not _reranker_available(), reason="PyTorch or reranker model not available"),
]


@pytest.fixture(scope="module")
def reranker():
    from src.retrieval.rerank import CrossEncoderReranker
    try:
        return CrossEncoderReranker()
    except OSError as e:  # e.g. PyTorch DLLs can't load (missing VC++ runtime on Windows)
        pytest.skip(f"PyTorch could not load: {e}")


def test_relevant_passage_scores_higher(reranker):
    good, bad = reranker("How many days of annual leave do I get?", [
        "Annual leave: 18 days per year.",
        "Printer maintenance is on the 15th of every month.",
    ])
    assert good > bad


def test_cross_language_english_question_bangla_passage(reranker):
    bus_bn = "ফেরার বাস দুই রুটেই সন্ধ্যা ৬:১৫ মিনিটে অফিস থেকে ছাড়ে।"
    good, bad = reranker("What time does the return office bus leave?", [
        bus_bn,
        "Passwords must be at least 14 characters long.",
    ])
    assert good > bad


def test_empty_passages(reranker):
    assert reranker("anything", []) == []
