"""Pipeline tests with fakes: fake embedder, fake reranker, in-memory Qdrant."""

import pytest
from qdrant_client import QdrantClient

from src import config
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all
from src.retrieval import pipeline
from src.retrieval.fusion import rrf
from src.retrieval.pipeline import MODES, Retriever
from src.retrieval.vector_store import Hit, build_index
from tests.fakes import fake_embed, fake_rerank_by_length

QUESTION = "What does error ND-1042 mean?"


@pytest.fixture(scope="module")
def retriever():
    chunks = chunk_all(load_all(config.DATA_DIR))
    client = QdrantClient(":memory:")
    build_index(client, "test", chunks, fake_embed)
    yield Retriever(chunks, client, "test", fake_embed, rerank_fn=fake_rerank_by_length, candidates=20)
    client.close()


@pytest.mark.parametrize("mode", MODES)
def test_every_mode_returns_at_most_k(retriever, mode):
    results = retriever.retrieve(QUESTION, mode=mode, k=5)
    assert 0 < len(results) <= 5
    assert len({r.chunk.chunk_id for r in results}) == len(results)  # no duplicates


def test_trail_only_has_stages_that_ran(retriever):
    [v] = retriever.retrieve(QUESTION, mode="vector", k=1)
    assert v.vector_rank == 1 and v.bm25_rank is None and v.rrf_score is None and v.rerank_score is None

    [b] = retriever.retrieve(QUESTION, mode="bm25", k=1)
    assert b.bm25_rank == 1 and b.vector_rank is None


def test_trace_lists_each_search_and_does_not_change_results(retriever):
    plain = retriever.retrieve(QUESTION, mode="rerank", k=5)
    trace = {}
    traced = retriever.retrieve(QUESTION, mode="rerank", k=5, trace=trace)

    assert [r.chunk.chunk_id for r in traced] == [r.chunk.chunk_id for r in plain]
    assert set(trace) == {"vector", "bm25"}
    # BM25 only returns chunks sharing a word with the question, so there can be fewer than 5.
    ranks = [h["rank"] for h in trace["bm25"]]
    assert ranks == list(range(1, len(ranks) + 1)) and 0 < len(ranks) <= 5
    assert len(trace["vector"]) == 5
    top_bm25 = retriever.retrieve(QUESTION, mode="bm25", k=1)[0].chunk
    assert (trace["bm25"][0]["source"], trace["bm25"][0]["section"]) == (top_bm25.source, top_bm25.section)


def test_trace_only_has_searches_that_ran(retriever):
    trace = {}
    retriever.retrieve(QUESTION, mode="vector", k=5, trace=trace)
    assert trace["bm25"] == [] and len(trace["vector"]) == 5


def test_hybrid_order_is_rrf_of_vector_and_bm25(retriever):
    vector_ids = [r.chunk.chunk_id for r in retriever.retrieve(QUESTION, mode="vector", k=20)]
    bm25_ids = [r.chunk.chunk_id for r in retriever.retrieve(QUESTION, mode="bm25", k=20)]
    expected = [cid for cid, _ in rrf([vector_ids, bm25_ids])][:5]

    hybrid = retriever.retrieve(QUESTION, mode="hybrid", k=5)
    assert [r.chunk.chunk_id for r in hybrid] == expected
    assert all(r.rrf_score is not None for r in hybrid)


def test_rerank_order_comes_from_reranker(retriever):
    results = retriever.retrieve(QUESTION, mode="rerank", k=5)
    # The fake reranker scores by length, so results must be longest-first.
    lengths = [len(r.chunk.embed_text) for r in results]
    assert lengths == sorted(lengths, reverse=True)
    assert all(r.rerank_score is not None and r.rrf_score is not None for r in results)


def test_reranker_only_sees_the_candidates(retriever):
    # Every reranked result must come from the top-20 fused list, not the whole corpus.
    fused = {r.chunk.chunk_id for r in retriever.retrieve(QUESTION, mode="hybrid", k=20)}
    reranked = {r.chunk.chunk_id for r in retriever.retrieve(QUESTION, mode="rerank", k=5)}
    assert reranked <= fused


def test_candidates_do_not_change_search_depth(monkeypatch):
    # Regression test: CANDIDATES used to also limit how deep vector/BM25 searched.
    # Setup: chunk X is 6th in BOTH lists (outside each top 5), so RRF ranks it #1:
    # 1/66 + 1/66 = 0.030 > 1/61 = 0.016. Searching only 5 deep would lose X entirely.
    chunks = chunk_all(load_all(config.DATA_DIR))[:11]
    by_id = {c.chunk_id: c for c in chunks}
    ids = list(by_id)
    x = ids[10]
    vector_order = ids[0:5] + [x]
    bm25_order = ids[5:10] + [x]

    def fake_vector_search(client, collection, question, embed_fn, k):
        return [Hit(1.0, cid, by_id[cid].source, by_id[cid].section, by_id[cid].status, by_id[cid].content)
                for cid in vector_order[:k]]

    monkeypatch.setattr(pipeline, "search", fake_vector_search)
    keep_order = lambda question, passages: [-float(i) for i in range(len(passages))]  # noqa: E731
    r = Retriever(chunks, None, "t", fake_embed, keep_order, candidates=5, search_depth=20)
    r.bm25.search = lambda question, k: [(by_id[cid], 1.0) for cid in bm25_order[:k]]

    results = r.retrieve("anything", mode="rerank", k=5)
    assert results[0].chunk.chunk_id == x
    assert (results[0].vector_rank, results[0].bm25_rank) == (6, 6)


def test_rerank_mode_without_reranker_fails_clearly():
    chunks = chunk_all(load_all(config.DATA_DIR))[:3]
    client = QdrantClient(":memory:")
    build_index(client, "t", chunks, fake_embed)
    with pytest.raises(ValueError, match="rerank_fn"):
        Retriever(chunks, client, "t", fake_embed, rerank_fn=None).retrieve("leave", mode="rerank")
    client.close()
