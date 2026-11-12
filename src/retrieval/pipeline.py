"""Retrieval pipeline: vector + BM25 -> RRF -> rerank -> top k.

Modes (so each stage can be measured on its own):
  vector : vector search only (M2 baseline)
  bm25   : BM25 only
  hybrid : vector + BM25 fused with RRF
  rerank : hybrid, then the cross-encoder re-scores the top CANDIDATES (full pipeline)
"""

from dataclasses import dataclass
from typing import Literal

from qdrant_client import QdrantClient

from src import config
from src.ingest.chunk import Chunk
from src.retrieval.bm25 import BM25Index
from src.retrieval.fusion import rrf
from src.retrieval.rerank import RerankFn
from src.retrieval.vector_store import EmbedFn, search

Mode = Literal["vector", "bm25", "hybrid", "rerank"]
MODES: tuple[Mode, ...] = ("vector", "bm25", "hybrid", "rerank")


@dataclass(frozen=True)
class Retrieved:
    chunk: Chunk
    score: float                 # the score that decided the order in this mode
    # Debug trail: where this chunk stood at each stage (None = stage not run / not found there)
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rrf_score: float | None = None
    rerank_score: float | None = None


class Retriever:
    def __init__(
        self,
        chunks: list[Chunk],
        client: QdrantClient,
        collection: str,
        embed_fn: EmbedFn,
        rerank_fn: RerankFn | None = None,
        candidates: int = config.CANDIDATES,
        search_depth: int = config.SEARCH_DEPTH,
    ):
        self.chunks = {chunk.chunk_id: chunk for chunk in chunks}
        self.bm25 = BM25Index(chunks)
        self.client = client
        self.collection = collection
        self.embed_fn = embed_fn
        self.rerank_fn = rerank_fn
        self.candidates = candidates
        self.search_depth = search_depth

    def retrieve(self, question: str, mode: Mode = "rerank", k: int = 5) -> list[Retrieved]:
        # How deep each search goes is independent of how many candidates get reranked:
        # changing CANDIDATES must not change what vector/BM25 find.
        n = max(self.search_depth, self.candidates, k)

        # 1. The two fast searches (only the ones this mode needs).
        vector: list[tuple[str, float]] = []
        bm25: list[tuple[str, float]] = []
        if mode != "bm25":
            vector = [(h.chunk_id, h.score)
                      for h in search(self.client, self.collection, question, self.embed_fn, k=n)]
        if mode != "vector":
            bm25 = [(chunk.chunk_id, score) for chunk, score in self.bm25.search(question, k=n)]

        vector_rank = {cid: rank for rank, (cid, _) in enumerate(vector, start=1)}
        bm25_rank = {cid: rank for rank, (cid, _) in enumerate(bm25, start=1)}

        # 2. Decide the order.
        rrf_scores: dict[str, float] = {}
        rerank_scores: dict[str, float] = {}
        if mode == "vector":
            ordered = vector
        elif mode == "bm25":
            ordered = bm25
        else:
            fused = rrf([[cid for cid, _ in vector], [cid for cid, _ in bm25]])
            rrf_scores = dict(fused)
            if mode == "hybrid":
                ordered = fused
            else:
                if self.rerank_fn is None:
                    raise ValueError("mode 'rerank' needs a rerank_fn")
                candidates = [cid for cid, _ in fused[: self.candidates]]
                scores = self.rerank_fn(question, [self.chunks[cid].embed_text for cid in candidates])
                rerank_scores = dict(zip(candidates, scores))
                ordered = sorted(rerank_scores.items(), key=lambda pair: (-pair[1], pair[0]))

        # 3. Top k, with the debug trail.
        return [
            Retrieved(
                chunk=self.chunks[cid],
                score=score,
                vector_rank=vector_rank.get(cid),
                bm25_rank=bm25_rank.get(cid),
                rrf_score=rrf_scores.get(cid),
                rerank_score=rerank_scores.get(cid),
            )
            for cid, score in ordered[:k]
        ]
