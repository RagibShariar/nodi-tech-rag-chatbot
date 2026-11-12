"""Reciprocal Rank Fusion: merge several ranked lists into one.

score(item) = sum over lists of 1 / (k + rank_in_list)

Uses ranks, not raw scores, because BM25 scores (0 to ~20) and cosine scores (0–1) are on
different scales. An item missing from a list simply gets nothing from that list.
"""

from src import config


def rrf(rankings: list[list[str]], k: int = config.RRF_K) -> list[tuple[str, float]]:
    """rankings: lists of IDs, best first. Returns (id, rrf_score), best first; ties broken by ID."""
    # k = 60 is the standard value from the RRF paper (Cormack et al., 2009). Bigger k flattens
    # the difference between rank 1 and rank 10; smaller k rewards top ranks more.
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
