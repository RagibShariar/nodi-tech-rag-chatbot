"""Cross-encoder reranker.

An embedding model (bi-encoder) reads the question and the chunk SEPARATELY and compares vectors.
A cross-encoder reads them TOGETHER and outputs one relevance score: more accurate, much slower.
So it only re-scores the top candidates from the fast searches, never the whole corpus.
"""

from typing import Callable

from src import config

# Any function (question, passages) -> one score per passage (higher = more relevant).
# Tests pass a fake one, so they don't need PyTorch or the 2.3 GB model.
RerankFn = Callable[[str, list[str]], list[float]]


class CrossEncoderReranker:
    def __init__(self, model_name: str | None = None):
        # Imported here, not at the top: it loads PyTorch (seconds, ~1 GB RAM), which the
        # vector/bm25/hybrid modes don't need.
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name or config.RERANK_MODEL, max_length=512)

    def __call__(self, question: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        return [float(score) for score in self.model.predict([(question, p) for p in passages])]
