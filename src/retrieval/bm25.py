"""BM25 keyword search over chunks (in memory).

BM25 scores a chunk higher when it contains the query's words, especially RARE words
(IDF: "nd-1042" appears in 2 chunks, "what" in many), with diminishing returns for
repeats and a penalty for very long chunks.
"""

import re

from rank_bm25 import BM25Okapi

from src.ingest.chunk import Chunk

# One "word": lowercase letters, digits, or anything in the Bangla Unicode block (U+0980–U+09FF).
# Python's \w does NOT match Bangla vowel signs (া, ি, ...) so \w+ would cut Bangla words apart.
_WORD = r"[0-9a-zঀ-৿]+"
# Words joined by hyphens stay together: "nd-1042", "x-nodi-key", "hmac-sha256".
_TOKEN_RE = re.compile(rf"{_WORD}(?:-{_WORD})*")


def tokenize(text: str) -> list[str]:
    """'Error ND-1042!' -> ['error', 'nd-1042', 'nd', '1042']

    Hyphenated tokens are kept whole AND split, so "ND-1042", "nd 1042" and "1042" all match.
    """
    tokens: list[str] = []
    for token in _TOKEN_RE.findall(text.lower()):
        tokens.append(token)
        if "-" in token:
            tokens.extend(token.split("-"))
    return tokens


class BM25Index:
    def __init__(self, chunks: list[Chunk]):
        self.chunks = chunks
        # Same text as vector search: context line + content.
        self._bm25 = BM25Okapi([tokenize(chunk.embed_text) for chunk in chunks])

    def search(self, query: str, k: int) -> list[tuple[Chunk, float]]:
        """Top-k (chunk, score), best first. Chunks with score 0 (no shared words) are left out."""
        scores = self._bm25.get_scores(tokenize(query))
        ranked = sorted(zip(self.chunks, scores), key=lambda pair: (-pair[1], pair[0].chunk_id))
        return [(chunk, float(score)) for chunk, score in ranked[:k] if score > 0]
