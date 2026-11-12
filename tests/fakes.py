"""Fake models for fast unit tests (no Ollama, no PyTorch)."""

import re
import zlib

from src.retrieval.embed import Embedding

FAKE_DIM = 1024  # with only 64, many words hash into the same slot and rankings become noise


def fake_embed(text: str, kind: str) -> Embedding:
    """Bag-of-words vector: texts sharing words get similar vectors. Deterministic (crc32, not hash())."""
    vector = [0.0] * FAKE_DIM
    vector[0] = 0.1  # never an all-zero vector
    words = re.findall(r"\w+", text.lower())  # words without punctuation
    for word in words:
        vector[1 + zlib.crc32(word.encode()) % (FAKE_DIM - 1)] += 1.0
    return Embedding(vector=vector, tokens=len(words))


def fake_rerank_by_length(question: str, passages: list[str]) -> list[float]:
    """Scores = passage length, so 'longest first'. Easy to predict in tests."""
    return [float(len(p)) for p in passages]
