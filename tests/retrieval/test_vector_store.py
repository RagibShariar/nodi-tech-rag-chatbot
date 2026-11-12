"""Vector store tests with a FAKE embedder and an in-memory Qdrant: fast, no Ollama needed."""

import re
import uuid
import zlib

import pytest
from qdrant_client import QdrantClient

from src import config
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all
from src.retrieval.embed import Embedding
from src.retrieval.vector_store import build_index, collection_name, point_id, search

DIM = 1024  # with only 64, many words hash into the same slot and rankings become noise


def fake_embed(text: str, kind: str) -> Embedding:
    """Bag-of-words vector: texts sharing words get similar vectors. Deterministic (crc32, not hash())."""
    vector = [0.0] * DIM
    vector[0] = 0.1  # never an all-zero vector
    words = re.findall(r"\w+", text.lower())  # words without punctuation
    for word in words:
        vector[1 + zlib.crc32(word.encode()) % (DIM - 1)] += 1.0
    return Embedding(vector=vector, tokens=len(words))


@pytest.fixture(scope="module")
def chunks():
    return chunk_all(load_all(config.DATA_DIR))


@pytest.fixture
def client():
    c = QdrantClient(":memory:")
    yield c
    c.close()


def test_index_all_chunks(client, chunks):
    tokens = build_index(client, "test", chunks, fake_embed)
    assert client.count("test").count == len(chunks) == 50
    assert len(tokens) == 50

    [point] = client.retrieve("test", ids=[point_id("employee_handbook_v3.md#4")], with_payload=True)
    assert point.payload["section"] == "3. Leave > Carry forward"
    assert {"chunk_id", "source", "content", "status", "tokens"} <= point.payload.keys()


def test_reindex_does_not_duplicate(client, chunks):
    build_index(client, "test", chunks, fake_embed)
    build_index(client, "test", chunks, fake_embed)
    assert client.count("test").count == 50


def test_rebuild_removes_stale_points(client, chunks):
    build_index(client, "test", chunks[:3], fake_embed)
    build_index(client, "test", chunks[:2], fake_embed)
    assert client.count("test").count == 2


def test_search_returns_k_sorted_hits(client, chunks):
    build_index(client, "test", chunks, fake_embed)
    # Words that only appear in the handbook's carry-forward chunk.
    hits = search(client, "test", "used by 31 March otherwise they lapse", fake_embed, k=5)

    assert len(hits) == 5
    assert [h.score for h in hits] == sorted((h.score for h in hits), reverse=True)
    assert hits[0].chunk_id == "employee_handbook_v3.md#4"


def test_point_id_is_deterministic_uuid():
    assert point_id("a.md#0") == point_id("a.md#0")
    assert point_id("a.md#0") != point_id("a.md#1")
    uuid.UUID(point_id("a.md#0"))  # raises if not a valid UUID


def test_collection_name_per_model():
    assert collection_name("bge-m3") == "chunks_bge-m3"
    assert collection_name("qwen3-embedding:0.6b") == "chunks_qwen3-embedding_0.6b"
