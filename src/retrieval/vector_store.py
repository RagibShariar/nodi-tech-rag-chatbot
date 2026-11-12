"""Store chunk vectors in Qdrant and search them.

Postgres analogy: collection = table, point = row (id + vector + payload columns).
"""

import uuid
from dataclasses import asdict, dataclass
from typing import Callable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src import config
from src.ingest.chunk import Chunk
from src.retrieval.embed import Embedding, Kind

# Any function (text, "query"/"document") -> Embedding. The real one calls Ollama;
# tests pass a fake one, so they run without a model.
EmbedFn = Callable[[str, Kind], Embedding]


@dataclass(frozen=True)
class Hit:
    score: float  # cosine similarity, higher = more similar
    chunk_id: str
    source: str
    section: str
    status: str
    content: str


def get_client() -> QdrantClient:
    if config.QDRANT_URL:
        return QdrantClient(url=config.QDRANT_URL)  # a Qdrant server
    return QdrantClient(path=str(config.QDRANT_PATH))  # local mode: runs in-process, saves to a folder


def collection_name(model: str) -> str:
    # One collection per model, because vector sizes differ (bge-m3: 1024, nomic: 768).
    return "chunks_" + model.replace(":", "_")  # ":" is not allowed in collection names


def point_id(chunk_id: str) -> str:
    # Qdrant IDs must be integers or UUIDs. uuid5 is deterministic: same chunk_id -> same UUID.
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


def build_index(client: QdrantClient, collection: str, chunks: list[Chunk], embed_fn: EmbedFn) -> list[int]:
    """(Re)build the collection from scratch. Returns the token count of each chunk."""
    # Embed everything first: if the model fails halfway, the old index is still intact.
    embeddings = [embed_fn(chunk.embed_text, "document") for chunk in chunks]

    # Drop + recreate (approved in docs/specs/M2.md) so deleted/renamed chunks never linger.
    if client.collection_exists(collection):
        client.delete_collection(collection)
    client.create_collection(
        collection,
        vectors_config=VectorParams(size=len(embeddings[0].vector), distance=Distance.COSINE),
    )
    client.upsert(
        collection,
        points=[
            PointStruct(id=point_id(c.chunk_id), vector=e.vector, payload={**asdict(c), "tokens": e.tokens})
            for c, e in zip(chunks, embeddings)
        ],
    )
    return [e.tokens for e in embeddings]


def search(client: QdrantClient, collection: str, question: str, embed_fn: EmbedFn, k: int = 5) -> list[Hit]:
    query_vector = embed_fn(question, "query").vector
    result = client.query_points(collection, query=query_vector, limit=k, with_payload=True)
    return [
        Hit(
            score=point.score,
            chunk_id=point.payload["chunk_id"],
            source=point.payload["source"],
            section=point.payload["section"],
            status=point.payload["status"],
            content=point.payload["content"],
        )
        for point in result.points  # already sorted by score, highest first
    ]
