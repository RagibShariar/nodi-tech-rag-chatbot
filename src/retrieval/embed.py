"""Turn text into an embedding vector by calling Ollama's REST API.

POST {OLLAMA_URL}/api/embed  {"model": "bge-m3", "input": "some text"}
  -> {"embeddings": [[0.01, -0.03, ...]], "prompt_eval_count": 12}
"""

import math
from dataclasses import dataclass
from typing import Literal

import httpx

from src import config

Kind = Literal["query", "document"]

# Some models were trained with a prefix that says whether a text is a question or a document.
# Without the prefix they still work, but worse. Keyed by model name without the ":tag".
PREFIXES: dict[str, dict[Kind, str]] = {
    "nomic-embed-text": {"query": "search_query: ", "document": "search_document: "},
    "qwen3-embedding": {
        "query": "Instruct: Given a question about company documents, "
                 "retrieve passages that answer the question\nQuery: ",
        "document": "",
    },
    # bge-m3: no prefix needed
}


@dataclass(frozen=True)
class Embedding:
    vector: list[float]
    tokens: int  # how many tokens the model read (incl. prefix)


def with_prefix(text: str, model: str, kind: Kind) -> str:
    base_name = model.split(":")[0]  # "qwen3-embedding:0.6b" -> "qwen3-embedding"
    return PREFIXES.get(base_name, {}).get(kind, "") + text


def embed(text: str, kind: Kind, model: str | None = None) -> Embedding:
    model = model or config.EMBED_MODEL
    response = httpx.post(
        f"{config.OLLAMA_URL}/api/embed",
        json={"model": model, "input": with_prefix(text, model, kind)},
        timeout=120,  # first call loads the model into memory, which can take a while on CPU
    )
    response.raise_for_status()
    data = response.json()
    return Embedding(vector=data["embeddings"][0], tokens=data["prompt_eval_count"])


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """For inspection and tests only. Qdrant computes this itself during search.

    cos = (a · b) / (|a| * |b|): 1 = same direction, 0 = unrelated.
    """
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)
