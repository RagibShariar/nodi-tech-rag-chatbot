import httpx
import pytest

from src import config
from src.retrieval.embed import cosine_similarity, embed, with_prefix

# ---------- Unit tests (no Ollama needed) ----------


def test_prefixes_per_model():
    assert with_prefix("hi", "bge-m3", "query") == "hi"
    assert with_prefix("hi", "nomic-embed-text", "query") == "search_query: hi"
    assert with_prefix("hi", "nomic-embed-text", "document") == "search_document: hi"
    assert with_prefix("hi", "qwen3-embedding:0.6b", "query").startswith("Instruct: ")
    assert with_prefix("hi", "qwen3-embedding:0.6b", "query").endswith("\nQuery: hi")
    assert with_prefix("hi", "qwen3-embedding:0.6b", "document") == "hi"


def test_cosine_similarity_basics():
    assert cosine_similarity([1, 0], [2, 0]) == pytest.approx(1.0)   # same direction, different length
    assert cosine_similarity([1, 0], [0, 3]) == pytest.approx(0.0)   # perpendicular
    assert cosine_similarity([1, 0], [-1, 0]) == pytest.approx(-1.0)  # opposite


# ---------- Integration tests (real Ollama; skipped if it isn't running) ----------


def _ollama_has(model: str) -> bool:
    try:
        tags = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).json()
    except httpx.HTTPError:
        return False
    return any(m["name"].split(":")[0] == model.split(":")[0] for m in tags.get("models", []))


needs_bge_m3 = pytest.mark.skipif(not _ollama_has("bge-m3"), reason="Ollama with bge-m3 not available")


@pytest.mark.integration
@needs_bge_m3
def test_bge_m3_vector_shape_and_tokens():
    e = embed("Annual leave is 18 days.", "document", model="bge-m3")
    assert len(e.vector) == 1024
    assert e.tokens > 0


@pytest.mark.integration
@needs_bge_m3
def test_same_text_same_vector():
    a = embed("Annual leave is 18 days.", "document", model="bge-m3").vector
    b = embed("Annual leave is 18 days.", "document", model="bge-m3").vector
    assert cosine_similarity(a, b) == pytest.approx(1.0, abs=1e-6)


@pytest.mark.integration
@needs_bge_m3
def test_similar_meaning_scores_higher():
    q = embed("How many days of annual leave do I get?", "query", model="bge-m3").vector
    leave = embed("Annual leave: 18 days per year.", "document", model="bge-m3").vector
    printer = embed("Printer maintenance is on the 15th of every month.", "document", model="bge-m3").vector
    assert cosine_similarity(q, leave) > cosine_similarity(q, printer)
