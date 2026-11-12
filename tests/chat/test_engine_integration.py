"""End-to-end with the real Ollama chat model (skipped if it isn't pulled). Slow: ~1 minute on CPU.

Uses vector-only retrieval to skip the 20s reranker load; the full pipeline is covered by
scripts/run_conversations.py.
"""

import httpx
import pytest

from src import config
from scripts.run_conversations import check


def _ollama_has(model: str) -> bool:
    try:
        tags = httpx.get(f"{config.OLLAMA_URL}/api/tags", timeout=3).json()
    except httpx.HTTPError:
        return False
    return any(m["name"] == model for m in tags.get("models", []))


pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(not (_ollama_has(config.CHAT_MODEL) and _ollama_has("bge-m3:latest")),
                       reason=f"Ollama with {config.CHAT_MODEL} and bge-m3 not available"),
]


def test_annual_leave_is_18_with_citation():
    from src.chat.engine import create_engine

    engine = create_engine(mode="vector")
    events = list(engine.ask("integration", "How many days of annual leave do I get?"))
    answer = events[-1]["answer"]
    sources = {s["source"] for s in next(e for e in events if e["type"] == "sources")["sources"]}

    result = check(("C1", 1), answer, sources)
    assert result["facts_ok"], answer                      # contains 18; 15 only if called outdated
    assert "employee_handbook_v3.md" in result["cited"], answer
