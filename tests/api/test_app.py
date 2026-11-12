"""API tests with a fake engine (FakeLLM + fake retriever): no Ollama, no reranker."""

import json

import pytest
from fastapi.testclient import TestClient

from src import config
from src.api.app import create_app
from src.chat.engine import ChatEngine
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all
from src.retrieval.pipeline import Retrieved
from tests.fakes import FakeLLM


@pytest.fixture(scope="module")
def chunk():
    return {c.chunk_id: c for c in chunk_all(load_all(config.DATA_DIR))}["employee_handbook_v3.md#3"]


def make_client(llm, chunk) -> TestClient:
    engine = ChatEngine(llm, lambda query: [Retrieved(chunk, 0.9)])
    return TestClient(create_app(engine))


def read_events(response) -> list[dict]:
    return [json.loads(line) for line in response.text.splitlines() if line.strip()]


def test_index_serves_the_page(chunk):
    with make_client(FakeLLM(), chunk) as client:
        r = client.get("/")
    assert r.status_code == 200
    assert "<title>Nodi Tech Assistant</title>" in r.text
    assert "escapeHtml" in r.text  # output is escaped before rendering (XSS guard)


def test_chat_streams_ndjson_events(chunk):
    with make_client(FakeLLM("You get 18 days [employee_handbook_v3.md]."), chunk) as client:
        r = client.post("/chat", json={"session_id": "s1", "message": "How many days of annual leave?"})
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("application/x-ndjson")
    events = read_events(r)
    assert [e["type"] for e in events][:2] == ["rewrite", "sources"]
    assert events[-1]["type"] == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "You get 18 days [employee_handbook_v3.md]."


def test_reset_clears_the_session(chunk):
    # After reset, the next question is a first turn again: no rewrite call (FakeLLM has no reply for one).
    with make_client(FakeLLM("A1", "A2"), chunk) as client:
        client.post("/chat", json={"session_id": "s1", "message": "Q1"})
        assert client.post("/reset", json={"session_id": "s1"}).json() == {"ok": True}
        events = read_events(client.post("/chat", json={"session_id": "s1", "message": "Q2"}))
    assert events[0]["rewritten"] is False
    assert events[-1]["answer"] == "A2"


def test_empty_message_rejected(chunk):
    with make_client(FakeLLM(), chunk) as client:
        assert client.post("/chat", json={"session_id": "s1", "message": ""}).status_code == 422


def test_failure_becomes_error_event(chunk):
    class BrokenLLM:
        def stream(self, messages):
            raise RuntimeError("Ollama is not running")
            yield  # makes this a generator

    with make_client(BrokenLLM(), chunk) as client:
        events = read_events(client.post("/chat", json={"session_id": "s1", "message": "hi"}))
    assert events[-1] == {"type": "error", "message": "Ollama is not running"}
