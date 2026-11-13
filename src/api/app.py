"""FastAPI app: the chat page, a streaming chat endpoint and a small info endpoint.

Run:  python -m uvicorn src.api.app:app --port 8000     (startup takes ~20s: loads the reranker)

POST /chat  {"session_id": "...", "message": "...", "scope": "all"}
  -> text/event-stream (Server-Sent Events): each event is "event: <name>\\ndata: <json>\\n\\n"
     event: delta    data: {"text": "..."}                       many, as the answer streams
     event: replace  data: {"text": "..."}                       only if the leak guard stopped the answer
     event: done     data: {"answer", "answered", "sources": [{"id", "file", "status", "section", "snippet"}],
                            "retrieved": [{"source", "section", "status", "vector_rank", "bm25_rank",
                                           "rrf_score", "rerank_score", "cited"}],   (all top-5 chunks)
                            "trace": {"vector": [...], "bm25": [...]}, "query", "rewritten", "timing"}
     event: error    data: {"message": "..."}                    if something fails mid-way
  History stays on the server, keyed by session_id (docs/decisions.md: the page never sends history, so a
  client can't forge earlier "assistant" turns).
POST /reset {"session_id": "..."}
POST /feedback {"session_id", "question", "answer", "rating": "up" | "down" | "none", "sources": [...]}
  -> appended as one JSON line to artifacts/feedback.jsonl (gitignored)
GET  /info  -> {"documents", "chunks", "retrieval_mode", "chat_model", "embed_model"}
"""

import json
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src import config
from src.chat.citations import cited_sources, is_answered
from src.chat.engine import ChatEngine

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)
    scope: Literal["all"] = "all"  # metadata filter: a stretch goal, only "all" for now


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)


class FeedbackRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    question: str = Field(min_length=1, max_length=2000)
    answer: str = Field(max_length=20000)
    rating: Literal["up", "down", "none"]  # "none" = the user took their rating back
    sources: list[str] = Field(default_factory=list, max_length=20)
    query: str = Field(default="", max_length=2000)


DEFAULT_FEEDBACK_FILE = config.ROOT / "artifacts" / "feedback.jsonl"


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


def corpus_info() -> dict:
    from src.ingest.chunk import chunk_all
    from src.ingest.parse import load_all
    chunks = chunk_all(load_all(config.DATA_DIR))
    return {"documents": len({c.source for c in chunks}), "chunks": len(chunks)}


def retrieved_view(retrieved: list[dict], cards: list[dict]) -> list[dict]:
    """The top-5 chunks for the page's 'Sources searched' list and debug table (no full text)."""
    cited = {c["file"] for c in cards}
    keys = ("source", "section", "status", "vector_rank", "bm25_rank", "rrf_score", "rerank_score")
    return [{**{k: item.get(k) for k in keys}, "cited": item["source"] in cited} for item in retrieved]


def create_app(engine: ChatEngine | None = None, info: dict | None = None,
               feedback_file: Path = DEFAULT_FEEDBACK_FILE) -> FastAPI:
    """Tests pass a fake engine, info and a temporary feedback file; the real ones are built at startup (slow)."""
    feedback_lock = threading.Lock()  # requests run in worker threads: one writer at a time

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
        else:
            from src.chat.engine import create_engine
            app.state.engine = create_engine()
        app.state.info = {
            **(info if info is not None else corpus_info()),
            "retrieval_mode": config.RETRIEVAL_MODE,
            "chat_model": config.CHAT_MODEL,
            "embed_model": config.EMBED_MODEL,
        }
        yield

    app = FastAPI(title="Nodi Tech RAG Chatbot", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.get("/info")
    def get_info(request: Request) -> dict:
        return request.app.state.info

    @app.post("/chat")
    def chat(req: ChatRequest, request: Request) -> StreamingResponse:
        chat_engine: ChatEngine = request.app.state.engine

        def events():
            # A plain (sync) generator: Starlette runs it in a worker thread, so a slow LLM
            # doesn't block the server for other requests.
            retrieved, trace, query, rewritten = [], {}, req.message, False
            try:
                for event in chat_engine.ask(req.session_id, req.message):
                    kind = event["type"]
                    if kind == "rewrite":
                        query, rewritten = event["query"], event["rewritten"]
                    elif kind == "sources":
                        retrieved, trace = event["sources"], event.get("trace", {})
                    elif kind == "token":
                        yield sse("delta", {"text": event["text"]})
                    elif kind == "replace":
                        yield sse("replace", {"text": event["answer"]})
                    elif kind == "done":
                        cards = cited_sources(event["answer"], retrieved)
                        yield sse("done", {"answer": event["answer"], "answered": is_answered(cards),
                                           "sources": cards, "retrieved": retrieved_view(retrieved, cards),
                                           "trace": trace, "query": query, "rewritten": rewritten,
                                           "timing": event["timing"]})
            except Exception as e:  # e.g. Ollama not running: tell the page instead of cutting the stream
                yield sse("error", {"message": str(e)})

        return StreamingResponse(events(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache"})

    @app.post("/feedback")
    def feedback(req: FeedbackRequest) -> dict:
        # Logged for humans to review later; the text comes from the page, so treat it as untrusted data.
        record = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), **req.model_dump()}
        feedback_file.parent.mkdir(parents=True, exist_ok=True)
        with feedback_lock, feedback_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return {"ok": True}

    @app.post("/reset")
    def reset(req: ResetRequest, request: Request) -> dict:
        request.app.state.engine.reset(req.session_id)
        return {"ok": True}

    return app


app = create_app()  # what uvicorn imports
