"""FastAPI app: the chat page and a streaming chat endpoint.

Run:  python -m uvicorn src.api.app:app --port 8000     (startup takes ~20s: loads the reranker)

POST /chat  {"session_id": "...", "message": "..."}
  -> NDJSON stream, one event per line (same idea as Ollama's own streaming):
     {"type": "rewrite", ...}  {"type": "sources", ...}  {"type": "token", "text": ...} ...  {"type": "done", ...}
     or {"type": "error", "message": ...} if something fails mid-way.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from src.chat.engine import ChatEngine

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)
    message: str = Field(min_length=1, max_length=2000)


class ResetRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=100)


def create_app(engine: ChatEngine | None = None) -> FastAPI:
    """Tests pass a fake engine; the real one is built once at startup (slow)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
        else:
            from src.chat.engine import create_engine
            app.state.engine = create_engine()
        yield

    app = FastAPI(title="Nodi Tech RAG Chatbot", lifespan=lifespan)

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    @app.post("/chat")
    def chat(req: ChatRequest, request: Request) -> StreamingResponse:
        chat_engine: ChatEngine = request.app.state.engine

        def events():
            # A plain (sync) generator: Starlette runs it in a worker thread, so a slow LLM
            # doesn't block the server for other requests.
            try:
                for event in chat_engine.ask(req.session_id, req.message):
                    yield json.dumps(event, ensure_ascii=False) + "\n"
            except Exception as e:  # e.g. Ollama not running: tell the page instead of cutting the stream
                yield json.dumps({"type": "error", "message": str(e)}) + "\n"

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.post("/reset")
    def reset(req: ResetRequest, request: Request) -> dict:
        request.app.state.engine.reset(req.session_id)
        return {"ok": True}

    return app


app = create_app()  # what uvicorn imports
