"""Chat engine: rewrite -> retrieve -> build prompt -> stream answer -> save to history.

`ask()` yields events, in this order:
  {"type": "rewrite", "query": ..., "rewritten": bool}
  {"type": "sources", "sources": [{"source", "section", "status", "score"}, ...]}
  {"type": "token", "text": ...}            (many)
  {"type": "done", "answer": ..., "timing": {...}}
"""

import time
from typing import Callable, Iterator

from src import config
from src.chat.history import ChatHistory
from src.chat.llm import LLM, OllamaChat
from src.chat.prompts import build_messages
from src.chat.rewrite import rewrite_query
from src.retrieval.pipeline import Retrieved

TOP_K = 5
Event = dict
RetrieveFn = Callable[[str], list[Retrieved]]


class ChatEngine:
    def __init__(self, llm: LLM, retrieve: RetrieveFn):
        self.llm = llm
        self.retrieve = retrieve
        self.sessions: dict[str, ChatHistory] = {}  # in memory: lost on restart (fine for now)

    def history(self, session_id: str) -> ChatHistory:
        return self.sessions.setdefault(session_id, ChatHistory())

    def reset(self, session_id: str) -> None:
        self.sessions.pop(session_id, None)

    def ask(self, session_id: str, message: str) -> Iterator[Event]:
        history = self.history(session_id)
        start = time.perf_counter()
        timing: dict[str, float | int] = {}

        # 1. Follow-up -> standalone question (skipped on the first turn).
        query = rewrite_query(self.llm, history, message)
        timing["rewrite_s"] = round(time.perf_counter() - start, 2)
        yield {"type": "rewrite", "query": query, "rewritten": query != message}

        # 2. Retrieve with the standalone question.
        t = time.perf_counter()
        results = self.retrieve(query)
        timing["retrieve_s"] = round(time.perf_counter() - t, 2)
        yield {"type": "sources", "sources": [
            {"source": r.chunk.source, "section": r.chunk.section, "status": r.chunk.status,
             "score": round(r.score, 4)} for r in results
        ]}

        # 3. Answer, streamed. History holds the user's ORIGINAL words; the prompt gets the standalone query.
        messages = build_messages(history.messages(), [r.chunk for r in results], query)
        t = time.perf_counter()
        pieces: list[str] = []
        for piece in self.llm.stream(messages):
            if not pieces:
                timing["first_token_s"] = round(time.perf_counter() - start, 2)  # what the user feels
            pieces.append(piece)
            yield {"type": "token", "text": piece}
        timing["answer_s"] = round(time.perf_counter() - t, 2)
        timing["total_s"] = round(time.perf_counter() - start, 2)
        stats = getattr(self.llm, "last_stats", None)  # real Ollama client only
        if stats:
            timing["prompt_tokens"] = stats.prompt_tokens
            timing["output_tokens"] = stats.output_tokens

        answer = "".join(pieces)
        history.add(message, answer)
        yield {"type": "done", "answer": answer, "timing": timing}


def create_engine(mode: str = config.RETRIEVAL_MODE) -> ChatEngine:
    """Wire the real parts together. Slow (loads the reranker, ~20s): call once at startup."""
    from src.ingest.chunk import chunk_all
    from src.ingest.parse import load_all
    from src.retrieval.embed import embed
    from src.retrieval.pipeline import Retriever
    from src.retrieval.vector_store import collection_name, get_client

    rerank_fn = None
    if mode == "rerank":
        from src.retrieval.rerank import CrossEncoderReranker
        rerank_fn = CrossEncoderReranker()

    retriever = Retriever(
        chunk_all(load_all(config.DATA_DIR)),
        get_client(),  # stays open while the app runs (local mode locks the folder)
        collection_name(config.EMBED_MODEL),
        lambda text, kind: embed(text, kind),
        rerank_fn,
    )
    return ChatEngine(OllamaChat(), lambda query: retriever.retrieve(query, mode=mode, k=TOP_K))
