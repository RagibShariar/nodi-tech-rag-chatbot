"""Chat engine: rewrite -> retrieve -> build prompt -> stream answer -> save to history.

`ask()` yields events, in this order:
  {"type": "rewrite", "query": ..., "rewritten": bool}
  {"type": "sources", "sources": [{"source", "section", "status", "score", "content",
                                    "vector_rank", "bm25_rank", "rrf_score", "rerank_score"}, ...],
   "trace": {"vector": [...], "bm25": [...]}}   (trace is empty if the retriever can't produce one)
  {"type": "token", "text": ...}            (many)
  {"type": "replace", "answer": ...}        (only if the leak guard stopped the answer)
  {"type": "done", "answer": ..., "timing": {...}}
"""

import inspect
import time
from typing import Callable, Iterator

from src import config
from src.chat.guardrails import REFUSAL, LeakGuard
from src.chat.history import ChatHistory
from src.chat.llm import LLM, OllamaChat
from src.chat.prompts import PROTECTED_TEXT, build_messages
from src.chat.rewrite import rewrite_query
from src.retrieval.pipeline import Retrieved

TOP_K = 5
Event = dict
RetrieveFn = Callable[..., list[Retrieved]]  # retrieve(query) or retrieve(query, trace=dict)
LEAK_GUARD = LeakGuard(PROTECTED_TEXT)


def _accepts_trace(fn: Callable) -> bool:
    try:
        return "trace" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False


class ChatEngine:
    def __init__(self, llm: LLM, retrieve: RetrieveFn):
        self.llm = llm
        self.retrieve = retrieve
        self._trace = _accepts_trace(retrieve)  # test fakes take only the query
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
        trace: dict = {}
        results = self.retrieve(query, trace=trace) if self._trace else self.retrieve(query)
        timing["retrieve_s"] = round(time.perf_counter() - t, 2)
        yield {"type": "sources", "trace": trace, "sources": [
            {"source": r.chunk.source, "section": r.chunk.section, "status": r.chunk.status,
             "score": round(r.score, 4), "content": r.chunk.content,
             "vector_rank": r.vector_rank, "bm25_rank": r.bm25_rank,
             "rrf_score": None if r.rrf_score is None else round(r.rrf_score, 4),
             "rerank_score": None if r.rerank_score is None else round(r.rerank_score, 4)} for r in results
        ]}

        # 3. Answer, streamed. History holds the user's ORIGINAL words; the prompt gets the standalone query.
        messages = build_messages(history.messages(), [r.chunk for r in results], query)
        t = time.perf_counter()
        pieces: list[str] = []
        leaked = False
        for piece in self.llm.stream(messages):
            # Leak guard: check BEFORE sending. If this piece would complete a 6-word copy of the
            # rules, stop. At most 5 copied words can reach the user (a streaming trade-off).
            if LEAK_GUARD.leaked("".join(pieces) + piece):
                leaked = True
                break  # leaving the loop closes the HTTP stream, so Ollama stops writing too
            if not pieces:
                timing["first_token_s"] = round(time.perf_counter() - start, 2)  # what the user feels
            pieces.append(piece)
            yield {"type": "token", "text": piece}
        if leaked:
            pieces = [REFUSAL]
            yield {"type": "replace", "answer": REFUSAL}
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

    chunks = chunk_all(load_all(config.DATA_DIR))
    retriever = Retriever(
        chunks,
        get_client(),  # stays open while the app runs (local mode locks the folder)
        collection_name(config.EMBED_MODEL),
        lambda text, kind: embed(text, kind),
        rerank_fn,
    )
    return ChatEngine(OllamaChat(),
                      lambda query, trace=None: retriever.retrieve(query, mode=mode, k=TOP_K, trace=trace))
