"""Chat with a local LLM through Ollama's REST API, streaming the answer.

POST {OLLAMA_URL}/api/chat  {"model": ..., "messages": [...], "stream": true, ...}
Ollama answers with NDJSON: one JSON object per line, each carrying the next piece of text:
  {"message": {"role": "assistant", "content": "Annual"}, "done": false}
  {"message": {"role": "assistant", "content": " leave"}, "done": false}
  ...
  {"done": true, "prompt_eval_count": 1834, "prompt_eval_duration": 41000000000, "eval_count": 57, ...}
"""

import json
from dataclasses import dataclass
from typing import Iterator, Protocol

import httpx

from src import config

Message = dict[str, str]  # {"role": "system" | "user" | "assistant", "content": "..."}


class LLM(Protocol):
    """Anything that can stream a reply to a list of messages. Tests use a fake one."""

    def stream(self, messages: list[Message]) -> Iterator[str]: ...


@dataclass(frozen=True)
class Stats:
    """From Ollama's final 'done' line. Durations come in nanoseconds; we store seconds."""
    prompt_tokens: int = 0       # prefill: tokens the model READ (system + history + documents + question)
    prompt_seconds: float = 0.0
    output_tokens: int = 0       # generation: tokens the model WROTE
    output_seconds: float = 0.0

    @property
    def prefill_tps(self) -> float:
        return self.prompt_tokens / self.prompt_seconds if self.prompt_seconds else 0.0

    @property
    def output_tps(self) -> float:
        return self.output_tokens / self.output_seconds if self.output_seconds else 0.0


class OllamaChat:
    def __init__(self, model: str | None = None, client: httpx.Client | None = None):
        self.model = model or config.CHAT_MODEL
        # On CPU the model may read the prompt for a minute before the first byte arrives,
        # so the read timeout (time between two chunks) must be generous.
        self.client = client or httpx.Client(timeout=httpx.Timeout(10.0, read=300.0))
        self.last_stats = Stats()

    def stream(self, messages: list[Message]) -> Iterator[str]:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "think": False,  # Qwen can "think out loud" first: many hidden tokens = many seconds on CPU
            "options": {
                "temperature": config.LLM_TEMPERATURE,
                "seed": config.LLM_SEED,
                "num_ctx": config.NUM_CTX,
            },
        }
        with self.client.stream("POST", f"{config.OLLAMA_URL}/api/chat", json=payload) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.strip():
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(f"Ollama error: {data['error']}")
                piece = data.get("message", {}).get("content", "")
                if piece:
                    yield piece
                if data.get("done"):
                    self.last_stats = Stats(
                        prompt_tokens=data.get("prompt_eval_count", 0),
                        prompt_seconds=data.get("prompt_eval_duration", 0) / 1e9,
                        output_tokens=data.get("eval_count", 0),
                        output_seconds=data.get("eval_duration", 0) / 1e9,
                    )

    def complete(self, messages: list[Message]) -> str:
        """Whole reply as one string (for short internal calls like query rewriting)."""
        return "".join(self.stream(messages))
