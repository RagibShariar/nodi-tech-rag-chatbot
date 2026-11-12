"""Chat history for one session: keeps only the last N turns (1 turn = 1 question + 1 answer).

Older turns are dropped, not summarized: the context window (and, on CPU, prefill time) is limited.
"""

from src import config
from src.chat.llm import Message


class ChatHistory:
    def __init__(self, max_turns: int = config.HISTORY_TURNS):
        self.max_turns = max_turns
        self._turns: list[tuple[str, str]] = []  # (user message, assistant answer)

    def add(self, question: str, answer: str) -> None:
        self._turns.append((question, answer))
        self._turns = self._turns[-self.max_turns:]  # keep only the newest N

    def messages(self) -> list[Message]:
        out: list[Message] = []
        for question, answer in self._turns:
            out.append({"role": "user", "content": question})
            out.append({"role": "assistant", "content": answer})
        return out

    def clear(self) -> None:
        self._turns.clear()

    def __len__(self) -> int:
        return len(self._turns)
