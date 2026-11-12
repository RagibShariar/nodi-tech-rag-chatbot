"""Query rewriting: turn a follow-up ("Does it have SSO?") into a standalone question
("Does the NodiDesk Growth plan have SSO?"), so retrieval knows what "it" means.

Uses its own small prompt (not the answering prompt): a 4B model does one job at a time better.
"""

from src.chat.history import ChatHistory
from src.chat.llm import LLM, Message

REWRITE_SYSTEM = """You rewrite the user's last message as a standalone question.

Rules:
- Use the conversation to replace words like "it", "its", "that", "and what about ..." with what they refer to.
- Keep the user's language.
- If the message is already a complete question, return it unchanged.
- Output only the question. No explanation, no quotes, no answer.

Example:
Conversation:
User: What does the Growth plan cost?
Assistant: BDT 2,500 per agent per month.
Last message: Does it have SSO?
Standalone question: Does the Growth plan have SSO?"""

MAX_ANSWER_CHARS = 300  # old answers are shortened in the transcript: fewer tokens, faster prefill
MAX_QUESTION_CHARS = 300  # longer output means the model rambled -> don't trust it


def _transcript(history: ChatHistory) -> str:
    lines = []
    for message in history.messages():
        text = " ".join(message["content"].split())
        if message["role"] == "assistant" and len(text) > MAX_ANSWER_CHARS:
            text = text[:MAX_ANSWER_CHARS] + "..."
        lines.append(f"{message['role'].capitalize()}: {text}")
    return "\n".join(lines)


def build_rewrite_messages(history: ChatHistory, message: str) -> list[Message]:
    return [
        {"role": "system", "content": REWRITE_SYSTEM},
        {"role": "user", "content": f"Conversation:\n{_transcript(history)}\n"
                                    f"Last message: {message}\nStandalone question:"},
    ]


def clean_rewrite(raw: str, fallback: str) -> str:
    """Take the first non-empty line, strip labels and quotes. Bad output -> the original message."""
    lines = [line.strip() for line in raw.strip().splitlines() if line.strip()]
    if not lines:
        return fallback
    text = lines[0]
    if text.lower().startswith("standalone question:"):
        text = text[len("standalone question:"):].strip()
    text = text.strip("\"'`")
    if not text or len(text) > MAX_QUESTION_CHARS:
        return fallback
    return text


def rewrite_query(llm: LLM, history: ChatHistory, message: str) -> str:
    if len(history) == 0:
        return message  # first question: nothing to resolve, skip the LLM call
    raw = "".join(llm.stream(build_rewrite_messages(history, message)))
    return clean_rewrite(raw, fallback=message)
