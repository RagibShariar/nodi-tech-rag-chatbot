import json

import httpx
import pytest

from src.chat.llm import OllamaChat


def _ndjson(*objects) -> bytes:
    return "\n".join(json.dumps(o) for o in objects).encode() + b"\n"


def _fake_ollama(body: bytes, sent: list):
    """An httpx client whose 'server' returns `body` and records the request payload."""
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(json.loads(request.content))
        return httpx.Response(200, content=body)
    return httpx.Client(transport=httpx.MockTransport(handler))


def test_streams_pieces_and_reads_stats():
    body = _ndjson(
        {"message": {"role": "assistant", "content": "Annual"}, "done": False},
        {"message": {"role": "assistant", "content": " leave"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True,
         "prompt_eval_count": 1000, "prompt_eval_duration": 20_000_000_000,
         "eval_count": 50, "eval_duration": 10_000_000_000},
    )
    llm = OllamaChat(model="m", client=_fake_ollama(body, []))

    assert list(llm.stream([{"role": "user", "content": "hi"}])) == ["Annual", " leave"]
    assert llm.last_stats.prompt_tokens == 1000
    assert llm.last_stats.prefill_tps == pytest.approx(50.0)   # 1000 tokens / 20 s
    assert llm.last_stats.output_tps == pytest.approx(5.0)     # 50 tokens / 10 s


def test_sends_streaming_no_thinking_and_options():
    sent = []
    body = _ndjson({"message": {"content": "ok"}, "done": True})
    llm = OllamaChat(model="qwen-test", client=_fake_ollama(body, sent))
    llm.complete([{"role": "user", "content": "hi"}])

    payload = sent[0]
    assert payload["model"] == "qwen-test"
    assert payload["stream"] is True
    assert payload["think"] is False
    assert {"temperature", "seed", "num_ctx"} <= payload["options"].keys()


def test_complete_joins_pieces():
    body = _ndjson({"message": {"content": "18 "}, "done": False}, {"message": {"content": "days"}, "done": True})
    assert OllamaChat(client=_fake_ollama(body, [])).complete([]) == "18 days"


def test_error_line_raises():
    body = _ndjson({"error": "model not found"})
    with pytest.raises(RuntimeError, match="model not found"):
        OllamaChat(client=_fake_ollama(body, [])).complete([])
