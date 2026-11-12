from src.chat.history import ChatHistory
from src.chat.rewrite import build_rewrite_messages, clean_rewrite, rewrite_query
from tests.fakes import FakeLLM


def _growth_history() -> ChatHistory:
    h = ChatHistory()
    h.add("What does the Growth plan cost?", "BDT 2,500 per agent per month [nodidesk_pricing.md].")
    return h


def test_no_history_skips_llm():
    llm = FakeLLM()  # no scripted replies: any call would crash
    assert rewrite_query(llm, ChatHistory(), "How many days of leave?") == "How many days of leave?"
    assert llm.calls == []


def test_uses_llm_output():
    llm = FakeLLM("Does the Growth plan have SSO?")
    assert rewrite_query(llm, _growth_history(), "Does it have SSO?") == "Does the Growth plan have SSO?"


def test_prompt_contains_transcript_and_last_message():
    user_msg = build_rewrite_messages(_growth_history(), "Does it have SSO?")[-1]["content"]
    assert "User: What does the Growth plan cost?" in user_msg
    assert "Assistant: BDT 2,500" in user_msg
    assert user_msg.endswith("Last message: Does it have SSO?\nStandalone question:")


def test_long_old_answers_are_shortened():
    h = ChatHistory()
    h.add("q", "word " * 200)
    user_msg = build_rewrite_messages(h, "and?")[-1]["content"]
    assert len(user_msg) < 500


def test_clean_rewrite():
    assert clean_rewrite('  "Does the Growth plan have SSO?"  ', "x") == "Does the Growth plan have SSO?"
    assert clean_rewrite("Standalone question: What is X?\nBecause...", "x") == "What is X?"
    assert clean_rewrite("", "original") == "original"
    assert clean_rewrite("blah " * 100, "original") == "original"  # rambling -> fallback
