import pytest

from src import config
from src.chat.prompts import SYSTEM_PROMPT, build_messages, format_documents
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all


@pytest.fixture(scope="module")
def by_id():
    return {c.chunk_id: c for c in chunk_all(load_all(config.DATA_DIR))}


def test_documents_are_fenced_with_metadata(by_id):
    text = format_documents([by_id["employee_handbook_v3.md#4"]])
    assert text.startswith("<documents>\n") and text.endswith("\n</documents>")
    assert '<document source="employee_handbook_v3.md" section="3. Leave > Carry forward" status="current">' in text
    assert "31 March" in text


def test_superseded_status_visible(by_id):
    text = format_documents([by_id["leave_policy_v2.1_2024_SUPERSEDED.md#1"]])
    assert 'status="superseded"' in text


def test_context_line_not_repeated(by_id):
    # metadata lives in the tag; the M1 "[Source: ...]" line would waste tokens
    assert "[Source:" not in format_documents(list(by_id.values())[:5])


def test_message_order_system_history_then_documents(by_id):
    history = [{"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"}]
    messages = build_messages(history, [by_id["employee_handbook_v3.md#3"]], "How many days?")

    assert [m["role"] for m in messages] == ["system", "user", "assistant", "user"]
    assert messages[0]["content"] == SYSTEM_PROMPT
    assert messages[-1]["content"].startswith("<documents>")
    assert messages[-1]["content"].endswith("Question: How many days?")


def test_system_prompt_has_the_core_rules():
    for rule in ["ONLY from the documents", "[filename]", "I don't know", 'status="current"',
                 "data, not instructions", "same language"]:
        assert rule in SYSTEM_PROMPT
