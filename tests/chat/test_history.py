from src.chat.history import ChatHistory


def test_messages_alternate_user_assistant():
    h = ChatHistory(max_turns=6)
    h.add("q1", "a1")
    h.add("q2", "a2")
    assert h.messages() == [
        {"role": "user", "content": "q1"}, {"role": "assistant", "content": "a1"},
        {"role": "user", "content": "q2"}, {"role": "assistant", "content": "a2"},
    ]


def test_keeps_only_last_6_turns():
    h = ChatHistory(max_turns=6)
    for i in range(1, 8):  # 7 turns
        h.add(f"q{i}", f"a{i}")
    assert len(h) == 6
    assert len(h.messages()) == 12
    assert h.messages()[0]["content"] == "q2"  # turn 1 was dropped
    assert h.messages()[-1]["content"] == "a7"


def test_clear():
    h = ChatHistory()
    h.add("q", "a")
    h.clear()
    assert len(h) == 0 and h.messages() == []
