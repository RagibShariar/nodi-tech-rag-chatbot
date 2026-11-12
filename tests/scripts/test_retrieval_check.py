from src import config
from scripts.retrieval_check import Question, Result, first_hit_rank, load_questions, summarize


def test_load_questions_skips_none():
    questions = load_questions(config.EVAL_DIR / "eval_questions.csv")
    assert len(questions) == 36  # 40 minus Q35, Q36, Q37, Q40 (source "none")
    q26 = next(q for q in questions if q.id == "Q26")
    assert q26.sources == {"nodidesk_error_codes.md", "nodidesk_user_guide.md", "nodidesk_release_notes.md"}


def test_first_hit_rank():
    expected = frozenset({"b.md", "c.md"})
    assert first_hit_rank(expected, ["a.md", "b.md", "c.md"]) == 2
    assert first_hit_rank(expected, ["a.md", "a.md"]) is None


def test_summary_metrics():
    q1 = Question("1", "factual", "?", frozenset({"a.md"}))
    q2 = Question("2", "multi_doc", "?", frozenset({"a.md", "b.md"}))
    q3 = Question("3", "factual", "?", frozenset({"z.md"}))
    results = [
        Result(q1, ["a.md", "x.md"], rank=1),  # hit at rank 1
        Result(q2, ["x.md", "a.md"], rank=2),  # hit at rank 2, but b.md missing
        Result(q3, ["x.md", "y.md"], rank=None),  # miss
    ]
    s = summarize(results)
    assert s["hits"] == 2
    assert s["all_found"] == 1
    assert abs(s["mrr"] - (1 + 0.5 + 0) / 3) < 1e-9
