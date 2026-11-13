import csv

import pytest

from scripts.run_eval import (
    ask, check_answer, consistency, evaluate_one, load_eval_rows, rejudge, summarize, version_check,
    write_review_csv,
)
from src import config
from src.chat.engine import ChatEngine
from src.eval.judge import Judge
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all
from src.retrieval.pipeline import Retrieved
from tests.fakes import FakeLLM

ROWS = {r["id"]: r for r in load_eval_rows(config.EVAL_DIR / "eval_questions.csv")}

# --- loading ------------------------------------------------------------------------------------

def test_loads_all_40_with_bom_stripped_and_sources_split():
    assert len(ROWS) == 40
    assert "Q01" in ROWS  # a BOM would turn the first header into "﻿id"
    assert ROWS["Q25"]["sources"] == ["employee_handbook_v3.md", "expense_reimbursement_policy.pdf"]
    assert ROWS["Q36"]["sources"] == []  # "none" -> no source document
    assert sum(bool(r["sources"]) for r in ROWS.values()) == 36  # same 36 as retrieval_check

# --- version conflict ---------------------------------------------------------------------------

@pytest.mark.parametrize("answer, ok", [
    ("You get 18 days of annual leave. [employee_handbook_v3.md]", True),
    ("18 days. The old 15 days in the 2024 policy is superseded.", True),
    ("You get 15 days per year. [leave_policy_v2.1_2024_SUPERSEDED.md]", False),  # filename isn't "marked outdated"
    ("Either 15 or 18 days, depending on the policy.", False),  # old value given, not called outdated
    ("You get 118 days.", False),  # whole numbers only
])
def test_q29_version_check(answer, ok):
    assert version_check("Q29", answer) is ok


@pytest.mark.parametrize("answer, ok", [
    ("Up to 5 days, used by 31 March.", True),
    ("Up to 5 days by 31 March; the previous 10-day rule no longer applies.", True),
    ("You can carry forward up to 10 days.", False),
    ("Up to 15 days.", False),  # "5" inside "15" must not count
    ("Up to 5.5 days.", False),
])
def test_q30_version_check(answer, ok):
    assert version_check("Q30", answer) is ok


def test_version_check_only_applies_to_q29_q30():
    assert version_check("Q01", "15") is None

# --- per-answer checks --------------------------------------------------------------------------

def test_hit_and_citation_for_an_in_scope_question():
    c = check_answer(ROWS["Q15"], "Rate limit exceeded. [nodidesk_error_codes.md]",
                     ["nodidesk_api_reference.md", "nodidesk_error_codes.md"])
    assert c["hit"] and c["all_found"] and c["cites_retrieved"]
    assert c["cited"] == ["nodidesk_error_codes.md"]


def test_multi_source_question_hit_but_not_all_found():
    c = check_answer(ROWS["Q25"], "2 days. [employee_handbook_v3.md]", ["employee_handbook_v3.md"])
    assert c["hit"] is True and c["all_found"] is False


def test_miss_and_citing_a_file_that_was_not_retrieved():
    c = check_answer(ROWS["Q08"], "14 characters. [it_security_policy.md]", ["nodidesk_faq.md"])
    assert c["hit"] is False and c["cites_retrieved"] is False


def test_out_of_scope_question_is_not_measured_for_retrieval_or_citation():
    c = check_answer(ROWS["Q36"], "I can only help with questions about Nodi Tech.", ["nodidesk_faq.md"])
    assert c["hit"] is None and c["all_found"] is None and c["cites_retrieved"] is None

# --- summary ------------------------------------------------------------------------------------

def _result(qid, verdict, hit=True, total_s=10.0, **extra):
    return {"id": qid, "type": ROWS[qid]["type"], "verdict": verdict, "hit": hit, "all_found": hit,
            "cites_retrieved": hit, "version_ok": extra.get("version_ok"), "timing": {"total_s": total_s}}


def test_summary_counts():
    results = [
        _result("Q01", "CORRECT", total_s=8.0),
        _result("Q02", "PARTIAL", total_s=12.0),
        _result("Q29", "INCORRECT", hit=False, version_ok=False),
        _result("Q36", "CORRECT", hit=None),
        _result("Q40", "ERROR", hit=None),
    ]
    s = summarize(results)
    assert s["verdicts"] == {"CORRECT": 2, "PARTIAL": 1, "INCORRECT": 1, "ERROR": 1}
    assert s["correct_rate"] == pytest.approx(2 / 5)  # PARTIAL and ERROR don't count
    assert s["hit"] == (2, 3)                         # Q36/Q40 not measured
    assert s["version_ok"] == (0, 1)
    assert s["guardrails_correct"] == (1, 2)          # Q36 yes, Q40 ERROR
    assert s["by_type"]["factual"] == {"CORRECT": 1, "PARTIAL": 1}
    assert s["avg_total_s"] == pytest.approx(10.0)  # (8 + 12 + 10 + 10 + 10) / 5
    assert s["max_total_s"] == 12.0

# --- consistency across seeds -------------------------------------------------------------------

def test_consistency_across_three_seeds():
    def run(v1, v2, v3):
        return [_result("Q01", v1), _result("Q02", v2), _result("Q03", v3)]
    runs = {
        42: run("CORRECT", "CORRECT", "INCORRECT"),
        1: run("CORRECT", "PARTIAL", "INCORRECT"),
        2: run("CORRECT", "CORRECT", "INCORRECT"),
    }
    c = consistency(runs)
    assert c["seeds"] == [1, 2, 42]
    assert c["correct_rate_per_seed"] == {1: pytest.approx(1 / 3), 2: pytest.approx(2 / 3), 42: pytest.approx(2 / 3)}
    assert c["mean_correct_rate"] == pytest.approx(5 / 9)
    assert c["correct_in_k_seeds"] == {3: 1, 2: 1, 0: 1}
    assert c["unstable"] == ["Q02"]  # right in some seeds, wrong in others

# --- running one question (fake bot + fake judge, no Ollama) ------------------------------------

@pytest.fixture(scope="module")
def chunks():
    return {c.chunk_id: c for c in chunk_all(load_all(config.DATA_DIR))}


def _engine(chunks, *bot_replies):
    retrieve = lambda query: [Retrieved(chunks["nodidesk_error_codes.md#0"], 0.9)]  # noqa: E731
    return ChatEngine(FakeLLM(*bot_replies), retrieve)


def test_evaluate_one_runs_bot_then_judge(chunks):
    engine = _engine(chunks, "Rate limit exceeded (HTTP 429); wait for Retry-After. [nodidesk_error_codes.md]")
    judge_llm = FakeLLM('{"reason": "All key facts.", "verdict": "CORRECT"}')
    r = evaluate_one(engine, Judge(judge_llm), "fake-judge", ROWS["Q15"], seed=42)

    assert r["verdict"] == "CORRECT" and r["judge_model"] == "fake-judge" and r["seed"] == 42
    assert r["hit"] is True and r["cites_retrieved"] is True
    assert r["retrieved"][0]["source"] == "nodidesk_error_codes.md"
    assert "<bot_answer>\nRate limit exceeded" in judge_llm.calls[0][1]["content"]  # judge saw the bot's answer
    assert engine.sessions == {}  # fresh session: nothing left behind for the next question


def test_each_question_starts_without_history(chunks):
    # Two questions, two bot replies, NO rewrite reply scripted: if history leaked from Q15 into Q19,
    # the engine would call the LLM to rewrite and FakeLLM would run out of replies.
    engine = _engine(chunks, "answer one", "answer two")
    judge = Judge(FakeLLM(*['{"reason": "r", "verdict": "PARTIAL"}'] * 2))
    evaluate_one(engine, judge, "j", ROWS["Q15"], seed=1)
    r = evaluate_one(engine, judge, "j", ROWS["Q19"], seed=1)
    assert r["answer"] == "answer two"


def test_a_crash_is_retried_then_recorded_not_raised(chunks):
    engine = _engine(chunks)  # no replies scripted: the fake LLM raises IndexError on every attempt
    answer, sources, timing, error = ask(engine, "s", "anything", retries=1)
    assert answer == "" and error.startswith("IndexError")
    assert timing == {"attempts": 2}
    assert sources[0]["source"] == "nodidesk_error_codes.md"  # retrieval worked; kept despite the crash


def test_a_crash_followed_by_success_counts_the_attempts(chunks):
    class FlakyLLM(FakeLLM):
        def stream(self, messages):
            if not self.calls:  # first call only: crash like Ollama's HTTP 500
                self.calls.append(messages)
                raise RuntimeError("Ollama 500")
            yield from super().stream(messages)

    engine = ChatEngine(FlakyLLM("18 days."), lambda q: [Retrieved(chunks["nodidesk_error_codes.md#0"], 0.9)])
    answer, _, timing, error = ask(engine, "s", "q")
    assert (answer, error, timing["attempts"]) == ("18 days.", None, 2)


def test_bot_crash_is_error_and_never_sent_to_the_judge(chunks):
    judge_llm = FakeLLM()  # would raise if called
    r = evaluate_one(_engine(chunks), Judge(judge_llm), "j", ROWS["Q15"], seed=42)
    assert r["verdict"] == "ERROR" and r["reason"].startswith("bot crashed")
    assert judge_llm.calls == []
    assert r["hit"] is True           # retrieval did run: not a "retrieval miss"
    assert r["cites_retrieved"] is None


def test_judge_crash_is_retried_then_error():
    result = {"question": "q", "expected_answer": "e", "answer": "a"}
    from scripts.run_eval import grade
    r = grade(result, Judge(FakeLLM()), "j")  # FakeLLM with no replies raises on each call
    assert r["verdict"] == "ERROR" and r["reason"].startswith("judge crashed") and r["judge_attempts"] == 2

# --- rejudge -------------------------------------------------------------------------------------

def test_rejudge_regrades_saved_answers_and_keeps_the_old_verdict():
    saved = [{"question": "q", "expected_answer": "18 days.", "answer": "15 days.",
              "verdict": "CORRECT", "reason": "old", "judge_model": "old-judge"}]
    judge_llm = FakeLLM('{"reason": "Outdated value.", "verdict": "INCORRECT"}')
    # No engine is passed to rejudge() at all: it cannot regenerate answers.
    r = rejudge(saved, Judge(judge_llm), "new-judge")[0]

    assert len(judge_llm.calls) == 1
    assert r["answer"] == "15 days."  # untouched
    assert (r["verdict"], r["judge_model"]) == ("INCORRECT", "new-judge")
    assert r["previous"] == {"verdict": "CORRECT", "reason": "old", "judge_model": "old-judge"}

# --- review sheet --------------------------------------------------------------------------------

def test_review_csv_hides_the_judge_and_never_overwrites(tmp_path):
    results = [{"id": "Q33", "type": "bangla", "question": ROWS["Q33"]["question"],
                "expected_answer": ROWS["Q33"]["expected_answer"], "answer": " ৮০ টাকা \n", "verdict": "PARTIAL"}]
    path = tmp_path / "review.csv"

    assert write_review_csv(path, results) is True
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == ["id", "type", "question", "expected_answer", "answer", "human", "note"]
    assert "PARTIAL" not in path.read_text(encoding="utf-8-sig")  # judge's verdict not shown
    assert rows[0]["answer"] == "৮০ টাকা" and rows[0]["human"] == ""

    path.write_text("half graded", encoding="utf-8")
    assert write_review_csv(path, results) is False
    assert path.read_text(encoding="utf-8") == "half graded"

# --- judge vs human ------------------------------------------------------------------------------

def test_kappa_hand_computed():
    # po = 3/4. Human: C C I C (C=3, I=1); judge: C C I I (C=2, I=2).
    # pe = (3*2 + 1*2) / 16 = 0.5  ->  kappa = (0.75 - 0.5) / (1 - 0.5) = 0.5
    pairs = [("CORRECT", "CORRECT"), ("CORRECT", "CORRECT"), ("INCORRECT", "INCORRECT"), ("CORRECT", "INCORRECT")]
    from scripts.run_eval import cohen_kappa
    assert cohen_kappa(pairs) == pytest.approx(0.5)


def test_kappa_punishes_a_judge_that_always_says_correct():
    from scripts.run_eval import cohen_kappa
    pairs = [("CORRECT", "CORRECT")] * 9 + [("INCORRECT", "CORRECT")]
    assert cohen_kappa(pairs) == pytest.approx(0.0)  # 90% agreement, but no better than chance


def _labels_csv(tmp_path, grades):
    path = tmp_path / "labels.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "type", "question", "expected_answer", "answer", "human", "note"])
        for qid, grade_, note in grades:
            w.writerow([qid, "t", "q", "e", "a", grade_, note])
    return path


def test_human_labels_accept_shortcuts_and_reject_blanks(tmp_path):
    from scripts.run_eval import load_human_labels
    labels = load_human_labels(_labels_csv(tmp_path, [("Q01", "c", ""), ("Q02", " partial ", "missing X"), ("Q03", "I", "")]))
    assert {k: v["label"] for k, v in labels.items()} == {"Q01": "CORRECT", "Q02": "PARTIAL", "Q03": "INCORRECT"}
    assert labels["Q02"]["note"] == "missing X"

    with pytest.raises(SystemExit, match="Q02=.empty.*Q03=MAYBE"):
        load_human_labels(_labels_csv(tmp_path, [("Q01", "C", ""), ("Q02", "", ""), ("Q03", "maybe", "")]))


def test_compare_with_human():
    from scripts.run_eval import compare_with_human
    def r(qid, verdict):
        return {"id": qid, "question": "q", "answer": "a", "expected_answer": "e", "verdict": verdict, "reason": "why"}
    results = [r("Q01", "CORRECT"), r("Q02", "CORRECT"), r("Q03", "PARTIAL"), r("Q04", "ERROR")]
    labels = {q: {"label": l, "note": ""} for q, l in
              [("Q01", "CORRECT"), ("Q02", "PARTIAL"), ("Q03", "PARTIAL"), ("Q04", "INCORRECT")]}
    c = compare_with_human(results, labels)

    assert c["n"] == 3 and c["judge_errors"] == ["Q04"]           # ERROR not compared
    assert c["exact_agreement"] == pytest.approx(2 / 3)
    assert c["correct_agreement"] == pytest.approx(2 / 3)          # Q02: judge CORRECT, human not
    assert (c["too_lenient"], c["too_strict"]) == (1, 0)
    assert c["human_correct_rate"] == pytest.approx(1 / 4) and c["judge_correct_rate"] == pytest.approx(2 / 4)
    assert [d["id"] for d in c["disagreements"]] == ["Q02"]


def test_compare_refuses_labels_from_a_different_sheet():
    from scripts.run_eval import compare_with_human
    with pytest.raises(SystemExit, match="Q05"):
        compare_with_human([{"id": "Q05", "verdict": "CORRECT"}], {"Q01": {"label": "CORRECT", "note": ""}})
