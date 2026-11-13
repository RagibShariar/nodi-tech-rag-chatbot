"""M6 evaluation: all 40 questions in eval/eval_questions.csv through the full chatbot, graded by the judge.

Per question: retrieval hit (expected document in the top 5), judge verdict vs expected_answer, whether the
answer cites a retrieved file, the Q29/Q30 version-conflict check, and timing. Each question runs in a
fresh session (no history), with the same engine the web page uses.

Usage:
  python -u -m scripts.run_eval                          # all 40, seed 42 -> artifacts/eval_seed42.json
  python -u -m scripts.run_eval --seeds 42 1 2           # one full run per seed + consistency report
  python -u -m scripts.run_eval --only Q29 Q30           # a few questions (saved as eval_seed42_partial.json)
  python -u -m scripts.run_eval --rejudge artifacts/eval_seed42.json   # re-grade saved answers, no chatbot
  python -m scripts.run_eval --compare-human docs/eval/human_labels.csv artifacts/eval_seed42.json
"""

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from scripts.retrieval_check import first_hit_rank
from scripts.run_conversations import CITATION_RE, OUTDATED_RE
from src import config
from src.eval.judge import ERROR, VERDICTS

ARTIFACTS = config.ROOT / "artifacts"
REVIEW_CSV = ARTIFACTS / "eval_for_review.csv"
TARGETS = {"hit": 0.90, "correct": 0.85}  # CLAUDE.md

# Version conflict (CLAUDE.md target): the current value must be given; the superseded value may only
# appear when the answer calls it outdated (same rule as M4). Whole numbers only: "5" must not match "15".
VERSION_CHECKS = {
    "Q29": {"current": "18", "superseded": "15"},  # annual leave days
    "Q30": {"current": "5", "superseded": "10"},   # carry-forward days
}
GUARDRAIL_IDS = ("Q35", "Q36", "Q37", "Q38", "Q39", "Q40")  # 3 out-of-scope + 3 guardrail


def load_eval_rows(path: Path) -> list[dict]:
    """All 40 rows. 'sources' is empty for questions with no source document (out-of-scope, Q40)."""
    with path.open(encoding="utf-8-sig") as f:  # the CSV starts with a BOM
        rows = list(csv.DictReader(f))
    for row in rows:
        row["sources"] = sorted(s.strip() for s in row.pop("source_documents").split(";")
                                if s.strip() and s.strip() != "none")
    return rows


def _has_number(text: str, number: str) -> bool:
    return re.search(rf"(?<![\d.,]){re.escape(number)}(?![\d.,]\d|\d)", text) is not None


def version_check(qid: str, answer: str) -> bool | None:
    """True/False for Q29/Q30, None for every other question."""
    rule = VERSION_CHECKS.get(qid)
    if rule is None:
        return None
    no_citations = CITATION_RE.sub("", answer)  # "..._SUPERSEDED.md" must not count as calling it outdated
    if not _has_number(no_citations, rule["current"]):
        return False
    gives_old = _has_number(no_citations, rule["superseded"])
    return not gives_old or OUTDATED_RE.search(no_citations.lower()) is not None


def check_answer(row: dict, answer: str, retrieved: list[str]) -> dict:
    """Automatic (non-judge) checks for one answer. `retrieved`: source file of each top-5 chunk, best first."""
    expected = frozenset(row["sources"])
    cited = sorted(set(CITATION_RE.findall(answer)))
    in_scope = bool(expected)
    return {
        # Retrieval and citations are only measured for questions that have a source document.
        "hit": first_hit_rank(expected, retrieved) is not None if in_scope else None,
        "all_found": expected <= set(retrieved) if in_scope else None,
        "cited": cited,
        "cites_retrieved": bool(set(cited) & set(retrieved)) if in_scope else None,
        "version_ok": version_check(row["id"], answer),
    }


def _rate(values: list[bool | None]) -> tuple[int, int]:
    """(true count, measured count), ignoring None (not applicable)."""
    measured = [v for v in values if v is not None]
    return sum(measured), len(measured)


def summarize(results: list[dict]) -> dict:
    """One seed's results -> the numbers we report. Each result has the check_answer() keys + 'verdict',
    'type', 'id' and 'timing' (with 'total_s')."""
    verdicts = Counter(r["verdict"] for r in results)
    by_type: dict[str, Counter] = defaultdict(Counter)
    for r in results:
        by_type[r["type"]][r["verdict"]] += 1
    guard = [r for r in results if r["id"] in GUARDRAIL_IDS]
    totals = [r["timing"]["total_s"] for r in results if r["timing"].get("total_s") is not None]
    return {
        "n": len(results),
        "verdicts": {v: verdicts.get(v, 0) for v in (*VERDICTS, ERROR)},
        "correct_rate": verdicts.get("CORRECT", 0) / len(results),  # only CORRECT counts toward the target
        "hit": _rate([r["hit"] for r in results]),
        "all_found": _rate([r["all_found"] for r in results]),
        "cites_retrieved": _rate([r["cites_retrieved"] for r in results]),
        "version_ok": _rate([r["version_ok"] for r in results]),
        "guardrails_correct": (sum(r["verdict"] == "CORRECT" for r in guard), len(guard)),
        "by_type": {t: dict(c) for t, c in by_type.items()},
        "bot_errors": sum(bool(r.get("bot_error")) for r in results),
        "retried": sum(r["timing"].get("attempts", 1) > 1 for r in results),
        "avg_total_s": sum(totals) / len(totals) if totals else None,
        "max_total_s": max(totals) if totals else None,
    }


def consistency(runs: dict[int, list[dict]]) -> dict:
    """Several seeds -> how stable each question is. runs: {seed: results}, same questions in each run."""
    seeds = sorted(runs)
    per_question: dict[str, list[str]] = defaultdict(list)
    for seed in seeds:
        for r in runs[seed]:
            per_question[r["id"]].append(r["verdict"])
    correct_counts = {qid: v.count("CORRECT") for qid, v in per_question.items()}
    rates = [sum(r["verdict"] == "CORRECT" for r in runs[s]) / len(runs[s]) for s in seeds]
    return {
        "seeds": seeds,
        "correct_rate_per_seed": dict(zip(seeds, rates)),
        "mean_correct_rate": sum(rates) / len(rates),
        # e.g. {3: 30, 2: 5, 1: 3, 0: 2} = 30 questions correct in all 3 seeds, 2 never
        "correct_in_k_seeds": dict(sorted(Counter(correct_counts.values()).items(), reverse=True)),
        "unstable": sorted(qid for qid, k in correct_counts.items() if 0 < k < len(seeds)),
        "verdicts": dict(per_question),
    }


# ---------- running ----------

def ask(engine, session: str, question: str, retries: int = 1) -> tuple[str, list[dict], dict, str | None]:
    """One question in a fresh session -> (answer, retrieved chunks, timing, error).

    Never raises: a crash on one question must not lose a 40-minute run. A crash (e.g. Ollama's GPU runner
    failing to load: HTTP 500) is retried `retries` times; the number of attempts is recorded in timing so
    retries are visible in the report, never hidden. If every attempt fails, error says why."""
    error, sources = None, []
    for attempt in range(1, retries + 2):
        engine.reset(session)
        sources, done = [], None
        try:
            for event in engine.ask(session, question):  # stream: keep the sources even if the answer crashes
                if event["type"] == "sources":
                    sources = event["sources"]
                elif event["type"] == "done":
                    done = event
            return done["answer"], sources, {**done["timing"], "attempts": attempt}, None
        except Exception as e:  # noqa: BLE001 (every failure is recorded, whatever it is)
            error = f"{type(e).__name__}: {e}"
        finally:
            engine.reset(session)
    return "", sources, {"attempts": retries + 1}, error


def grade(result: dict, judge, judge_model: str) -> dict:
    """Add the judge's verdict to a result (in place) and return it. A bot crash is not graded: it is an
    ERROR (counted as not correct, reported separately), never a judge opinion about an error message."""
    if result.get("bot_error"):
        result.update(verdict=ERROR, reason=f"bot crashed: {result['bot_error']}", judge_raw="",
                      judge_model=judge_model, judge_s=0.0)
        return result
    start = time.perf_counter()
    for attempt in (1, 2):  # the judge runs on the same Ollama, so it can crash too: one retry
        try:
            j = judge(result["question"], result["expected_answer"], result["answer"])
            result.update(verdict=j.verdict, reason=j.reason, judge_raw=j.raw)
            break
        except Exception as e:  # noqa: BLE001
            result.update(verdict=ERROR, reason=f"judge crashed: {type(e).__name__}: {e}", judge_raw="")
    result.update(judge_model=judge_model, judge_attempts=attempt, judge_s=round(time.perf_counter() - start, 2))
    return result


def evaluate_one(engine, judge, judge_model: str, row: dict, seed: int) -> dict:
    answer, sources, timing, error = ask(engine, f"eval-{seed}-{row['id']}", row["question"])
    retrieved = [s["source"] for s in sources]
    result = {"id": row["id"], "type": row["type"], "question": row["question"],
              "expected_answer": row["expected_answer"], "sources": row["sources"], "seed": seed,
              "answer": answer, "bot_error": error, "retrieved": sources, "timing": timing,
              **check_answer(row, answer, retrieved)}
    if error and not sources:
        result["hit"] = result["all_found"] = None  # crashed before retrieval finished: not measured
    if error:
        result["cites_retrieved"] = None  # no answer to check
    return grade(result, judge, judge_model)


def warm_up(llm, tries: int = 3) -> None:
    """Load the chat model before the first question, so a cold-start crash isn't charged to a question."""
    for attempt in range(1, tries + 1):
        try:
            llm.complete([{"role": "user", "content": "Reply with: ok"}])
            return
        except Exception as e:  # noqa: BLE001
            print(f"warm-up attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
    raise SystemExit("The chat model could not be loaded; is Ollama running?")


def rejudge(results: list[dict], judge, judge_model: str) -> list[dict]:
    """Re-grade saved answers WITHOUT regenerating them (no chatbot involved). The old verdict is kept."""
    for r in results:
        r["previous"] = {k: r.get(k) for k in ("verdict", "reason", "judge_model")}
        grade(r, judge, judge_model)
    return results


def save(path: Path, meta: dict, results: list[dict]) -> None:
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps({**meta, "results": results}, ensure_ascii=False, indent=2), encoding="utf-8")


def write_review_csv(path: Path, results: list[dict]) -> bool:
    """CSV for human grading. The judge's verdict is left out so it can't anchor the human grade.
    Never overwrites: a half-graded file is precious. Returns False if it already existed."""
    if path.exists():
        return False
    with path.open("w", encoding="utf-8-sig", newline="") as f:  # -sig: Excel shows Bangla correctly
        writer = csv.writer(f)
        writer.writerow(["id", "type", "question", "expected_answer", "answer", "human", "note"])
        for r in results:
            writer.writerow([r["id"], r["type"], r["question"], r["expected_answer"], r["answer"].strip(), "", ""])
    return True


# ---------- reporting ----------

def _pct(pair: tuple[int, int]) -> str:
    true, n = pair
    return f"{true}/{n} = {true / n:.1%}" if n else "n/a"


def _met(ok: bool) -> str:
    return "MET" if ok else "NOT MET"


def print_result(r: dict) -> None:
    flags = []
    if r["hit"] is False:
        flags.append("RETRIEVAL MISS")
    if r["cites_retrieved"] is False:
        flags.append("no valid citation")
    if r["version_ok"] is False:
        flags.append("VERSION CONFLICT")
    if r["timing"].get("attempts", 1) > 1:
        flags.append(f"{r['timing']['attempts']} attempts")
    flag_text = "  " + "; ".join(flags) if flags else ""
    print(f"\n=== {r['id']} [{r['type']}] {r['verdict']}{flag_text}")
    print(f"Q      : {r['question']}")
    print(f"A      : {r['answer'].strip() or '(none)'}{'  [BOT ERROR] ' + r['bot_error'] if r.get('bot_error') else ''}")
    print(f"expect : {r['expected_answer']}")
    print(f"judge  : {r['reason']}")
    print(f"top 5  : {', '.join(s['source'] for s in r['retrieved'])}")
    print(f"time   : answer {r['timing'].get('total_s', '-')}s | judge {r['judge_s']}s", flush=True)


def print_summary(title: str, s: dict) -> None:
    hit_true, hit_n = s["hit"]
    g_true, g_n = s["guardrails_correct"]
    vo_true, vo_n = s["version_ok"]
    v = s["verdicts"]
    print(f"\n===== {title} =====")
    print(f"retrieval hit@5 : {_pct(s['hit'])}   (target >= {TARGETS['hit']:.0%}: "
          f"{_met(hit_n > 0 and hit_true / hit_n >= TARGETS['hit'])})")
    print(f"answers CORRECT : {v['CORRECT']}/{s['n']} = {s['correct_rate']:.1%}   "
          f"(target >= {TARGETS['correct']:.0%}: {_met(s['correct_rate'] >= TARGETS['correct'])})")
    print(f"                  PARTIAL {v['PARTIAL']} | INCORRECT {v['INCORRECT']} | ERROR {v['ERROR']} "
          f"(bot crashed {s['bot_errors']}, judge unreadable or crashed {v['ERROR'] - s['bot_errors']})")
    print(f"retried         : {s['retried']} question(s) needed a retry after a crash")
    print(f"Q35-Q40         : {g_true}/{g_n} CORRECT   (target 6/6: {_met(g_n == 6 and g_true == 6)})")
    print(f"version Q29/Q30 : {vo_true}/{vo_n}   (target all: {_met(vo_n > 0 and vo_true == vo_n)})")
    print(f"all sources@5   : {_pct(s['all_found'])}")
    print(f"cites retrieved : {_pct(s['cites_retrieved'])}")
    if s["avg_total_s"] is not None:
        print(f"answer time     : avg {s['avg_total_s']:.1f}s, max {s['max_total_s']:.1f}s")
    print("by type:")
    for qtype, counts in s["by_type"].items():
        others = {k: n for k, n in counts.items() if k != "CORRECT"}
        print(f"  {qtype:<17} CORRECT {counts.get('CORRECT', 0)}/{sum(counts.values())}"
              f"{'   ' + str(others) if others else ''}")


def print_consistency(c: dict) -> None:
    n = len(c["seeds"])
    print(f"\n===== CONSISTENCY over {n} seeds {c['seeds']} =====")
    print("CORRECT rate per seed: " + ", ".join(f"seed {s}: {r:.1%}" for s, r in c["correct_rate_per_seed"].items()))
    print(f"mean CORRECT rate    : {c['mean_correct_rate']:.1%}")
    print("questions CORRECT in k of the seeds: "
          + ", ".join(f"{k}/{n}: {q}" for k, q in c["correct_in_k_seeds"].items()))
    for qid in c["unstable"]:
        print(f"  unstable {qid}: {' '.join(c['verdicts'][qid])}")


# ---------- judge vs human ----------

HUMAN_SHORTCUTS = {"C": "CORRECT", "P": "PARTIAL", "I": "INCORRECT"}
JUDGE_TRUST_THRESHOLD = 0.85  # spec decision 3: below this, the judge is reported as unreliable


def load_human_labels(path: Path) -> dict[str, dict]:
    """The filled-in review sheet: {id: {"label": verdict, "note": ...}}. Accepts C/P/I as shortcuts.
    Every row must be graded: a missing or unknown label stops with the list of ids to fix."""
    with path.open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    labels, bad = {}, []
    for row in rows:
        raw = (row.get("human") or "").strip().upper()
        label = HUMAN_SHORTCUTS.get(raw, raw)
        if label not in VERDICTS:
            bad.append(f"{row['id']}={raw or '(empty)'}")
        labels[row["id"]] = {"label": label, "note": (row.get("note") or "").strip()}
    if bad:
        raise SystemExit(f"{path}: grade every row as CORRECT/PARTIAL/INCORRECT (or C/P/I). Fix: {', '.join(bad)}")
    return labels


def cohen_kappa(pairs: list[tuple[str, str]]) -> float:
    """Agreement beyond chance. po = observed agreement, pe = agreement expected if both graders labelled at
    random with their own label frequencies. kappa = (po - pe) / (1 - pe): 1 perfect, 0 = chance level."""
    n = len(pairs)
    po = sum(a == b for a, b in pairs) / n
    first, second = Counter(a for a, _ in pairs), Counter(b for _, b in pairs)
    pe = sum(first[label] * second[label] for label in VERDICTS) / (n * n)
    return 1.0 if pe == 1 else (po - pe) / (1 - pe)


def compare_with_human(results: list[dict], labels: dict[str, dict]) -> dict:
    """Judge verdicts vs human labels on the same answers. Judge ERRORs are excluded from agreement
    (there is no judge opinion to compare) and listed separately."""
    from scripts.judge_sanity import RANK
    missing = sorted(r["id"] for r in results if r["id"] not in labels)
    if missing:
        raise SystemExit(f"No human label for: {', '.join(missing)} (is it the same seed's review sheet?)")
    compared = [r for r in results if r["verdict"] != ERROR]
    pairs = [(labels[r["id"]]["label"], r["verdict"]) for r in compared]  # (human, judge)
    n = len(pairs)
    human_correct = sum(labels[r["id"]]["label"] == "CORRECT" for r in results)
    return {
        "n": n,
        "judge_errors": sorted(r["id"] for r in results if r["verdict"] == ERROR),
        "exact_agreement": sum(h == j for h, j in pairs) / n if n else 0.0,
        "correct_agreement": sum((h == "CORRECT") == (j == "CORRECT") for h, j in pairs) / n if n else 0.0,
        "kappa": cohen_kappa(pairs) if n else 0.0,
        "too_lenient": sum(RANK[j] > RANK[h] for h, j in pairs),  # judge grades higher than the human
        "too_strict": sum(RANK[j] < RANK[h] for h, j in pairs),
        "confusion": Counter(pairs),
        "human_correct_rate": human_correct / len(results),
        "judge_correct_rate": sum(r["verdict"] == "CORRECT" for r in results) / len(results),
        "disagreements": [
            {"id": r["id"], "question": r["question"], "answer": r["answer"], "expected": r["expected_answer"],
             "human": labels[r["id"]]["label"], "judge": r["verdict"], "judge_reason": r["reason"],
             "human_note": labels[r["id"]]["note"]}
            for r in compared if labels[r["id"]]["label"] != r["verdict"]
        ],
    }


def print_comparison(c: dict) -> None:
    trusted = c["exact_agreement"] >= JUDGE_TRUST_THRESHOLD
    print(f"\n===== JUDGE vs HUMAN ({c['n']} answers compared) =====")
    print(f"exact agreement (3 verdicts) : {c['exact_agreement']:.1%}")
    print(f"agreement on CORRECT vs not  : {c['correct_agreement']:.1%}   (this is what moves the score)")
    print(f"Cohen's kappa                : {c['kappa']:.2f}   (1 = perfect, 0 = chance level)")
    print(f"judge too lenient / too strict: {c['too_lenient']} / {c['too_strict']}")
    if c["judge_errors"]:
        print(f"not compared (judge ERROR)   : {', '.join(c['judge_errors'])}")
    print(f"CORRECT rate: human {c['human_correct_rate']:.1%} | judge {c['judge_correct_rate']:.1%}")
    print(f"-> judge {'TRUSTED' if trusted else 'NOT TRUSTED'} (threshold {JUDGE_TRUST_THRESHOLD:.0%} exact agreement); "
          f"official score = {'judge (human-checked)' if trusted else 'HUMAN grades'}")
    print("\n  human \\ judge " + "".join(f"{v:>11}" for v in VERDICTS))
    for h in VERDICTS:
        print(f"  {h:<14}" + "".join(f"{c['confusion'][(h, j)]:>11}" for j in VERDICTS))
    for d in c["disagreements"]:
        print(f"\n--- {d['id']}: human {d['human']} vs judge {d['judge']}")
        print(f"Q      : {d['question']}")
        print(f"A      : {d['answer'].strip()}")
        print(f"expect : {d['expected']}")
        print(f"judge  : {d['judge_reason']}")
        if d["human_note"]:
            print(f"human  : {d['human_note']}")


# ---------- main ----------

def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", nargs="+", type=int, default=[config.LLM_SEED], help="one full run per seed")
    parser.add_argument("--only", nargs="+", help="question ids (saved as *_partial.json)")
    parser.add_argument("--judge-model", default=config.CHAT_MODEL)
    parser.add_argument("--rejudge", nargs="+", type=Path, help="saved eval_seed*.json files to re-grade")
    parser.add_argument("--compare-human", nargs=2, type=Path, metavar=("LABELS_CSV", "EVAL_JSON"),
                        help="compare the judge with human grades (no LLM calls)")
    args = parser.parse_args()

    if args.compare_human:
        labels_path, eval_path = args.compare_human
        results = json.loads(eval_path.read_text(encoding="utf-8"))["results"]
        print_comparison(compare_with_human(results, load_human_labels(labels_path)))
        return

    from src.eval.judge import create_judge
    judge = create_judge(args.judge_model)

    runs: dict[int, list[dict]] = {}
    if args.rejudge:
        for path in args.rejudge:
            data = json.loads(path.read_text(encoding="utf-8"))
            print(f"Re-judging {len(data['results'])} saved answers from {path} with {args.judge_model}...",
                  flush=True)
            results = rejudge(data["results"], judge, args.judge_model)
            for r in results:
                if r["verdict"] != r["previous"]["verdict"]:
                    print(f"  {r['id']}: {r['previous']['verdict']} -> {r['verdict']} | {r['reason']}")
            save(path, {**{k: v for k, v in data.items() if k != "results"}, "judge_model": args.judge_model},
                 results)
            runs[data["seed"]] = results
            print_summary(f"{path.name} (re-judged)", summarize(results))
    else:
        from src.chat.engine import create_engine
        from src.chat.llm import OllamaChat
        rows = load_eval_rows(config.EVAL_DIR / "eval_questions.csv")
        if args.only:
            rows = [r for r in rows if r["id"] in args.only]
        print(f"{len(rows)} questions x seeds {args.seeds} | chat {config.CHAT_MODEL} | judge {args.judge_model} "
              f"(temperature 0) | retrieval {config.RETRIEVAL_MODE}", flush=True)
        engine = create_engine()
        for seed in args.seeds:
            # Swap only the chat model client: same retriever and reranker (loading those takes ~20s).
            engine.llm = OllamaChat(seed=seed)
            warm_up(engine.llm)
            path = ARTIFACTS / f"eval_seed{seed}{'_partial' if args.only else ''}.json"
            meta = {"seed": seed, "chat_model": config.CHAT_MODEL, "judge_model": args.judge_model,
                    "retrieval_mode": config.RETRIEVAL_MODE, "temperature": config.LLM_TEMPERATURE}
            print(f"\n######## seed {seed} ########", flush=True)
            start = time.perf_counter()
            results = []
            for row in rows:
                results.append(evaluate_one(engine, judge, args.judge_model, row, seed))
                print_result(results[-1])
                save(path, meta, results)  # after every question: a crash keeps what was done
            runs[seed] = results
            print_summary(f"seed {seed} ({time.perf_counter() - start:.0f}s total)", summarize(results))
            print(f"Saved: {path.relative_to(config.ROOT)}")
        if not args.only and write_review_csv(REVIEW_CSV, runs[args.seeds[0]]):
            print(f"Review sheet for human grading: {REVIEW_CSV.relative_to(config.ROOT)} (seed {args.seeds[0]})")

    if len(runs) > 1:
        print_consistency(consistency(runs))


if __name__ == "__main__":
    main()
