"""Test the judge on the frozen sanity set (scripts/judge_sanity.json): cases with a known verdict.

The set was written and its hash recorded BEFORE the judge's first run (docs/experiments.md E5). The
script refuses to run if the file changed, so the cases can't be quietly edited to make the judge pass.

Usage:  python -u -m scripts.judge_sanity [--model qwen3.5:4b] [--out judge_sanity_results.json]
"""

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

from src import config
from src.eval.judge import ERROR, VERDICTS

SANITY_FILE = Path(__file__).with_name("judge_sanity.json")
FROZEN_SHA256 = "917cb2ec647b8bc97ada2cc0b55fb14ad907cb1020b505968dedcecf0dce9dfa"

# Ordered from worst to best, to tell a lenient judge (grades too high) from a strict one (too low).
RANK = {"INCORRECT": 0, "PARTIAL": 1, "CORRECT": 2}


def file_sha256(path: Path) -> str:
    """Hash of the LF version (what git stores); a Windows checkout may turn the working copy into CRLF."""
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def load_cases(path: Path = SANITY_FILE, expected_sha: str = FROZEN_SHA256) -> list[dict]:
    actual = file_sha256(path)
    if actual != expected_sha:
        raise SystemExit(f"{path.name} changed since it was frozen (sha256 {actual}, expected {expected_sha}). "
                         "Don't edit a frozen test set; if a change is really needed, disclose it in "
                         "docs/experiments.md and update the hash.")
    return json.loads(path.read_text(encoding="utf-8"))


def summarize(rows: list[dict]) -> dict:
    """rows: [{"expected": verdict, "got": verdict or ERROR}, ...]"""
    errors = sum(r["got"] == ERROR for r in rows)
    graded = [r for r in rows if r["got"] != ERROR]
    return {
        "total": len(rows),
        "agree": sum(r["got"] == r["expected"] for r in rows),
        "errors": errors,
        "too_lenient": sum(RANK[r["got"]] > RANK[r["expected"]] for r in graded),  # inflates the score
        "too_strict": sum(RANK[r["got"]] < RANK[r["expected"]] for r in graded),
        "confusion": Counter((r["expected"], r["got"]) for r in rows),
    }


def print_summary(s: dict) -> None:
    print(f"\nSUMMARY: agree {s['agree']}/{s['total']} | too lenient {s['too_lenient']} | "
          f"too strict {s['too_strict']} | unreadable (ERROR) {s['errors']}")
    columns = (*VERDICTS, ERROR)
    print("\n  expected \\ judge " + "".join(f"{c:>11}" for c in columns))
    for expected in VERDICTS:
        print(f"  {expected:<17}" + "".join(f"{s['confusion'][(expected, got)]:>11}" for got in columns))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=config.CHAT_MODEL, help="judge model in Ollama")
    parser.add_argument("--out", default="judge_sanity_results.json")
    args = parser.parse_args()

    from src.eval.judge import create_judge
    cases = load_cases()
    judge = create_judge(args.model)
    print(f"{len(cases)} sanity cases, judge={args.model}, temperature=0", flush=True)

    rows = []
    for case in cases:
        start = time.perf_counter()
        j = judge(case["question"], case["expected_answer"], case["answer"])
        seconds = time.perf_counter() - start
        ok = j.verdict == case["verdict"]
        rows.append({"id": case["id"], "label": case["label"], "expected": case["verdict"],
                     "got": j.verdict, "reason": j.reason, "raw": j.raw, "seconds": round(seconds, 1)})
        print(f"{'ok  ' if ok else 'MISS'} {case['id']} {case['label']:<46} expected {case['verdict']:<9} "
              f"judge {j.verdict:<9} {seconds:5.1f}s | {j.reason}", flush=True)

    out = config.ROOT / "artifacts" / args.out
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"model": args.model, "rows": rows}, ensure_ascii=False, indent=2), encoding="utf-8")
    print_summary(summarize(rows))
    print(f"\nSaved: {out.relative_to(config.ROOT)} | avg {sum(r['seconds'] for r in rows) / len(rows):.1f}s per judgement")


if __name__ == "__main__":
    main()
