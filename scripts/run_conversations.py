"""Run eval/conversation_tests.json through the real chat engine and check each turn.

Checks are CRUDE (substring matching on key facts + "cites a retrieved file"). They catch obvious
failures; they are not the M6 evaluation. Read every answer too.

Usage:  python -u -m scripts.run_conversations [--mode rerank]
"""

import argparse
import json
import re
import sys

from src import config
from src.chat.engine import create_engine

# Key facts per (conversation, turn), from the spec (docs/specs/M4.md). Case-insensitive substrings.
MUST_CONTAIN = {
    ("C1", 1): ["18"], ("C1", 2): ["5", "31 March"], ("C1", 3): ["14"],
    ("C2", 1): ["2,500"], ("C2", 2): ["Enterprise"], ("C2", 3): ["60"],
    ("C3", 1): ["Retry-After"], ("C3", 2): ["600"], ("C3", 3): ["X-Nodi-Signature"],
}
# The superseded value may only appear when it is called outdated (the system prompt asks for that,
# and eval Q29's expected answer does it too: "The old 15 days ... is superseded").
# Changed after the first run, which wrongly failed a correct answer; see docs/experiments.md (E3).
SUPERSEDED_VALUES = {("C1", 1): "15"}
# Whole words only ("old" must not match "hold"), and searched with citations removed:
# the filename "leave_policy_..._SUPERSEDED.md" must not count as calling the value outdated.
OUTDATED_RE = re.compile(r"\b(outdated|superseded|old|previous|no longer)\b")

CITATION_RE = re.compile(r"\[([^\[\]\s]+\.(?:md|pdf))\]")


def check(key: tuple[str, int], answer: str, retrieved: set[str]) -> dict:
    low = answer.lower()
    missing = [f for f in MUST_CONTAIN.get(key, []) if f.lower() not in low]
    forbidden = []
    old = SUPERSEDED_VALUES.get(key)
    if old and old in low and not OUTDATED_RE.search(CITATION_RE.sub("", answer).lower()):
        forbidden.append(f"{old} (given as the answer, not marked outdated)")
    cited = set(CITATION_RE.findall(answer))
    return {
        "facts_ok": not missing and not forbidden,
        "missing": missing,
        "forbidden": forbidden,
        "cites_retrieved": bool(cited & retrieved),
        "cited": sorted(cited),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default=config.RETRIEVAL_MODE)
    args = parser.parse_args()

    print(f"Loading engine (mode={args.mode}, model={config.CHAT_MODEL})...", flush=True)
    engine = create_engine(args.mode)
    conversations = json.loads((config.EVAL_DIR / "conversation_tests.json").read_text(encoding="utf-8"))

    rows = []
    saved = []  # full results, written to artifacts/ so answers can be re-checked without re-running
    for conv in conversations:
        session = f"test-{conv['id']}"
        engine.reset(session)
        for i, turn in enumerate(conv["turns"], start=1):
            events = list(engine.ask(session, turn["user"]))
            rewrite = next(e for e in events if e["type"] == "rewrite")
            sources = next(e for e in events if e["type"] == "sources")["sources"]
            done = events[-1]
            result = check((conv["id"], i), done["answer"], {s["source"] for s in sources})
            rows.append((conv["id"], i, result, done["timing"]))
            saved.append({"conversation": conv["id"], "turn": i, "user": turn["user"],
                          "rewritten": rewrite["query"], "sources": sources, "answer": done["answer"],
                          "expected": turn["expected"], "check": result, "timing": done["timing"]})

            t = done["timing"]
            print(f"\n=== {conv['id']} turn {i} ===")
            print(f"user     : {turn['user']}")
            if rewrite["rewritten"]:
                print(f"rewritten: {rewrite['query']}")
            print(f"sources  : {', '.join(s['source'] + ' > ' + s['section'] for s in sources)}")
            print(f"answer   : {done['answer'].strip()}")
            print(f"expected : {turn['expected']}")
            print(f"check    : facts {'OK' if result['facts_ok'] else 'FAIL'}"
                  f"{' missing=' + str(result['missing']) if result['missing'] else ''}"
                  f"{' forbidden=' + str(result['forbidden']) if result['forbidden'] else ''}"
                  f" | citation {'OK' if result['cites_retrieved'] else 'FAIL'} {result['cited']}")
            print(f"timing   : rewrite {t['rewrite_s']}s | retrieve {t['retrieve_s']}s | "
                  f"first token {t.get('first_token_s', '-')}s | total {t['total_s']}s | "
                  f"prompt {t.get('prompt_tokens', '-')} tok | output {t.get('output_tokens', '-')} tok", flush=True)

    out = config.ROOT / "artifacts" / "conversation_results.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(saved, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved: {out.relative_to(config.ROOT)}")

    n = len(rows)
    facts = sum(r[2]["facts_ok"] for r in rows)
    cites = sum(r[2]["cites_retrieved"] for r in rows)
    totals = [r[3]["total_s"] for r in rows]
    firsts = [r[3]["first_token_s"] for r in rows if "first_token_s" in r[3]]
    print(f"\nSUMMARY  key facts {facts}/{n} | citations {cites}/{n} | "
          f"first token avg {sum(firsts) / len(firsts):.1f}s | total avg {sum(totals) / n:.1f}s, max {max(totals):.1f}s")


if __name__ == "__main__":
    main()
