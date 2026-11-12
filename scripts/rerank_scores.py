"""Top rerank score for every eval question: data for the relevance-gate decision (M5).

If in-scope questions always score clearly higher than out-of-scope ones, a threshold between them
can answer "I don't know" without calling the LLM.

Usage:  python -u -m scripts.rerank_scores
"""

import csv
import json
import sys

from src import config
from src.chat.engine import TOP_K
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all
from src.retrieval.embed import embed
from src.retrieval.pipeline import Retriever
from src.retrieval.rerank import CrossEncoderReranker
from src.retrieval.vector_store import collection_name, get_client


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    with (config.EVAL_DIR / "eval_questions.csv").open(encoding="utf-8-sig") as f:
        questions = list(csv.DictReader(f))

    client = get_client()
    try:
        retriever = Retriever(chunk_all(load_all(config.DATA_DIR)), client, collection_name(config.EMBED_MODEL),
                              lambda text, kind: embed(text, kind), CrossEncoderReranker())
        rows = []
        for q in questions:
            top = retriever.retrieve(q["question"], mode="rerank", k=TOP_K)[0]
            in_scope = q["source_documents"].strip() != "none"
            rows.append({"id": q["id"], "type": q["type"], "in_scope": in_scope, "score": round(top.score, 4),
                         "top": f"{top.chunk.source} > {top.chunk.section}", "question": q["question"]})
            print(f"{q['id']} {top.score:7.4f} {'IN ' if in_scope else 'OUT'} {q['question'][:60]}", flush=True)
    finally:
        client.close()

    (config.ROOT / "artifacts" / "rerank_scores.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")

    inside = sorted(r["score"] for r in rows if r["in_scope"])
    outside = sorted(r["score"] for r in rows if not r["in_scope"])
    print(f"\nIN-scope  ({len(inside)}): min {inside[0]:.4f} | lowest 5: {inside[:5]}")
    print(f"OUT-scope ({len(outside)}): max {outside[-1]:.4f} | all: {outside}")
    print("Lowest in-scope questions:")
    for r in sorted((r for r in rows if r["in_scope"]), key=lambda r: r["score"])[:5]:
        print(f"  {r['id']} {r['score']:.4f} [{r['type']}] {r['question']}  -> {r['top']}")


if __name__ == "__main__":
    main()
