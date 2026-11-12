"""Baseline retrieval check: is the expected source document in the top-k chunks?

Retrieval only: no LLM, no answer grading (that's M6).

Usage:
  python -m scripts.retrieval_check                       # all three models
  python -m scripts.retrieval_check --models bge-m3 -k 5
"""

import argparse
import csv
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from src import config
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all
from src.retrieval.embed import embed
from src.retrieval.vector_store import build_index, collection_name, get_client, search

DEFAULT_MODELS = ["bge-m3", "nomic-embed-text", "qwen3-embedding:0.6b"]


@dataclass(frozen=True)
class Question:
    id: str
    type: str
    question: str
    sources: frozenset[str]  # expected source documents


@dataclass(frozen=True)
class Result:
    question: Question
    returned: list[str]  # source of each of the top-k chunks, best first
    rank: int | None     # rank of the first chunk from an expected source (None = miss)

    @property
    def hit(self) -> bool:
        return self.rank is not None

    @property
    def all_found(self) -> bool:
        return self.question.sources <= set(self.returned)

    @property
    def reciprocal_rank(self) -> float:
        return 1 / self.rank if self.rank else 0.0


def load_questions(path: Path) -> list[Question]:
    """Questions with at least one expected source (skips the 'none' ones: out-of-scope etc.)."""
    with path.open(encoding="utf-8-sig") as f:  # the CSV starts with a BOM; "-sig" strips it
        rows = list(csv.DictReader(f))
    questions = []
    for row in rows:
        sources = frozenset(s.strip() for s in row["source_documents"].split(";") if s.strip() != "none")
        if sources:
            questions.append(Question(row["id"], row["type"], row["question"], sources))
    return questions


def first_hit_rank(expected: frozenset[str], returned: list[str]) -> int | None:
    for rank, source in enumerate(returned, start=1):
        if source in expected:
            return rank
    return None


def summarize(results: list[Result]) -> dict:
    n = len(results)
    return {
        "hits": sum(r.hit for r in results),
        "all_found": sum(r.all_found for r in results),
        "mrr": sum(r.reciprocal_rank for r in results) / n,
        "n": n,
    }


def run_model(client, model: str, questions: list[Question], k: int) -> list[Result]:
    collection = collection_name(model)
    embed_fn = lambda text, kind: embed(text, kind, model)  # noqa: E731
    if not client.collection_exists(collection):
        print(f"  collection {collection} missing -> indexing 50 chunks (slow on CPU)...", flush=True)
        build_index(client, collection, chunk_all(load_all(config.DATA_DIR)), embed_fn)

    results = []
    for q in questions:
        returned = [hit.source for hit in search(client, collection, q.question, embed_fn, k=k)]
        results.append(Result(q, returned, first_hit_rank(q.sources, returned)))
    return results


def print_report(model: str, results: list[Result], k: int, seconds: float) -> None:
    s = summarize(results)
    print(f"\n=== {model} ===  ({seconds:.0f}s)")
    print(f"hit@{k}         : {s['hits']}/{s['n']} = {s['hits'] / s['n']:.1%}")
    print(f"all sources@{k} : {s['all_found']}/{s['n']}")
    print(f"MRR           : {s['mrr']:.3f}")

    by_type: dict[str, list[Result]] = defaultdict(list)
    for r in results:
        by_type[r.question.type].append(r)
    print("by type:")
    for qtype, rs in by_type.items():
        print(f"  {qtype:<17} {sum(r.hit for r in rs)}/{len(rs)}")

    misses = [r for r in results if not r.hit]
    partial = [r for r in results if r.hit and not r.all_found]
    for title, group in (("MISSES", misses), ("PARTIAL (multi-source, not all found)", partial)):
        if group:
            print(f"{title}:")
            for r in group:
                print(f"  {r.question.id} [{r.question.type}] {r.question.question}")
                print(f"      expected: {', '.join(sorted(r.question.sources))}")
                print(f"      got     : {', '.join(r.returned)}")


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Retrieval hit@k on the eval questions.")
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    questions = load_questions(config.EVAL_DIR / "eval_questions.csv")
    print(f"{len(questions)} questions with an expected source document, top-{args.k}")

    summary = {}
    client = get_client()
    try:
        for model in args.models:
            start = time.perf_counter()
            results = run_model(client, model, questions, args.k)
            print_report(model, results, args.k, time.perf_counter() - start)
            summary[model] = summarize(results)
    finally:
        client.close()

    print(f"\n{'model':<22} {'hit@' + str(args.k):>8} {'all@' + str(args.k):>8} {'MRR':>7}")
    for model, s in summary.items():
        print(f"{model:<22} {s['hits'] / s['n']:>8.1%} {s['all_found']:>5}/{s['n']} {s['mrr']:>7.3f}")


if __name__ == "__main__":
    main()
