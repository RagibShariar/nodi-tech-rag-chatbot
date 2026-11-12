"""Search CLI: question -> top-k chunks, with a per-stage debug trail.

Usage:  python -m src.retrieval.search "What does error ND-1042 mean?" [--mode rerank] [-k 5]
"""

import argparse
import sys

from src import config
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all
from src.retrieval.embed import embed
from src.retrieval.pipeline import MODES, Retrieved, Retriever
from src.retrieval.vector_store import collection_name, get_client


def trail(r: Retrieved) -> str:
    parts = []
    if r.vector_rank is not None:
        parts.append(f"vec #{r.vector_rank}")
    if r.bm25_rank is not None:
        parts.append(f"bm25 #{r.bm25_rank}")
    if r.rrf_score is not None:
        parts.append(f"rrf {r.rrf_score:.4f}")
    if r.rerank_score is not None:
        parts.append(f"rerank {r.rerank_score:.3f}")
    return " | ".join(parts)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Search the indexed chunks.")
    parser.add_argument("question")
    parser.add_argument("--mode", choices=MODES, default=config.RETRIEVAL_MODE)
    parser.add_argument("--model", default=config.EMBED_MODEL, help="embedding model")
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    rerank_fn = None
    if args.mode == "rerank":
        from src.retrieval.rerank import CrossEncoderReranker  # loads PyTorch only when needed
        rerank_fn = CrossEncoderReranker()

    client = get_client()
    try:
        retriever = Retriever(
            chunk_all(load_all(config.DATA_DIR)), client, collection_name(args.model),
            lambda text, kind: embed(text, kind, args.model), rerank_fn,
        )
        results = retriever.retrieve(args.question, mode=args.mode, k=args.k)
    finally:
        client.close()

    print(f"Q: {args.question}   [mode={args.mode}, embed={args.model}]")
    for rank, r in enumerate(results, start=1):
        flag = "  (SUPERSEDED)" if r.chunk.status == "superseded" else ""
        preview = " ".join(r.chunk.content.split())[:100]
        print(f"{rank}. {r.chunk.source} > {r.chunk.section}{flag}")
        print(f"     [{trail(r)}]")
        print(f"     {preview}")


if __name__ == "__main__":
    main()
