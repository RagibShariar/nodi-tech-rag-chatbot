"""Search CLI: question -> top-k chunks from Qdrant.

Usage:  python -m src.retrieval.search "How many days of annual leave do I get?" [--model bge-m3] [-k 5]
"""

import argparse
import sys

from src import config
from src.retrieval.embed import embed
from src.retrieval.vector_store import collection_name, get_client, search


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Vector search over the indexed chunks.")
    parser.add_argument("question")
    parser.add_argument("--model", default=config.EMBED_MODEL)
    parser.add_argument("-k", type=int, default=5)
    args = parser.parse_args()

    client = get_client()
    try:
        hits = search(client, collection_name(args.model), args.question,
                      lambda text, kind: embed(text, kind, args.model), k=args.k)
    finally:
        client.close()

    print(f"Q: {args.question}   [{args.model}]")
    for rank, hit in enumerate(hits, start=1):
        flag = "  (SUPERSEDED)" if hit.status == "superseded" else ""
        preview = " ".join(hit.content.split())[:110]
        print(f"{rank}. {hit.score:.3f}  {hit.source} > {hit.section}{flag}")
        print(f"          {preview}")


if __name__ == "__main__":
    main()
