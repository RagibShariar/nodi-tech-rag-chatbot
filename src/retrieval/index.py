"""Indexing CLI: data/ -> chunks -> embeddings -> Qdrant collection.

Usage:  python -m src.retrieval.index [--model bge-m3]
"""

import argparse
import sys
import time

from src import config
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all
from src.retrieval.embed import embed
from src.retrieval.vector_store import build_index, collection_name, get_client


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="Embed all chunks into Qdrant.")
    parser.add_argument("--model", default=config.EMBED_MODEL, help="Ollama embedding model")
    args = parser.parse_args()

    chunks = chunk_all(load_all(config.DATA_DIR))
    collection = collection_name(args.model)
    client = get_client()
    try:
        start = time.perf_counter()
        tokens = build_index(client, collection, chunks, lambda text, kind: embed(text, kind, args.model))
        seconds = time.perf_counter() - start
        count = client.count(collection).count
    finally:
        client.close()  # local mode locks the folder until the client is closed

    biggest = max(zip(tokens, chunks), key=lambda pair: pair[0])
    print(f"model       : {args.model}")
    print(f"collection  : {collection}  ({count} points)")
    print(f"time        : {seconds:.1f}s for {len(chunks)} chunks")
    print(f"tokens/chunk: min {min(tokens)}, max {max(tokens)}, avg {sum(tokens) // len(tokens)}, total {sum(tokens)}")
    print(f"biggest     : {biggest[1].chunk_id} ({biggest[0]} tokens)")


if __name__ == "__main__":
    main()
