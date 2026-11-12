"""Ingestion CLI: parse + chunk every file in data/, print the chunks, save them as JSONL.

Usage:  python -m src.ingest.run
"""

import json
import sys
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
OUT_FILE = ROOT / "artifacts" / "chunks.jsonl"


def main() -> None:
    # Windows consoles/pipes may default to cp1252, which crashes on Bangla text.
    sys.stdout.reconfigure(encoding="utf-8")

    chunks = chunk_all(load_all(DATA_DIR))

    for c in chunks:
        print("=" * 80)
        print(f"{c.chunk_id}  |  section: {c.section}  |  {len(c.content)} chars  |  {c.status}")
        print("-" * 80)
        print(c.content)

    OUT_FILE.parent.mkdir(exist_ok=True)
    with OUT_FILE.open("w", encoding="utf-8") as f:
        for c in chunks:
            # ensure_ascii=False keeps Bangla readable in the file instead of ন escapes
            f.write(json.dumps(asdict(c), ensure_ascii=False) + "\n")

    sizes = [len(c.content) for c in chunks]
    print("=" * 80)
    print(f"{len(chunks)} chunks from {len({c.source for c in chunks})} files -> {OUT_FILE.relative_to(ROOT)}")
    print(f"chunk size (chars): min {min(sizes)}, max {max(sizes)}, avg {sum(sizes) // len(sizes)}")
    for source, n in sorted(Counter(c.source for c in chunks).items()):
        print(f"  {n:3d}  {source}")


if __name__ == "__main__":
    main()
