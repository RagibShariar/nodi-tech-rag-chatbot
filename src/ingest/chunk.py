"""Step 2 of ingestion: split each markdown Document into Chunks.

Rules (see docs/specs/M1.md):
- One chunk per "##"/"###" section. Text before the first "##" is the "(intro)" section.
- A section longer than MAX_CHARS is split: by paragraph/table blocks, then by lines,
  then by words. A table is split by rows, and every piece repeats the header row.
- Every chunk gets a context line ([Source | title | Section]) prepended in embed_text.
"""

import re
import textwrap
from dataclasses import dataclass

from src.ingest.parse import Document

MAX_CHARS = 1500

HEADING_RE = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str    # "<source>#<n>", stable across runs
    source: str      # filename, used for citations
    doc_title: str
    section: str     # heading path, e.g. "3. Leave > Carry forward"
    status: str      # "current" or "superseded"
    content: str     # the section text (no heading line)
    embed_text: str  # context line + content: what gets embedded / BM25-indexed


# ---------- 1. Document -> sections ----------

def split_sections(text: str) -> list[tuple[str, str]]:
    """Return (section_path, content) pairs, in document order. Empty sections are dropped."""
    sections: list[tuple[str, str]] = []
    path: list[str] = []
    lines: list[str] = []

    def flush() -> None:
        content = "\n".join(lines).strip()
        if content:
            sections.append((" > ".join(path) or "(intro)", content))
        lines.clear()

    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if not match:
            lines.append(line)
            continue
        level, heading = len(match.group(1)), match.group(2)
        flush()
        if level == 1:
            continue  # the document title; already stored in Document.title
        # "##" is depth 0, "###" is depth 1, ... Keep the parents, replace the rest.
        path = path[: level - 2] + [heading]
    flush()
    return sections


# ---------- 2. Section -> pieces of at most MAX_CHARS ----------

def _is_table_row(line: str) -> bool:
    return line.lstrip().startswith("|")


def _blocks(content: str) -> list[str]:
    """Split content into blocks: paragraphs (separated by blank lines) and whole tables."""
    blocks: list[str] = []
    current: list[str] = []
    in_table = False
    for line in content.split("\n"):
        if not line.strip():
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        is_row = _is_table_row(line)
        if current and is_row != in_table:  # switching between table and text
            blocks.append("\n".join(current))
            current = []
        current.append(line)
        in_table = is_row
    if current:
        blocks.append("\n".join(current))
    return blocks


def _pack(units: list[str], sep: str, limit: int) -> list[str]:
    """Greedily join units with sep into pieces of at most `limit` chars."""
    pieces: list[str] = []
    current = ""
    for unit in units:
        candidate = unit if not current else current + sep + unit
        if len(candidate) <= limit:
            current = candidate
        else:
            if current:
                pieces.append(current)
            current = unit
    if current:
        pieces.append(current)
    return pieces


def _split_table(table: str, limit: int) -> list[str]:
    """Split a markdown table by rows. Every piece starts with the header + separator line."""
    lines = table.split("\n")
    header = "\n".join(lines[:2])
    row_limit = limit - len(header) - 1  # -1 for the "\n" after the header
    return [header + "\n" + rows for rows in _pack(lines[2:], "\n", row_limit)]


def _split_block(block: str, limit: int) -> list[str]:
    if len(block) <= limit:
        return [block]
    if all(_is_table_row(line) for line in block.split("\n")):
        return _split_table(block, limit)
    lines = block.split("\n")
    if len(lines) > 1:
        units = [piece for line in lines for piece in _split_block(line, limit)]
        return _pack(units, "\n", limit)
    return textwrap.wrap(block, limit)  # one very long line: break between words


def split_content(content: str, limit: int = MAX_CHARS) -> list[str]:
    if len(content) <= limit:
        return [content]  # common case: keep the original text untouched
    units = [piece for block in _blocks(content) for piece in _split_block(block, limit)]
    return _pack(units, "\n\n", limit)


# ---------- 3. Build Chunks ----------

def context_line(doc: Document, section: str) -> str:
    parts = [f"Source: {doc.source}", doc.title, f"Section: {section}"]
    if doc.status == "superseded":
        parts.append("Status: SUPERSEDED")
    return "[" + " | ".join(parts) + "]"


def chunk_document(doc: Document) -> list[Chunk]:
    chunks: list[Chunk] = []
    for section, content in split_sections(doc.text):
        for piece in split_content(content):
            chunks.append(
                Chunk(
                    chunk_id=f"{doc.source}#{len(chunks)}",
                    source=doc.source,
                    doc_title=doc.title,
                    section=section,
                    status=doc.status,
                    content=piece,
                    embed_text=context_line(doc, section) + "\n" + piece,
                )
            )
    return chunks


def chunk_all(docs: list[Document]) -> list[Chunk]:
    return [chunk for doc in docs for chunk in chunk_document(doc)]
