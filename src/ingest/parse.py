"""Step 1 of ingestion: turn every file in data/ into a markdown Document.

Markdown files are read as-is. The PDF is converted into the same markdown
format (headings, paragraphs, markdown table), so one chunker can handle all files.
"""

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import pdfplumber


@dataclass(frozen=True)
class Document:
    source: str  # filename, e.g. "employee_handbook_v3.md" (used for citations)
    title: str   # text of the first "# " heading
    text: str    # the whole document as markdown
    status: str  # "current" or "superseded"


# ---------- Markdown ----------

def load_markdown(path: Path) -> str:
    # Explicit utf-8: the Windows default (cp1252) would corrupt the Bangla file.
    # "utf-8-sig" also strips an invisible BOM character if a file starts with one.
    return path.read_text(encoding="utf-8-sig")


# ---------- PDF ----------

def table_to_markdown(rows: list[list[str | None]]) -> str:
    """Render extracted table rows as a markdown table. First row is the header."""
    clean = [[_clean_cell(cell) for cell in row] for row in rows]
    header, *body = clean
    lines = [
        "| " + " | ".join(header) + " |",
        "|" + "|".join("---" for _ in header) + "|",
    ]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def _clean_cell(cell: str | None) -> str:
    # Empty cells come back as None. Wrapped cells contain "\n": join them with a space.
    if cell is None:
        return ""
    return " ".join(cell.split())


def _line_size(line: dict) -> float:
    return round(line["chars"][0]["size"], 1)


def _inside(line: dict, bbox: tuple[float, float, float, float]) -> bool:
    x0, top, x1, bottom = bbox
    return line["x0"] >= x0 and line["x1"] <= x1 and line["top"] >= top and line["bottom"] <= bottom


def _page_items(page) -> list[tuple[float, str, object]]:
    """Return the page's content as (top_position, kind, payload), in reading order.

    Tables are extracted as rows. Text lines that sit inside a table area are
    skipped, otherwise the table would appear twice (once jumbled).
    """
    tables = page.find_tables()
    items = [(t.bbox[1], "table", t.extract()) for t in tables]
    for line in page.extract_text_lines():
        if not any(_inside(line, t.bbox) for t in tables):
            items.append((line["top"], "line", line))
    items.sort(key=lambda item: item[0])
    return items


def pdf_to_markdown(path: Path) -> str:
    with pdfplumber.open(path) as pdf:
        pages = [_page_items(page) for page in pdf.pages]

    # Decide heading levels from font size, across the whole document:
    #   biggest font -> "# " title, most common font -> body text,
    #   anything in between -> "## " section heading.
    sizes = [_line_size(p) for items in pages for (_, kind, p) in items if kind == "line"]
    title_size = max(sizes)
    body_size = Counter(sizes).most_common(1)[0][0]

    blocks: list[str] = []
    paragraph: list[str] = []
    prev_bottom = None

    def flush() -> None:
        nonlocal prev_bottom
        if paragraph:
            blocks.append(" ".join(paragraph))
            paragraph.clear()
        prev_bottom = None

    for items in pages:
        flush()  # a new page starts a new paragraph
        for _, kind, payload in items:
            if kind == "table":
                flush()
                blocks.append(table_to_markdown(payload))
                continue

            text = payload["text"].strip()
            size = _line_size(payload)
            if size == title_size and title_size > body_size:
                flush()
                blocks.append("# " + text)
            elif body_size < size < title_size:
                flush()
                blocks.append("## " + text)
            else:
                # A vertical gap bigger than the font size means a new paragraph.
                if prev_bottom is not None and payload["top"] - prev_bottom > size:
                    flush()
                paragraph.append(text)
                prev_bottom = payload["bottom"]
    flush()

    return "\n\n".join(blocks) + "\n"


# ---------- Any file -> Document ----------

def _extract_title(text: str, fallback: str) -> str:
    for line in text.splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return fallback


def load_document(path: Path) -> Document:
    if path.suffix.lower() == ".pdf":
        text = pdf_to_markdown(path)
    else:
        text = load_markdown(path)
    status = "superseded" if "SUPERSEDED" in path.name.upper() else "current"
    return Document(
        source=path.name,
        title=_extract_title(text, fallback=path.stem),
        text=text,
        status=status,
    )


def load_all(data_dir: Path) -> list[Document]:
    """Load every .md and .pdf file in data_dir, sorted by filename (deterministic order)."""
    paths = sorted(p for p in data_dir.iterdir() if p.suffix.lower() in {".md", ".pdf"})
    return [load_document(p) for p in paths]
