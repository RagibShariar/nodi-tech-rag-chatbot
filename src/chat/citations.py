"""Turn the bot's [file.md] citations into the numbered source list the chat page shows.

The model cites files by name, e.g. "18 days [employee_handbook_v3.md]". The page shows them as [1], [2]
chips that point to source cards. Numbers follow the order of first citation in the answer.

Only files that were actually retrieved for this answer get a card: a citation of a file we didn't give the
model can't be backed by a snippet, so the page leaves it as plain text.

The card's snippet is the line of the retrieved text that shares the most words with the answer (a table
row is shown with its column names), so the user sees the evidence, not just the start of the section.
"""

import re

CITATION_RE = re.compile(r"\[([^\[\]\s]+\.(?:md|pdf))\]")
SNIPPET_CHARS = 180
_WORD_RE = re.compile(r"\w+")
_TABLE_SEPARATOR_RE = re.compile(r"^\|?\s*:?-{3,}")


def snippet(text: str, limit: int = SNIPPET_CHARS) -> str:
    """One line, at most `limit` characters, cut at a word boundary."""
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    return flat[:limit].rsplit(" ", 1)[0] + "…"


def _cells(row: str) -> list[str]:
    return [c.strip() for c in row.strip().strip("|").split("|")]


def candidate_lines(content: str) -> list[str]:
    """Readable lines of a chunk: markdown table rows become 'Column: value · Column: value'."""
    lines = content.splitlines()
    out: list[str] = []
    header: list[str] | None = None
    for i, line in enumerate(lines):
        stripped = line.strip()
        if not stripped:
            header = None
            continue
        if stripped.startswith("|"):
            if _TABLE_SEPARATOR_RE.match(stripped):
                continue
            next_line = lines[i + 1].strip() if i + 1 < len(lines) else ""
            if _TABLE_SEPARATOR_RE.match(next_line):  # this row is the table's header
                header = _cells(stripped)
                continue
            cells = _cells(stripped)
            if header and len(header) == len(cells):
                out.append(" · ".join(f"{h}: {c}" for h, c in zip(header, cells) if c))
            else:
                out.append(" · ".join(c for c in cells if c))
        else:
            header = None
            out.append(re.sub(r"^[#>*\-\s]+|\*\*", "", stripped))
    return [line for line in out if line]


def best_line(answer: str, contents: list[str]) -> tuple[int, str]:
    """(index of the chunk, line) with the most words in common with the answer; ties keep the earliest."""
    answer_words = {w.lower() for w in _WORD_RE.findall(CITATION_RE.sub("", answer))}
    best = (-1, 0, "")  # (score, chunk index, line)
    for index, content in enumerate(contents):
        for line in candidate_lines(content):
            score = len(answer_words & {w.lower() for w in _WORD_RE.findall(line)})
            if score > best[0]:
                best = (score, index, line)
    return best[1], best[2]


def cited_sources(answer: str, retrieved: list[dict]) -> list[dict]:
    """retrieved: the engine's 'sources' event items (source, section, status, content), best first.
    Returns one card per cited, retrieved file: {"id", "file", "status", "section", "snippet"}."""
    by_file: dict[str, list[dict]] = {}
    for item in retrieved:
        by_file.setdefault(item["source"], []).append(item)

    cards: list[dict] = []
    for file in CITATION_RE.findall(answer):
        if file not in by_file or any(c["file"] == file for c in cards):
            continue
        chunks = by_file[file]
        index, line = best_line(answer, [c.get("content", "") for c in chunks])
        chunk = chunks[index]
        cards.append({"id": len(cards) + 1, "file": file, "status": chunk["status"],
                      "section": chunk["section"], "snippet": snippet(line)})
    return cards


def is_answered(cards: list[dict]) -> bool:
    """The system prompt makes the model cite every fact. An answer citing no retrieved document is a
    refusal, an "I don't know" or an out-of-scope reply, in any language: the page shows the dashed box."""
    return bool(cards)
