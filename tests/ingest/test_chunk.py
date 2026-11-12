import csv
from pathlib import Path

import pytest

from src.ingest.chunk import MAX_CHARS, chunk_all, chunk_document, split_content, split_sections
from src.ingest.parse import Document, load_all

ROOT = Path(__file__).resolve().parents[2]  # tests/ingest/ -> project root
DATA_DIR = ROOT / "data"
EVAL_CSV = ROOT / "eval" / "eval_questions.csv"


@pytest.fixture(scope="module")
def chunks():
    return chunk_all(load_all(DATA_DIR))


def _containing(chunks, text, source=None):
    return [c for c in chunks if text in c.content and (source is None or c.source == source)]


# ---------- Real data ----------

def test_every_file_produces_chunks(chunks):
    assert len({c.source for c in chunks}) == 12


def test_no_empty_or_oversized_chunks(chunks):
    assert all(c.content.strip() for c in chunks)
    assert all(len(c.content) <= MAX_CHARS for c in chunks)


def test_chunk_ids_unique(chunks):
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_pdf_table_header_and_row_in_same_chunk(chunks):
    [chunk] = _containing(chunks, "| Hotel outside Dhaka | BDT 6,000 per night | Pre-approval from team lead required |")
    assert "| Expense type | Limit | Notes |" in chunk.content
    assert chunk.section == "2. Limits"


def test_pdf_wrapped_cell_joined(chunks):
    assert _containing(chunks, "| Local transport (ride-share, CNG) |")


def test_pdf_table_not_duplicated(chunks):
    assert len(_containing(chunks, "Hotel outside Dhaka", "expense_reimbursement_policy.pdf")) == 1


def test_markdown_table_row_keeps_header(chunks):
    [chunk] = _containing(chunks, "ND-1042 | 429")
    assert "| Code | HTTP status |" in chunk.content


def test_heading_path(chunks):
    [chunk] = _containing(chunks, "31 March", "employee_handbook_v3.md")
    assert chunk.section == "3. Leave > Carry forward"


def test_heading_lines_not_in_content(chunks):
    assert not any(line.startswith("#") for c in chunks for line in c.content.splitlines())


def test_intro_section(chunks):
    [chunk] = _containing(chunks, "This handbook replaces all earlier HR policy documents")
    assert chunk.section == "(intro)"


def test_bangla_preserved(chunks):
    assert _containing(chunks, "৮:১৫", "office_transport_canteen_bn.md")


def test_exact_codes_preserved(chunks):
    assert _containing(chunks, "ND-1042", "nodidesk_error_codes.md")
    assert _containing(chunks, "SEC-EXC", "it_security_policy.md")


def test_status(chunks):
    for c in chunks:
        expected = "superseded" if c.source == "leave_policy_v2.1_2024_SUPERSEDED.md" else "current"
        assert c.status == expected


def test_context_line(chunks):
    for c in chunks:
        assert c.embed_text.startswith(f"[Source: {c.source} | {c.doc_title} | Section: {c.section}")
        assert c.embed_text.endswith(c.content)
    superseded = [c for c in chunks if c.status == "superseded"]
    assert all("Status: SUPERSEDED]" in c.embed_text.splitlines()[0] for c in superseded)


def test_deterministic():
    first = chunk_all(load_all(DATA_DIR))
    second = chunk_all(load_all(DATA_DIR))
    assert first == second


def test_every_eval_source_document_is_chunked(chunks):
    sources = {c.source for c in chunks}
    with EVAL_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            for name in row["source_documents"].split(";"):
                name = name.strip()
                if name != "none":
                    assert name in sources, f"{row['id']}: {name} not chunked"


# ---------- Splitting rules (synthetic input) ----------

def test_split_sections_heading_path():
    text = "# T\n\nintro\n\n## A\na\n### B\nb\n## C\nc\n"
    assert split_sections(text) == [("(intro)", "intro"), ("A", "a"), ("A > B", "b"), ("C", "c")]


def test_large_table_split_repeats_header():
    header = "| Code | Meaning |\n|---|---|"
    rows = [f"| X-{i:03d} | meaning number {i} |" for i in range(100)]
    pieces = split_content(header + "\n" + "\n".join(rows))

    assert len(pieces) > 1
    for piece in pieces:
        assert piece.startswith(header + "\n")
        assert len(piece) <= MAX_CHARS
    # every row appears exactly once across all pieces
    found = [line for p in pieces for line in p.splitlines()[2:]]
    assert found == rows


def test_long_section_splits_at_blank_lines():
    paragraphs = [f"Paragraph {i}. " + "word " * 60 for i in range(10)]  # ~315 chars each
    pieces = split_content("\n\n".join(p.strip() for p in paragraphs))

    assert len(pieces) > 1
    for piece in pieces:
        assert len(piece) <= MAX_CHARS
    # no paragraph was cut in half: each one is fully inside exactly one piece
    for p in paragraphs:
        assert sum(p.strip() in piece for piece in pieces) == 1


def test_single_huge_line_is_split_by_words():
    pieces = split_content("word " * 1000)
    assert len(pieces) > 1
    assert all(len(p) <= MAX_CHARS for p in pieces)


def test_chunk_ids_count_per_document():
    doc = Document(source="x.md", title="X", text="# X\n## A\na\n## B\nb\n", status="current")
    assert [c.chunk_id for c in chunk_document(doc)] == ["x.md#0", "x.md#1"]
