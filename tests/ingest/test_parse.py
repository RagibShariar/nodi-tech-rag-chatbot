from pathlib import Path

import pytest

from src.ingest.parse import load_all, table_to_markdown

DATA_DIR = Path(__file__).resolve().parents[2] / "data"  # tests/ingest/ -> project root


@pytest.fixture(scope="module")
def docs():
    return {d.source: d for d in load_all(DATA_DIR)}


@pytest.fixture(scope="module")
def pdf_text(docs):
    return docs["expense_reimbursement_policy.pdf"].text


def test_all_12_files_loaded(docs):
    assert len(docs) == 12
    assert all(d.text.strip() for d in docs.values())


def test_titles_come_from_h1(docs):
    assert docs["employee_handbook_v3.md"].title == "Nodi Tech Ltd. — Employee Handbook"
    assert docs["expense_reimbursement_policy.pdf"].title == "Nodi Tech Ltd. — Expense Reimbursement Policy"


def test_only_superseded_file_is_marked_superseded(docs):
    superseded = {s for s, d in docs.items() if d.status == "superseded"}
    assert superseded == {"leave_policy_v2.1_2024_SUPERSEDED.md"}


def test_bangla_text_preserved(docs):
    text = docs["office_transport_canteen_bn.md"].text
    assert "৮:১৫" in text
    assert "ক্যান্টিন" in text


def test_no_windows_line_endings(docs):
    assert all("\r" not in d.text for d in docs.values())


# ---------- PDF ----------

def test_pdf_headings_detected_from_font_size(pdf_text):
    lines = pdf_text.splitlines()
    assert lines[0] == "# Nodi Tech Ltd. — Expense Reimbursement Policy"
    for heading in ["1. How to claim", "2. Limits", "3. Approvals", "4. Not reimbursable"]:
        assert f"## {heading}" in lines


def test_pdf_table_is_a_markdown_table_with_header(pdf_text):
    lines = pdf_text.splitlines()
    assert "| Expense type | Limit | Notes |" in lines
    assert "| Hotel outside Dhaka | BDT 6,000 per night | Pre-approval from team lead required |" in lines
    assert "| Local transport (ride-share, CNG) | BDT 1,500 per day | Business travel only, not daily commute |" in lines


def test_pdf_table_rows_all_present(pdf_text):
    rows = [l for l in pdf_text.splitlines() if l.startswith("| ") and "---" not in l]
    assert len(rows) == 6  # header + 5 rows


def test_pdf_table_text_not_duplicated(pdf_text):
    assert pdf_text.count("Hotel outside Dhaka") == 1


def test_pdf_paragraph_lines_are_joined(pdf_text):
    assert "with a photo or scan of the receipt." in pdf_text


def test_pdf_footer_is_separate_paragraph(pdf_text):
    # The small footer must not be glued onto the last sentence of section 4.
    assert "\n\nQuestions: finance@noditech.example · Extension 230" in pdf_text


# ---------- table_to_markdown unit test ----------

def test_table_to_markdown_handles_wrapped_and_empty_cells():
    rows = [["A", "B"], ["x\ny", None]]
    assert table_to_markdown(rows) == "| A | B |\n|---|---|\n| x y |  |"
