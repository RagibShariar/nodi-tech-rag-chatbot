import pytest

from scripts.judge_sanity import SANITY_FILE, file_sha256, load_cases, summarize
from src.eval.judge import VERDICTS


def test_frozen_file_matches_recorded_hash_and_is_valid():
    cases = load_cases()  # raises if the file changed
    assert len(cases) == 14
    assert all(c["verdict"] in VERDICTS for c in cases)
    assert {"question", "expected_answer", "answer", "why"} <= cases[0].keys()


def test_edited_file_is_refused(tmp_path):
    edited = tmp_path / "judge_sanity.json"
    edited.write_bytes(SANITY_FILE.read_bytes().replace(b"INCORRECT", b"CORRECT", 1))
    with pytest.raises(SystemExit, match="changed since it was frozen"):
        load_cases(edited)


def test_hash_ignores_crlf_line_endings(tmp_path):
    crlf = tmp_path / "crlf.json"
    crlf.write_bytes(SANITY_FILE.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", b"\r\n"))
    assert file_sha256(crlf) == file_sha256(SANITY_FILE)


def test_summary_counts_lenient_strict_and_errors():
    rows = [
        {"expected": "CORRECT", "got": "CORRECT"},
        {"expected": "INCORRECT", "got": "CORRECT"},    # lenient
        {"expected": "PARTIAL", "got": "CORRECT"},      # lenient
        {"expected": "CORRECT", "got": "PARTIAL"},      # strict
        {"expected": "INCORRECT", "got": "ERROR"},      # unreadable: neither
    ]
    s = summarize(rows)
    assert (s["total"], s["agree"], s["too_lenient"], s["too_strict"], s["errors"]) == (5, 1, 2, 1, 1)
    assert s["confusion"][("INCORRECT", "CORRECT")] == 1
