import pytest

from src import config
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all
from src.retrieval.bm25 import BM25Index, tokenize


@pytest.fixture(scope="module")
def index():
    return BM25Index(chunk_all(load_all(config.DATA_DIR)))


# ---------- Tokenizer ----------

def test_tokenize_keeps_codes_whole_and_split():
    assert tokenize("Error ND-1042!") == ["error", "nd-1042", "nd", "1042"]
    assert tokenize("X-Nodi-Key") == ["x-nodi-key", "x", "nodi", "key"]


def test_tokenize_splits_filenames_and_punctuation():
    assert tokenize("employee_handbook_v3.md") == ["employee", "handbook", "v3", "md"]


def test_tokenize_keeps_bangla_words_whole():
    # "canteen" in Bangla contains vowel signs and a hasanta; \w+ would cut it into pieces.
    assert tokenize("ক্যান্টিন") == ["ক্যান্টিন"]
    assert tokenize("সকাল ৮:১৫ মিনিটে ছাড়ে।") == ["সকাল", "৮", "১৫", "মিনিটে", "ছাড়ে"]


# ---------- Search on the real chunks ----------

@pytest.mark.parametrize("query", ["What does error ND-1042 mean?", "I got error ND-1042."])
def test_nd_1042_finds_error_codes_in_top_2(index, query):
    # The brief's M3 requirement (the 2nd query is conversation C3, turn 1).
    # Top 2, not #1: the short "Rate limits" chunk also mentions ND-1042 and BM25's length
    # normalization favours it (see docs/specs/M3.md). The full pipeline must put it #1.
    top_two = [chunk.source for chunk, _ in index.search(query, k=2)]
    assert "nodidesk_error_codes.md" in top_two


@pytest.mark.parametrize("query", [
    "SEC-EXC",
    "What ticket type do I raise for a USB storage exception?",  # eval Q17, exact wording
])
def test_sec_exc_finds_devices_section_first(index, query):
    top_chunk, _ = index.search(query, k=5)[0]
    assert top_chunk.chunk_id == "it_security_policy.md#2"  # "2. Devices"


def test_no_shared_words_returns_nothing(index):
    assert index.search("zzzz qqqq", k=5) == []


def test_results_sorted_and_limited(index):
    results = index.search("annual leave days", k=3)
    assert len(results) == 3
    scores = [score for _, score in results]
    assert scores == sorted(scores, reverse=True)
