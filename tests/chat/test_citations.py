from src import config
from src.chat.citations import best_line, candidate_lines, cited_sources, is_answered, snippet
from src.ingest.chunk import chunk_all
from src.ingest.parse import load_all


def _src(source, status="current", section="S", content="text"):
    return {"source": source, "section": section, "status": status, "content": content}


def test_numbers_follow_first_citation_order_and_dedupe():
    retrieved = [_src("b.md"), _src("a.md"), _src("c.md")]
    cards = cited_sources("x [a.md] y [b.md] z [a.md]", retrieved)
    assert [(c["id"], c["file"]) for c in cards] == [(1, "a.md"), (2, "b.md")]


def test_citation_of_a_file_that_was_not_retrieved_gets_no_card():
    cards = cited_sources("[made_up.md] and [a.md]", [_src("a.md")])
    assert [c["file"] for c in cards] == ["a.md"]


def test_status_comes_from_metadata():
    assert cited_sources("[old.md]", [_src("old.md", status="superseded")])[0]["status"] == "superseded"


def test_answered_means_at_least_one_card():
    assert is_answered([{"id": 1}]) and not is_answered([])

# --- snippets ------------------------------------------------------------------------------------

TABLE = """Intro line about leave.

| Leave type | Days per year | Notes |
|---|---|---|
| Annual leave | 18 | Apply at least 3 working days in advance |
| Casual leave | 10 | Can be applied on the same day |"""


def test_table_rows_get_their_column_names():
    assert candidate_lines(TABLE) == [
        "Intro line about leave.",
        "Leave type: Annual leave · Days per year: 18 · Notes: Apply at least 3 working days in advance",
        "Leave type: Casual leave · Days per year: 10 · Notes: Can be applied on the same day",
    ]


def test_snippet_is_the_line_that_matches_the_answer():
    card = cited_sources("You get 18 days of annual leave [h.md].", [_src("h.md", content=TABLE)])[0]
    assert card["snippet"].startswith("Leave type: Annual leave · Days per year: 18")


def test_best_line_searches_every_retrieved_chunk_of_the_file():
    index, line = best_line("Casual leave is 10 days.", ["Unrelated text.", TABLE])
    assert index == 1 and "Casual leave" in line


def test_markdown_markers_are_stripped():
    assert candidate_lines("## Heading\n- **Bold** item") == ["Heading", "Bold item"]


def test_snippet_flattens_and_cuts_at_a_word():
    assert snippet("a\n  b") == "a b"
    s = snippet("word " * 100, limit=22)
    assert s == "word word word word…" and len(s) <= 23


def test_every_real_chunk_yields_readable_lines():
    # Real data: no table separator rows or empty lines leak into snippets.
    for chunk in chunk_all(load_all(config.DATA_DIR)):
        lines = candidate_lines(chunk.content)
        assert lines, chunk.chunk_id
        assert not any(line.startswith("|") or set(line) <= set("|-: ") for line in lines), chunk.chunk_id
