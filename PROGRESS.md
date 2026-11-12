# Progress

## Session 1 · 2026-11-12 · M1 Ingestion

### Done
- Python 3.12.10 installed, `.venv` created, `pdfplumber` + `pytest` pinned in `requirements.txt`.
- `src/ingest/parse.py`: markdown files read as utf-8; PDF converted to markdown
  (headings from font size, table via pdfplumber's table extractor, wrapped cells joined).
- `src/ingest/chunk.py`: one chunk per `##`/`###` section with heading path, max 1500 chars,
  tables split by rows with header repeated, context line in `embed_text`.
- `src/ingest/run.py`: prints all chunks, writes `artifacts/chunks.jsonl`.
- Result: **50 chunks from 12 files**, sizes 43 to 668 chars (avg 204). 33 tests pass.
- Tests moved to `tests/ingest/` (mirrors `src/`). README rewritten as a project README.
- Added `docs/project_notes.md`: study guide, updated every milestone.
- Committed and pushed to GitHub (private repo).

### Concepts covered
- Tokens (model's unit of text; Bangla uses more tokens than English for the same meaning).
- Chunking: why we split, too big vs too small, splitting by document structure.
- Why tables are tricky: rows need their header; PDFs store positioned text, not tables.
- Context line: puts document + section name into every chunk.

### Open questions / things to watch
- PDF footer ("Questions: finance@...") ends up inside section "4. Not reimbursable".
- Bangla doc has Bangla section names; English questions about it rely on cross-lingual embeddings (M2).
- FAQ is one chunk covering 6 topics; may weaken retrieval. Check in M6.
- Tiny chunks (43 to 66 chars) like "Casual leave: 10 days. Sick leave: 14 days." Check in M6.
- No real token counts yet; measure with the bge-m3 tokenizer in M2.
