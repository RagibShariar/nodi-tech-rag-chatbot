# Decisions

## 2026-11-12 · M1 · Python 3.12

Python was not installed. Installed 3.12.10 (user scope, via winget). It meets the 3.11+ requirement
and is widely supported by ML libraries we need later (Qdrant client, sentence-transformers).

## 2026-11-12 · M1 · pdfplumber for PDF parsing

- Need: the expense policy PDF has a table whose cells wrap over several lines.
- `pdfplumber` extracts tables as rows/cells, and gives font sizes, which we use to detect headings.
- Rejected: `pypdf` (plain text only, table comes out jumbled), `PyMuPDF` (good, but AGPL license).

## 2026-11-12 · M1 · Chunking by headings, deterministic context line

- One chunk per `##`/`###` section, max 1500 chars; tables never split from their header.
- Context line `[Source | title | Section | Status]` is built from metadata, not written by an LLM
  (the LLM-written version is a stretch goal in the brief).
- FAQ kept as one chunk and tiny sections not merged. Revisit with M6 eval numbers.
