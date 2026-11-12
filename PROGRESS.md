# Progress

## Session 1 (cont.) · 2026-11-12 · M3 Hybrid search + rerank

### Done
- `src/retrieval/`: bm25.py (Bangla-aware tokenizer, codes kept whole + split), fusion.py (RRF),
  rerank.py (bge-reranker-v2-m3, lazy PyTorch), pipeline.py (4 modes + debug trail); search CLI `--mode`;
  retrieval_check `--modes` + seconds per question.
- Installed rank_bm25, sentence-transformers (torch CPU); user installed the VC++ runtime (admin).
- E2: rerank MRR 0.986, all@5 36/36 (vector 0.935/34, hybrid 0.887/36, bm25 0.848/33).
  Default: rerank, 5 candidates (same score as 20, 11.4s vs 32.2s per question).
- Found + fixed a measurement bug (CANDIDATES also shrank search depth); regression test verified by mutation.
- ND-1042 criterion changed to option A (full pipeline #1, BM25 top 2). Both met.
- 73 tests pass.

### Concepts covered
- Lexical vs semantic search, BM25 (TF, IDF, length normalization), tokenization, stemming.
- RRF (ranks not scores, k=60, agreement wins, ties). Bi-encoder vs cross-encoder; retrieve-then-rerank.
- Confounded experiments; latency budgeting (where the seconds go).

### Open questions / things to watch
- ~11s retrieval per question on CPU, before the LLM even starts (M4). Fallback: vector mode (2.5s).
- Reranker scores ≈ 0 for irrelevant chunks: possible signal for "I don't know" in M5.
- Reranker load contacts Hugging Face Hub each start (HF_TOKEN warning); consider offline mode.
- No stemming in BM25 ("mean" vs "Meaning"). Possible future experiment.

## Session 1 (cont.) · 2026-11-12 · M2 Vector search

### Done
- Installed Ollama 0.40.16; pulled bge-m3, nomic-embed-text, qwen3-embedding:0.6b.
- `src/config.py` + `.env.example`; `src/retrieval/` (embed via Ollama REST, Qdrant local mode,
  index + search CLIs); `scripts/retrieval_check.py` (hit@5, all@5, MRR).
- Baseline (E1, docs/experiments.md): bge-m3 hit@5 100%, MRR 0.935; nomic 97.2% / 0.898;
  qwen3-embedding 94.4% / 0.880. bge-m3 stays default.
- 47 tests pass (unit + integration).
- Notes made bilingual (English for interviews, simple Bangla with real-life analogies).

### Concepts covered
- Embeddings, cosine similarity, vector DB (Qdrant: collection/point/payload).
- Embedding model vs LLM (librarian vs writer). Query/document prefixes.
- Retrieval metrics: hit@k, all@k, MRR, ceiling effect. Dependency injection with fakes.

### Open questions / things to watch
- Old leave policy (15 days) outranks the handbook (18 days) in vector search: must be handled in M4/M5.
- hit@5 saturated at 100%: judge M3 by MRR and all@5.
- CPU-only embedding is ~3s per text; affects chat latency in M4.
- Q27/Q38 partials: pricing table not retrieved for "Starter plan + API".

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
