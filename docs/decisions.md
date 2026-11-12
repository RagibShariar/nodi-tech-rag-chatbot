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

## 2026-11-12 · M2 · Ollama for embeddings

- One local runtime for embeddings now and the LLM in M4. Called over plain REST with `httpx`.
- Rejected: `sentence-transformers` (pulls in PyTorch, a second way of running models).
- Cost: CPU only on this machine (AMD Vega 11 iGPU not supported): ~3s per chunk with bge-m3.

## 2026-11-12 · M2 · Qdrant local mode instead of a server

- No Docker on this machine; 50 vectors don't need a server. Same client API; set `QDRANT_URL` to switch.
- Trade-offs: one process at a time (folder lock), no dashboard, exact search (no HNSW index).

## 2026-11-12 · M2 · bge-m3 as embedding model (by score)

- Compared bge-m3, nomic-embed-text, qwen3-embedding:0.6b on 36 eval questions (docs/experiments.md, E1).
- bge-m3: hit@5 100%, MRR 0.935, best on every metric, no misses. nomic missed the cross-language
  question; qwen3-embedding missed two exact-code questions.

## 2026-11-12 · M3 · rank_bm25 for keyword search

- In-memory BM25 over 50 chunks builds in milliseconds; no server.
- Rejected: Postgres full-text search (no Postgres here; `ts_rank` is not BM25).
- Tokenizer keeps hyphenated codes whole and split (`nd-1042`, `nd`, `1042`) and covers the Bangla
  Unicode block. No stemming yet ("mean" ≠ "Meaning" cost ND-1042 the #1 spot in BM25 alone).

## 2026-11-12 · M3 · sentence-transformers (PyTorch) for the reranker

- Reverses the M2 "no PyTorch" choice for one reason: Ollama 0.40.16 has no rerank endpoint (probed:
  `/api/rerank` and `/v1/rerank` return 404). Embeddings stay on Ollama.
- Windows needed the Microsoft VC++ runtime for PyTorch (installed by the user; admin required).
- Imported lazily, so vector/bm25/hybrid modes never load PyTorch.

## 2026-11-12 · M3 · Default retrieval: rerank, 5 candidates (by score)

- E2: rerank MRR 0.986 / all@5 36/36, vs vector 0.935 / 34 and hybrid 0.887 / 36.
- 5, 10 and 20 candidates scored identically; 5 is fastest (11.4s vs 32.2s per question on CPU).
- Search depth (20) and rerank candidates are separate settings (a coupled version gave a confounded
  measurement, see E2).
- Risk: ~11s retrieval latency on CPU. Fallback: `RETRIEVAL_MODE=vector`.

## 2026-11-12 · M3 · ND-1042 criterion changed (option A)

- Spec said "BM25 alone ranks error codes #1"; that was stricter than the brief ("find the error codes
  file"). Now: full pipeline #1 (met: 0.784 vs 0.649), BM25 alone top 2 (met: #2).
