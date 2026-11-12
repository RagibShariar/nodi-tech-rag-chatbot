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

## 2026-11-12 · M4 · qwen3.5:4b as the chat model

- Ollama registry checked directly: qwen3.5 has 0.8b/2b/4b/9b dense models (4b = 3.1 GB).
- 4b chosen by the user (no 2b download). E3: 8/9 key facts, 9/9 citations, "18 not 15" handled.
- `think: false`: Qwen's hidden reasoning tokens would add many seconds on CPU.
- `temperature 0.1` + fixed seed: factual, repeatable answers (identical across two runs).

## 2026-11-12 · M4 · Prompt design

- Stable parts first (system prompt, history), changing parts last (documents + question), so Ollama's
  KV cache can reuse the prefix.
- Documents fenced as `<document source section status>`; only `content` is sent (the M1 context line
  repeats the tag and costs ~1s of prefill per chunk on this CPU).
- Query rewriting uses its own short prompt with one example, not the answering prompt: a 4B model does
  one job at a time better. Not tuned on the 6 test follow-ups (that would be overfitting to the test set).

## 2026-11-12 · M4 · FastAPI + NDJSON streaming + plain HTML page

- `POST /chat` streams one JSON event per line (same format idea as Ollama); a sync generator so a slow
  answer doesn't block other requests. Errors become an `error` event instead of a cut stream.
- Page: plain HTML/JS, no framework. All server text is HTML-escaped before formatting (XSS guard).
- Sessions in memory (lost on restart). The API holds the Qdrant local folder lock while running.

## 2026-11-12 · M3 · ND-1042 criterion changed (option A)

- Spec said "BM25 alone ranks error codes #1"; that was stricter than the brief ("find the error codes
  file"). Now: full pipeline #1 (met: 0.784 vs 0.649), BM25 alone top 2 (met: #2).
