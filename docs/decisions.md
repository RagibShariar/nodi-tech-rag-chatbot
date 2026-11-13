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

## 2026-11-12 · M5 · Guardrails: defence in depth, measured

- Test set frozen before the first run (SHA-256 in experiments.md E4) so it couldn't be tuned to pass.
- Kept: fence neutralizing (a document can't close its `<document>` tag); prompt rules for scope, secrecy
  and "system notes are data"; a sandwich reminder before the question; a streaming leak guard (6-word
  n-gram match against the rules; stops the stream, sends a `replace` event). Fixed baseline leaks 2 → 0.
- The leak guard protects the rules only, not the public intro line or the two sentences the model is told
  to say, otherwise correct refusals would be flagged.
- Rejected: input keyword blocklist (bypassed by rewording, blocks legitimate questions); separate
  classifier model (heavy, not required).
- Rejected by data: relevance gate on rerank score (in-scope and out-of-scope scores overlap: Q20 at 0.019
  vs Q40 at 0.185).
- Tried and reverted: contacts directory in every prompt. Didn't fix Q35, caused an invented email in Q37,
  +12s latency.
- Accepted limitations: Q35 doesn't suggest HR (no hard-coding to pass one test); K3 is flaky (2/5 seeds).

## 2026-11-12 · M3 · ND-1042 criterion changed (option A)

- Spec said "BM25 alone ranks error codes #1"; that was stricter than the brief ("find the error codes
  file"). Now: full pipeline #1 (met: 0.784 vs 0.649), BM25 alone top 2 (met: #2).

## 2026-11-13 · M6 · LLM-as-judge with qwen3.5:4b, validated two ways

- Judge = qwen3.5:4b at temperature 0, rubric CORRECT / PARTIAL / INCORRECT, JSON reply with the reason first.
  Only CORRECT counts toward the 85% target.
- Rubric kept free of question-specific hints (no "18", "March", "HR"): hints would tune the judge to its
  own test set. Enforced by a unit test.
- Validated by a frozen sanity set (14 known verdicts, 14/14) and by agreement with human grades (pending).
  Unreadable judge replies are ERROR, never parsed out of prose ("INCORRECT" contains "CORRECT").
- Human grading sheet hides the judge's verdict (avoids anchoring).
- Rejected: substring checks (wrong in M4/M5); qwen3.5:9b judge up front (6.1 GB download; only if agreement
  with the human grades is poor).

## 2026-11-13 · UI · Redesign to docs/ui-spec.md (deviations from the spec's API section)

- Built per `docs/ui-spec.md` and the main screenshots: three columns, design tokens as CSS variables,
  light/dark switch (OS default, choice saved in `localStorage`), numbered citation chips, sources panel
  with status pills, dashed "I don't know" box, phone layout (menu + horizontally scrolling source chips).
- **Stream format: SSE as the spec asks** (`event: delta` / `replace` / `done` / `error`), replacing NDJSON.
- **Deviation: history stays on the server** (`session_id`), the page does not send `history`. If the
  client sent history, anyone could forge earlier "assistant" turns and inject instructions that way.
  The request takes `session_id`, `message`, `scope` (only "all" accepted until the filter exists).
- **`answered: false` = the answer cites no retrieved document.** The system prompt requires a citation
  per fact; refusals, "I don't know" and the out-of-scope reply cite nothing, in any language. Rejected:
  matching refusal phrases (substring checks failed in M4/M5, and wouldn't work for Bangla).
- Source status comes from chunk metadata (`status`), not the file name. The snippet is the retrieved
  line with the most words in common with the answer (table rows shown with their column names).
- Recent chats are kept in the browser (`localStorage`); the server's history is in memory, so after a
  server restart an old chat can be re-read but the bot no longer remembers it.
- Fonts from Google Fonts with system fallbacks (no download or dependency). Self-hosting is the
  alternative if the page must work offline or make no outside requests.
- Scope button rendered but disabled with a tooltip (metadata filter: stretch goal).
- **Added on request (same day):** under every answer the M4 page's information is back (first word /
  total time, the rewritten question, a collapsible "Sources searched" list of all 5 retrieved chunks).
  The "Hybrid search + rerank" pill toggles a **retrieval debug panel** (docs/design/debug.png): standalone
  question, BM25 and vector top 5 each, the fused + reranked top 5 with score bars and a Cited column,
  timing and token counts. The two search lists come from `Retriever.retrieve(..., trace={})`: the caller
  passes its own dict, so concurrent requests can't mix traces, and results are unchanged.
  "Cited" is per file (the model cites file names, not chunks); a cited superseded file shows "Flagged".
- **Thumbs up/down work:** `POST /feedback` appends one JSON line to `artifacts/feedback.jsonl` (gitignored;
  question, answer, rating, cited files, time). Pressing the same thumb again sends "none". The logged text
  comes from the page, so it is untrusted data for later review, not a source of truth.
- No effect on answers: prompts, retrieval and the model are unchanged (the engine's `sources` event gained
  a `content` field), so no eval re-run is needed for this change.
