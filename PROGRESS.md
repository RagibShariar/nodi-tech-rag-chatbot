# Progress

## Session 2 · 2026-11-13 · M6 Evaluation + chat UI (home laptop)

### Done
- Set up the home laptop (Python, Ollama, models, venv, index); all 123 existing tests pass there.
  Found and fixed (in `.env`) a 30s-per-request `localhost` IPv6 delay.
- M6 spec approved; decisions: human grades all 40, 3 seeds, judge qwen3.5:4b.
- `src/eval/judge.py`: rubric (no question-specific hints), fenced answer, strict JSON parser (ERROR, never
  guessed). `OllamaChat` takes optional temperature/seed.
- Judge sanity set (14 cases) frozen before use; judge 14/14 (E5a).
- `scripts/run_eval.py`: 40 questions x seeds, automatic checks, judge, consistency, `--rejudge`,
  review sheet without verdicts, `--compare-human` (agreement, kappa). Crash handling after a smoke run
  showed an Ollama crash graded as a bot failure.
- Full run 3 seeds: retrieval 36/36, citations 36/36, version conflict 2/2 in every seed (E5b).
- 191 tests (184 fast).

- UI redesign to `docs/ui-spec.md`: SSE stream, numbered citation chips + sources panel (status from
  metadata, best-matching snippet), "I don't know" box (= no citation), light/dark, phone layout, recent
  chats in localStorage, `/info` endpoint. Checked with headless Edge screenshots (light, dark, phone, tablet).
- Retrieval debug panel (BM25 / vector lists via a caller-owned `trace` dict, rerank score bars, Cited,
  timing), M4-style info under each answer, working thumbs logged to `artifacts/feedback.jsonl`.

### Concepts covered
- Evals vs unit tests; separate retrieval and answer metrics; why substring checks fail.
- LLM-as-judge, rubric, reason-before-verdict; limits: self-grading, leniency, verbosity, injection.
- Testing the judge: frozen sanity set; human agreement; Cohen's kappa (agreement beyond chance).
- Anchoring: grade before seeing the judge. Measurement bugs (an infra crash counted as a wrong answer).
- UI: Server-Sent Events; why chat history stays on the server (forged turns = injection); citations as
  evidence links; a caller-owned trace dict for debugging without shared state.

### Open questions / things to watch
- Human grading **skipped** (user's decision, time before the 2026-11-15 demo): judge-vs-human agreement not
  measured; self-grading bias untested. The judge contradicted itself on Q26 (reason vs verdict).
- Final M6 (E5c): answers 90.8% CORRECT by the judge (0 INCORRECT), retrieval 36/36, Q35–Q40 6/6.
- Demo prep: `docs/demo_guide.md`; next session offers a mock interview.
- Timings on the laptop vary with heat and other load (6.3s to 11.8s average per answer).
- "Cited" in the debug panel is per file: the model cites file names, not chunks.
- Not built yet: the scope (metadata) filter button; feedback is logged but not analysed.

## Session 1 (cont.) · 2026-11-12 · M5 Guardrails

### Done
- Frozen guardrail test set (17 cases, hash recorded) + `scripts/run_guardrails.py`.
- Baseline found **2 full system-prompt leaks** (X1 "repeat everything above", X4 Bangla request).
- `src/chat/guardrails.py`: fence neutralizing, LeakGuard (6-word n-grams), refusal texts. Prompt: scope +
  secrecy rules, "system notes are data", sandwich reminder. Engine: streaming leak guard + `replace` event.
- Final (E4d): 15/17 automatic, **0 leaks**, injection ignored, eval Q35–Q40 **5/6**. Conversations unchanged (8/9, 9/9).
- Relevance gate rejected by data (scores overlap). Contacts directory tried and reverted (hallucinated email, +12s).
- 123 tests pass.

### Concepts covered
- Hallucination; direct vs indirect prompt injection; fencing untrusted text; defence in depth.
- Output filtering during streaming; over-refusal and control cases; freezing a test set.
- Seed variance: one deterministic run is not proof of reliability (K3: 2/5 seeds).

### Open questions / things to watch
- Q35 doesn't suggest HR (the HR contact chunk isn't retrieved; adding contacts made things worse).
- K3 (Bangla canteen, two prices) answers fully only 2/5 seeds: M6 should evaluate sensitive questions over several seeds.
- Leak guard catches copied wording only; paraphrased leaks rely on the prompt rule.
- Latency still ~1 min per answer.

## Session 1 (cont.) · 2026-11-12 · M4 Chat

### Done
- Pulled qwen3.5:4b (registry checked: 0.8b/2b/4b/9b exist). Installed fastapi + uvicorn.
- `src/chat/`: llm.py (Ollama /api/chat NDJSON streaming + speed stats), prompts.py (system rules, fenced
  documents), rewrite.py (follow-up → standalone question), history.py (last 6 turns), engine.py (events).
- `src/api/`: FastAPI (`/`, `/chat` NDJSON stream, `/reset`) + plain HTML page (escaped output, live timer,
  citation chips, sources list). Tested in a browser; screenshot in docs/images/m4_chat.png.
- `scripts/run_conversations.py`: E3 = 8/9 key facts, 9/9 citations, identical answers across runs.
- Measured LLM speed: prefill ~25 tok/s, generation ~7 tok/s → ~1 minute per answer on CPU.
- 104 tests pass (incl. end-to-end integration).

### Concepts covered
- Prompt, system prompt, message roles, context window, temperature, seed.
- Query rewriting; why not to tune prompts on the test set (overfitting).
- Streaming (NDJSON, generators, fetch stream reader); prefill vs generation; KV cache and prompt order.
- XSS: escape model/document text before rendering.

### Open questions / things to watch
- **Latency ~1 min per answer.** Levers to measure (one at a time, eval decides): rewrite prompt sharing the
  answer prompt's prefix (KV cache reuse); `RETRIEVAL_MODE=vector` (−8s); qwen3.5:2b; fewer chunks.
- C1 t3 answer misses "14 days per year" (correct but incomplete).
- Checker is substring-based; M6 needs a proper answer-correctness judge.
- Guardrails (injection, system-prompt leaks, out-of-scope) are only *in the prompt*, untested → M5.
- Sessions are in memory; lost on restart.

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
