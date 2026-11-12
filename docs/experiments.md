# Experiments

Every change to chunking, embeddings, retrieval or prompts gets a row here, with numbers from an actual
run. The score decides, not impressions.

## How retrieval is measured (until M6)

`python -m scripts.retrieval_check`, on the 36 eval questions that have an expected source document
(Q35, Q36, Q37 and Q40 have source `none` and are tested in M5).

| Metric | Meaning |
|---|---|
| **hit@5** | At least one expected source document among the top-5 chunks. Target ≥ 90%. |
| **all@5** | *All* expected source documents among the top 5 (matters for multi-document questions). |
| **MRR** | Mean Reciprocal Rank: 1/rank of the first correct chunk (1st = 1.0, 2nd = 0.5, ..., miss = 0), averaged. Rewards putting the right document *first*. |

Caveats: 36 questions, so one question = 2.8 percentage points. hit@5 is **document-level**: a chunk
from the right file but the wrong section still counts. 5 of 50 chunks is 10% of the whole corpus.

---

## E1 · 2026-11-12 · M2 baseline: vector search only, three embedding models

Setup: M1 chunks (50, heading-based, context line), Qdrant local mode, cosine, top-5, no BM25/rerank.

| Model | hit@5 | all@5 | MRR | Run time (CPU, index + 36 queries) |
|---|---|---|---|---|
| **bge-m3** | **100.0%** (36/36) | **34/36** | **0.935** | 89s (queries only; index built earlier in 161s) |
| nomic-embed-text | 97.2% (35/36) | 33/36 | 0.898 | 221s |
| qwen3-embedding:0.6b | 94.4% (34/36) | 31/36 | 0.880 | 263s |

**Misses / partials:**
- **nomic-embed-text** missed **Q34** (cross_language: English question about the Bangla bus document).
  Expected: it's a mostly-English model.
- **qwen3-embedding:0.6b** missed **Q17** (`SEC-EXC`, USB exception) and **Q19** (`ND-4000`): the
  exact-code trap. Also partial on Q21.
- **All three** partial on **Q27** (pricing table not in top 5 for "Can a Starter plan customer use the
  API?"; the API reference alone answers it) and **Q38** (vendor notice not retrieved for "Is Enterprise
  free?", which is harmless: that file is the injection trap, not a source of the answer).

**Decision:** keep **bge-m3** as the default. It is best on all three metrics, and is the only model
with no misses. The gaps are small (1–2 questions), but they're on exactly the traps the brief warns
about (Bangla, exact codes).

**Implication for M3:** hit@5 is already at 100% with bge-m3, so it **can't show improvement**
(ceiling effect). M3 must be judged by **MRR** and **all@5**, and should not make hit@5 worse.

---

## E2 · 2026-11-12 · M3: BM25, hybrid (RRF), reranker

Setup: bge-m3 embeddings (E1), BM25 (`rank_bm25`, tokenizer keeps hyphenated codes + parts, Bangla-aware),
RRF k=60, reranker `BAAI/bge-reranker-v2-m3` on CPU. Search depth 20 per method. Top-5. Same 36 questions.
Command: `python -m scripts.retrieval_check --modes vector bm25 hybrid rerank` (+ `CANDIDATES=` env).

| Mode | hit@5 | all@5 | MRR | Time per question (CPU) |
|---|---|---|---|---|
| vector (= E1, regression check) | 100% | 34/36 | 0.935 | 2.5s |
| bm25 | 97.2% | 33/36 | 0.848 | ~0s |
| hybrid (vector + BM25, RRF) | 100% | **36/36** | 0.887 | 2.5s |
| rerank, 20 candidates | 100% | 36/36 | **0.986** | 32.2s |
| rerank, 10 candidates | 100% | 36/36 | 0.986 | 18.3s |
| **rerank, 5 candidates** | **100%** | **36/36** | **0.986** | **11.4s** |

**Findings:**
- **BM25 alone** missed **Q34** (English question, Bangla document: no shared words), as predicted.
  Partial on Q21 and Q26.
- **Hybrid** fixed all multi-document gaps (all@5 34 → 36) but **lowered MRR** (0.935 → 0.887): when vector
  and BM25 disagree about #1, RRF's averaging pushes vector's correct #1 down. Example: for "What does error
  ND-1042 mean?", the error-codes chunk (vector #1, BM25 #2) and the rate-limits chunk (vector #2, BM25 #1)
  get the **exact same** RRF score (1/61 + 1/62), and the alphabetical tie-break puts rate-limits first.
- **Rerank** keeps hybrid's coverage and fixes the order: MRR 0.986 (35 of 36 questions have a correct
  document at #1). For ND-1042 it scores error codes 0.784 vs rate limits 0.649.
- **Candidate count (5/10/20) didn't change any score** here, only latency.

**Measurement bug found and fixed:** the first 10- and 5-candidate runs also shrank the search depth
(code used `max(CANDIDATES, k)` for both), giving rerank@5 all@5 = 34/36 and MRR 0.972. That run mixed two
changes. Fixed by separating `SEARCH_DEPTH` (20) from `CANDIDATES`, added a regression test (verified to
fail on the old code), and re-measured. The table above has the corrected numbers.

**Latency breakdown** (one question): embed query 2.5s · Qdrant ~0s · BM25 ~0s · rerank 5 passages
6.7–10.3s · reranker load 21s once per process. The cost is compute (568M-parameter model, ~1,250 tokens
for 5 passages on a 4-core CPU), not a bug.

**Decision:** default `RETRIEVAL_MODE=rerank`, `SEARCH_DEPTH=20`, `CANDIDATES=5`. Best scores (by the rule
approved in the M3 spec), and 5 candidates ties 10/20 on score, so latency decides.
**Open concern for M4:** ~11s of retrieval per question *before* the LLM starts writing. If chat is too
slow, the measured fallback is `RETRIEVAL_MODE=vector` (2.5s, MRR 0.935, all@5 34/36).

---

## E3 · 2026-11-12 · M4: chat with qwen3.5:4b on the 3 multi-turn conversations

Setup: M3 retrieval (rerank, 5 candidates), `qwen3.5:4b` via Ollama, CPU only, `temperature 0.1`,
`seed 42`, `think: false`, `num_ctx 8192`, history 6 turns, query rewriting with its own prompt.
Command: `python -u -m scripts.run_conversations` (answers saved to `artifacts/conversation_results.json`).

### LLM speed on this CPU (Ryzen 5 3400G, 4 cores)

Measured with a fresh (uncached) prompt each time:

| Prompt size | Prefill (reading the prompt) | Generation (writing) |
|---|---|---|
| 451 tokens | 17.7s = 25 tok/s | 6.7 tok/s |
| 706 tokens | 29.4s = 24 tok/s | 7.8 tok/s |
| 1,403 tokens | 54.5s = 26 tok/s | 6.2 tok/s |

Prefill is linear at ~25 tok/s: **every 100 prompt tokens cost ~4 seconds.** Writing a 30–100-token answer
takes 4–15s. So prompt size, not answer length, drives latency. Ollama's KV cache makes an identical prompt
prefix nearly free (a repeated 429-token prompt prefilled in 0.3s).

### Results (9 turns)

| Check | Result |
|---|---|
| Key facts (crude substring check) | **8/9** |
| Citations of a retrieved file | **9/9** |
| "18 days, never 15" (C1 t1) | ✅ 18, and calls 15 "outdated and superseded" |
| Follow-ups rewritten correctly | 6/6 usable (C3 t3 added "on Nodide", a harmless typo) |
| Same answers as the first run | yes, word for word (fixed seed, low temperature) |

The one failure: **C1 t3** "And what about sick leave?" → rewritten "Can I carry forward sick leave?" →
"No, you cannot carry forward sick leave". Correct, but misses the expected "14 days per year".

### Latency (CPU)

| | first token | total |
|---|---|---|
| Reported by the script (all 9 turns) | avg 43.8s | avg 50.8s, max 80.4s |
| **Excluding C1 t1–t2** (cache hits, see below) | **avg 52.6s** | **avg 59.8s**, max 80.4s |

Breakdown per turn: rewrite 9–13s (follow-ups only) · retrieval 10–14s · prefill of 600–1,100 tokens 25–45s
· generation 3–15s. Later turns are slower because the history grows the prompt.

**Caveat:** C1 t1 and t2 (first token 12s and 13.5s) were the exact questions I had just asked in the browser
test, so Ollama still had those prompts in its KV cache. They are not representative; the second row is.

### Checker change (made after the first run, disclosed)

The first run's check marked C1 t1 as failed because the answer contained "15 days". The answer was correct:
it said 15 was outdated, which is what system-prompt rule 4 asks for and what eval Q29's expected answer
does. The check now fails "15" only when it is *not* called outdated/superseded/old/previous (whole words,
citations removed: the filename `..._SUPERSEDED.md` must not count). Both versions are unit-tested.
First run: 7/9 under the old check, 8/9 under the new one, same answers.

**Decision:** keep `qwen3.5:4b` (8/9 facts, 9/9 citations, follows the superseded rule). Latency of
~1 minute is accepted for now (correctness first, as agreed); speed levers are listed in PROGRESS.md.
