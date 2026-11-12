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
