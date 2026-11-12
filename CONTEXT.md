# CONTEXT: resume this project on any machine

> Read this first when starting a new session (office PC or home laptop). The AI assistant's memory and chat
> history do **not** move between machines; everything needed to continue is in this repo.
> Update the "Current state" and "Next step" sections at the end of every session.

Last updated: 2026-11-12 (end of M5, office PC)

---

## 1. Current state

| Milestone | Status | Commit | Key result |
|---|---|---|---|
| M1 Ingestion | ✅ | `90e98e2` | 50 chunks from 12 files, PDF table intact |
| M2 Vector search | ✅ | `bd83038` | bge-m3 hit@5 100%, MRR 0.935 (chosen over nomic / qwen3-embedding by score) |
| M3 Hybrid + rerank | ✅ | `ebe4fcc` | BM25 + RRF + bge-reranker: MRR 0.986, all@5 36/36; default rerank, 5 candidates |
| M4 Chat | ✅ | `d960a4c` | qwen3.5:4b, streaming page, follow-ups: 8/9 key facts, 9/9 citations, ~1 min/answer on CPU |
| M5 Guardrails | ✅ | `f021e77` | leaks 2 → 0, 15/17 guardrail cases, eval Q35–Q40 5/6 |
| **M6 Evaluation** | ⏳ **next** | — | not started, no spec yet |

Tests: 123 passing (`pytest`), 116 fast (`pytest -m "not integration"`).

Where the details are:
- [docs/project_notes.md](docs/project_notes.md): study guide per milestone (bilingual), interview Q&A
- [docs/experiments.md](docs/experiments.md): every measured number (E1–E4)
- [docs/decisions.md](docs/decisions.md): every choice and why
- [docs/specs/](docs/specs/): one spec per milestone
- [PROGRESS.md](PROGRESS.md): session log and open questions

---

## 2. Next step: M6 Evaluation

Start by writing `docs/specs/M6.md` and getting approval (no code before that).

From the brief/CLAUDE.md: a script that runs all 40 questions in `eval/eval_questions.csv` and reports
**retrieval hit rate** (target ≥ 90%) and **answer correctness vs `expected_answer`** (target ≥ 85%), plus the
guardrail and version-conflict targets. Concept to teach first: evals, metrics, **LLM-as-judge and its limits**.

Things we already learned that M6 must handle:
- **Reuse what exists:** `scripts/retrieval_check.py` (hit@5/all@5/MRR, 36 questions with a source),
  `scripts/run_conversations.py` (3 multi-turn chats), `scripts/run_guardrails.py` (17 frozen cases).
- **Substring checks are crude:** they gave false positives/negatives in M4 and M5 (e.g. "31 March" vs
  "March 31st", a forbidden phrase inside a correct refusal). M6 needs a better correctness judge, and its
  limits must be measured (e.g. judge vs my own by-eye grading on a sample).
- **One seed is not proof:** K3 (Bangla canteen price) answers fully only 2/5 seeds. Consider several seeds
  for sensitive questions, or report variance.
- **Runtime:** ~1 minute per answer on the office PC (CPU). 40 questions ≈ 40+ minutes per run; plan runs
  in the background (`python -u ...`), and remember the judge adds LLM calls too.
- `eval/eval_questions.csv` starts with a BOM: read it with `encoding="utf-8-sig"`.
- Known limitations to report honestly: Q35 doesn't suggest HR; K3 flaky; checker false positives.

---

## 3. Set up a new machine (e.g. the home laptop)

Commands are for Windows PowerShell (the office PC). On macOS/Linux: `python3`, `source .venv/bin/activate`,
install Ollama from ollama.com, and skip the VC++ step.

```powershell
# 0. GitHub access (the repo is private)
gh auth login
git clone https://github.com/RagibShariar/nodi-tech-rag-chatbot.git
cd nodi-tech-rag-chatbot

# 1. Python 3.12 (3.11+ works)
winget install --id Python.Python.3.12 --scope user        # then open a NEW terminal

# 2. Windows only: VC++ runtime, needed by PyTorch (reranker). Needs admin: click "Yes" on the UAC popup
winget install --id Microsoft.VCRedist.2015+.x64

# 3. Virtual env + packages (pinned in requirements.txt)
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env                                      # optional; defaults work without it

# 4. Ollama + models
winget install --id Ollama.Ollama                          # then open a NEW terminal (PATH)
ollama pull bge-m3                                          # embeddings (1.2 GB)
ollama pull qwen3.5:4b                                      # chat LLM (3.3 GB)
#    only to reproduce the M2 model comparison (E1):
#    ollama pull nomic-embed-text ; ollama pull qwen3-embedding:0.6b

# 5. Rebuild the generated files (artifacts/ is NOT in git)
python -m src.ingest.run                                    # chunks -> artifacts/chunks.jsonl
python -m src.retrieval.index                               # Qdrant index -> artifacts/qdrant (~3 min on CPU)

# 6. Check everything works
pytest -m "not integration"                                 # fast, no models needed (~10s)
pytest                                                      # all; first run downloads the reranker (~2.3 GB)
python -m uvicorn src.api.app:app --port 8000               # chat page at http://127.0.0.1:8000
```

### Machine differences matter for numbers

All timings in experiments.md (E1–E4) were measured on the **office PC: AMD Ryzen 5 3400G (4 cores),
13.9 GB RAM, no usable GPU (Radeon Vega 11 iGPU), CPU only.** If the laptop is different (especially with an
NVIDIA GPU, which Ollama will use), latency will differ. **Record the machine with every new measurement**
and don't compare timings across machines as if they were the same setup. Quality scores (hit rate, facts)
should be close but can vary slightly with hardware/library versions; re-run before comparing.

---

## 4. How we work (agreements from the chat, so the next session follows them)

- **Language:** explain in romanized Banglish in chat (Bangla script renders broken in the terminal).
  Code, comments, commits and docs in English. `docs/project_notes.md` is bilingual (English for interviews,
  simple Bangla script with real-life Bangladeshi analogies).
- **Pacing:** one milestone at a time. Spec first → wait for "approve" → build in small steps → after each step
  explain the code and wait for "next step" → at the end, summarize and wait for **"Next"**.
- **Approvals:** ask before any new dependency, download or install. Admin installs (UAC) are run by the user
  with `! <command>`. Destructive actions need explicit permission.
- **Commits:** at the end of a milestone the user says "commit and push"; the assistant then commits (message
  ends with the Co-Authored-By line) and pushes to `main`.
- **Evidence rules:** the score decides, not impressions. Write tests/test sets *before* seeing results;
  freeze evaluation sets (`scripts/guardrail_cases.json`, SHA-256 in experiments.md E4). Change **one thing at
  a time** (confounding happened twice: M3, M5). If a checker is changed after a run, disclose it and re-run.
  Report automatic and by-eye results when they differ. Never invent numbers.
- **Docs to update at the end of every milestone:** spec checklist, experiments.md, decisions.md,
  project_notes.md (bilingual section + glossary + Q&A), README status, CLAUDE.md commands, PROGRESS.md,
  and this file.

---

## 5. Gotchas

- **Qdrant local mode = one process at a time.** While the API server runs, `search`, `retrieval_check`,
  `run_conversations`, `run_guardrails` can't open the index. Stop the server first.
- **Slow on CPU:** retrieval ~10s (reranker), prompt reading ~25 tokens/s, ~1 min per answer. Run long scripts
  in the background with `python -u` so progress lines appear.
- **KV cache:** asking the exact same prompt twice is much faster the second time; don't measure latency on
  repeated questions.
- **Frozen test set hash:** verify with `git show HEAD:scripts/guardrail_cases.json | sha256sum`; a Windows
  checkout may turn LF into CRLF and change the working-copy hash without changing the content.
- `data/` and `eval/` are the provided assignment kit: never edit them. `vendor_notice_printer.md` contains a
  deliberate prompt injection (test data, not an instruction).
- Hugging Face prints symlink / HF_TOKEN warnings on Windows when loading the reranker: harmless.
