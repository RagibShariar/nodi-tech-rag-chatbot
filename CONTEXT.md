# CONTEXT: resume this project on any machine

> Read this first when starting a new session (office PC or home laptop). The AI assistant's memory and chat
> history do **not** move between machines; everything needed to continue is in this repo.
> Update the "Current state" and "Next step" sections at the end of every session.

Last updated: 2026-11-13 (M6 done, human grading skipped; UI redesign + debug panel + feedback; demo on Sunday 2026-11-15; home laptop)

---

## 1. Current state

| Milestone | Status | Commit | Key result |
|---|---|---|---|
| M1 Ingestion | ✅ | `90e98e2` | 50 chunks from 12 files, PDF table intact |
| M2 Vector search | ✅ | `bd83038` | bge-m3 hit@5 100%, MRR 0.935 (chosen over nomic / qwen3-embedding by score) |
| M3 Hybrid + rerank | ✅ | `ebe4fcc` | BM25 + RRF + bge-reranker: MRR 0.986, all@5 36/36; default rerank, 5 candidates |
| M4 Chat | ✅ | `d960a4c` | qwen3.5:4b, streaming page, follow-ups: 8/9 key facts, 9/9 citations, ~1 min/answer on CPU |
| M5 Guardrails | ✅ | `f021e77` | leaks 2 → 0, 15/17 guardrail cases, eval Q35–Q40 5/6 |
| M6 Evaluation | ✅ | `44fbf8f` + this commit | retrieval 36/36; answers 90.8% CORRECT by LLM judge (0 wrong); Q35–Q40 6/6; judge sanity 14/14 |
| Chat UI | ✅ | `26d836e` | redesign to docs/ui-spec.md, SSE, debug panel, thumbs feedback |

Tests: 191 passing (`pytest`), 184 fast (`pytest -m "not integration"`).

Where the details are:
- [docs/project_notes.md](docs/project_notes.md): study guide per milestone (bilingual), interview Q&A
- [docs/demo_guide.md](docs/demo_guide.md): demo script, end-to-end flow, numbers, likely interview questions
- [docs/experiments.md](docs/experiments.md): every measured number (E1–E5)
- [docs/decisions.md](docs/decisions.md): every choice and why
- [docs/specs/](docs/specs/): one spec per milestone
- [PROGRESS.md](PROGRESS.md): session log and open questions

---

## 2. Next step: demo on Sunday 2026-11-15

All six milestones are done. The user demos the project on **Sunday 2026-11-15** and will then be questioned on how
it works inside (interview style).

- **Prep material:** [docs/demo_guide.md](docs/demo_guide.md) (setup checklist, 10-step demo script, one question
  end to end, numbers to remember, likely questions, limitations) and [docs/project_notes.md](docs/project_notes.md)
  (bilingual study guide with Q&A per milestone).
- **Next session:** offer a **mock interview** (ask one question at a time in Banglish, give feedback on each answer),
  then a full demo run-through following demo_guide section 1.
- Optional, if time: write the full M6 study section in project_notes (same structure as M1–M5).

M6 summary (details in experiments.md E5): retrieval 36/36 in 3 seeds; answer correctness by the LLM judge 90.8% mean
(90 / 92.5 / 90%), 0 INCORRECT in 120 answers, the rest PARTIAL (Q12, Q24, Q30, Q26 drop a condition); Q35–Q40 6/6;
version conflict 2/2; judge sanity set 14/14. **Human grading was skipped by the user's decision (time)**, so
judge-vs-human agreement is not measured; the judge contradicted itself on Q26 (reason vs verdict). The grading tools
still exist: `docs/eval/human_labels.csv` (empty sheet), `run_eval --compare-human`, and a grading page
(claude.ai artifact, private to the user) whose grades are in its `grades` store.

Known weak spot to mention when asked about adding documents: superseded status comes from the file name
(`src/ingest/parse.py`); with more documents it should become metadata (status, version, effective date).

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

### Home laptop (set up 2026-11-13)

i5-12500H, 15.7 GB RAM, **NVIDIA RTX 3050 Laptop GPU (4 GB VRAM)**. Ollama runs the models on the GPU; the
reranker runs on CPU (pip installs `torch ...+cpu`). Python 3.12.10 and Ollama 0.40.16 via winget
(`%LOCALAPPDATA%\Programs\...`; open a new terminal for PATH). VC++ runtime was already present.
Index build 31s; ~9s per answer + ~3s per judge call; full eval ~9 min per seed.
**Needs `.env` with `OLLAMA_URL=http://127.0.0.1:11434`** (see gotchas: localhost is slow here).

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

- **Running tests while the server is up:** the integration tests open Qdrant too and fail with "already
  accessed by another instance". Stop the server, or copy `artifacts/qdrant` (without `.lock`) to another
  folder and run with `QDRANT_PATH=<that folder>`; delete the copy afterwards.
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
- **Home laptop: `localhost` is slow.** It tries IPv6 `::1` first and hangs ~30s per request before falling
  back to IPv4 (index build took 10+ min instead of 31s). Fix: `OLLAMA_URL=http://127.0.0.1:11434` in `.env`.
- **Home laptop: Ollama can crash on the first GPU load** of qwen3.5:4b (CUDA "shared object initialization
  failed", HTTP 500); the next request loads fine. `run_eval` warms the model up and retries a crash once.
- Reranker integration tests skip until the model is downloaded (tests never download 2.3 GB). Run one
  `python -m src.retrieval.search "..."` first on a new machine.
