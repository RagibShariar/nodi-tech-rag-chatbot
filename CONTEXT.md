# CONTEXT: resume this project on any machine

> Read this first when starting a new session (office PC or home laptop). The AI assistant's memory and chat
> history do **not** move between machines; everything needed to continue is in this repo.
> Update the "Current state" and "Next step" sections at the end of every session.

Last updated: 2026-11-13 (M6 built and run, waiting for the human grades; home laptop)

---

## 1. Current state

| Milestone | Status | Commit | Key result |
|---|---|---|---|
| M1 Ingestion | ✅ | `90e98e2` | 50 chunks from 12 files, PDF table intact |
| M2 Vector search | ✅ | `bd83038` | bge-m3 hit@5 100%, MRR 0.935 (chosen over nomic / qwen3-embedding by score) |
| M3 Hybrid + rerank | ✅ | `ebe4fcc` | BM25 + RRF + bge-reranker: MRR 0.986, all@5 36/36; default rerank, 5 candidates |
| M4 Chat | ✅ | `d960a4c` | qwen3.5:4b, streaming page, follow-ups: 8/9 key facts, 9/9 citations, ~1 min/answer on CPU |
| M5 Guardrails | ✅ | `f021e77` | leaks 2 → 0, 15/17 guardrail cases, eval Q35–Q40 5/6 |
| **M6 Evaluation** | ⏳ **in progress** | not committed yet | judge sanity 14/14; 40 q x 3 seeds run; hit@5 36/36; **waiting for human grades** |

Tests: 170 passing (`pytest`), 163 fast (`pytest -m "not integration"`).

Where the details are:
- [docs/project_notes.md](docs/project_notes.md): study guide per milestone (bilingual), interview Q&A
- [docs/experiments.md](docs/experiments.md): every measured number (E1–E4)
- [docs/decisions.md](docs/decisions.md): every choice and why
- [docs/specs/](docs/specs/): one spec per milestone
- [PROGRESS.md](PROGRESS.md): session log and open questions

---

## 2. Next step: M6 Evaluation, finish it

Spec approved (`docs/specs/M6.md`, decisions at the end: human grades all 40, 3 seeds, judge qwen3.5:4b).
Built: `src/eval/judge.py`, `scripts/judge_sanity.py` (+ frozen `scripts/judge_sanity.json`), `scripts/run_eval.py`.
Done: judge sanity 14/14 (E5a); full run 40 questions x seeds 42/1/2 (E5b, automatic metrics only).

**Waiting on Ragib:** grade the 40 seed-42 answers in **`docs/eval/human_labels.csv`** (`human` column:
C / P / I, optional `note`). Rubric = the judge's: C all key facts, nothing wrong; P nothing wrong but a key
fact/behaviour missing; I wrong, invented, different question or wrong behaviour. Ignore citations.
**Do not open `docs/eval/results/*.json` or the run output before grading:** they contain the judge's
verdicts and would anchor the grades (one verdict, Q12, was already seen; disclosed in E5b).
Judge-based numbers are deliberately not in any doc yet.

After grading:
1. `python -m scripts.run_eval --compare-human docs/eval/human_labels.csv docs/eval/results/eval_seed42.json`
   -> agreement %, CORRECT-vs-not agreement, Cohen's kappa, lenient/strict, every disagreement explained.
   Judge trusted if exact agreement >= 85%; otherwise the human grades are the official score (and maybe
   try qwen3.5:9b as judge: 6.1 GB download, ask first).
2. Show the judge numbers: correct rate per seed, Q35–Q40, consistency (unstable questions).
3. E5c in experiments.md; then end-of-milestone docs: project_notes M6 (bilingual) + project summary,
   decisions.md, README status, PROGRESS, this file. Then wait for "commit and push".

Note: the work of 2026-11-13 is **not committed** yet. To grade on another machine, commit and push first.

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
