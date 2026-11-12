# Nodi Tech RAG Chatbot

A local RAG chatbot that answers questions about **Nodi Tech Ltd.** (a fictional company) using only its
internal documents. Every answer cites its source; if the documents don't cover a question, it says
"I don't know".

Built as a learning project, one milestone at a time, with a spec, tests and an evaluation score for
each step. Full assignment: [rag_chatbot_project_brief.md](rag_chatbot_project_brief.md).

## Status

| # | Milestone | Status |
|---|---|---|
| M1 | Ingestion: parse markdown + PDF (with table), chunk by headings | ✅ Done |
| M2 | Vector search: embeddings in Qdrant | ⏳ |
| M3 | Hybrid search: BM25 + RRF fusion + reranker | ⏳ |
| M4 | Chat: streaming answers with citations, follow-up questions | ⏳ |
| M5 | Guardrails: "I don't know", prompt-injection defence | ⏳ |
| M6 | Evaluation: 40-question eval set | ⏳ |

## Architecture

```
INDEXING
data/ -> parse (md + PDF) -> chunk -> add context line -> embed -> vector DB (Qdrant)
                                                    \-> BM25 index
CHAT
user message -> rewrite follow-up into standalone question
  -> lexical search + vector search -> fuse (RRF) -> rerank -> top 5
  -> prompt: system rules + chat history + fenced chunks
  -> LLM answer with [source] citations -> stream to UI
```

## Stack

Python 3.12 · pdfplumber · pytest. Planned: FastAPI · Qwen via Ollama · bge-m3 embeddings · Qdrant ·
rank_bm25 · bge-reranker-v2-m3.

## Quick start (Windows / PowerShell)

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt

python -m src.ingest.run   # parse + chunk all documents, print chunks, write artifacts/chunks.jsonl
pytest                     # run all tests
```

## Project layout

```
.
├── data/                  # 12 source documents (read-only; fictional; contains deliberate traps)
├── eval/                  # 40 eval questions + 3 multi-turn conversation tests
├── src/
│   └── ingest/            # parse.py (md/PDF -> Document), chunk.py (Document -> Chunks), run.py (CLI)
├── tests/
│   └── ingest/            # tests mirror src/
├── docs/
│   ├── project_notes.md   # study guide: what/why/how per milestone, interview Q&A
│   ├── specs/             # one spec per milestone (written before code)
│   └── decisions.md       # why each tool/approach was chosen
├── PROGRESS.md            # session log: done, open questions
└── CLAUDE.md              # instructions for the AI pair programmer
```

## Documentation

- **[docs/project_notes.md](docs/project_notes.md)**: start here. How each part works and why.
- [docs/specs/](docs/specs/): the spec each milestone was built against.
- [docs/decisions.md](docs/decisions.md): tool and design choices with reasons.
- [PROGRESS.md](PROGRESS.md): session-by-session progress and open questions.

> **Note:** `data/vendor_notice_printer.md` deliberately contains a prompt injection. It is test data
> for the guardrails milestone, not an instruction.
