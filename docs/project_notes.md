# Project Notes: Nodi Tech RAG Chatbot

> My study guide for this project. One section per milestone: what we built, how it works, why we
> chose it, what went wrong, the numbers, and questions I should be able to answer.
> Updated at the end of every milestone. Read this before any demo, review or interview.

---

## Contents

- [The project in 60 seconds](#the-project-in-60-seconds)
- [Architecture and status](#architecture-and-status)
- [Glossary](#glossary)
- [M1 · Ingestion](#m1--ingestion) ✅
- [M2 · Vector search](#m2--vector-search) ⏳
- [M3 · Hybrid search + rerank](#m3--hybrid-search--rerank) ⏳
- [M4 · Chat](#m4--chat) ⏳
- [M5 · Guardrails](#m5--guardrails) ⏳
- [M6 · Evaluation](#m6--evaluation) ⏳
- [Numbers at a glance](#numbers-at-a-glance)

---

## The project in 60 seconds

A chatbot that answers questions about **Nodi Tech Ltd.** (a fictional company) using only 12 internal
documents: HR handbook, IT security policy, an expense policy PDF, product docs for "NodiDesk", a Bangla
document, an outdated policy, and a vendor notice that contains a hidden prompt injection.

It uses **RAG (Retrieval-Augmented Generation)**: instead of trusting what the LLM "remembers", we first
**search** the documents for the relevant pieces, then give those pieces to the LLM and ask it to answer
**only from them**, with a citation. If nothing relevant is found, it must say "I don't know".

Everything runs locally: Qwen via Ollama (LLM), bge-m3 (embeddings), Qdrant (vector DB), Python + FastAPI.

The data contains deliberate traps (exact codes, tables, an outdated policy, Bangla text, a prompt
injection, out-of-scope questions). Success is measured by a 40-question eval set, not by impressions.

---

## Architecture and status

```
INDEXING                                                        status
data/ -> parse (md + PDF) -> chunk -> add context line           ✅ M1
      -> embed -> vector DB (Qdrant)                             ⏳ M2
      -> BM25 index                                              ⏳ M3

CHAT
user message -> rewrite follow-up into standalone question       ⏳ M4
  -> lexical search + vector search -> fuse (RRF) -> rerank      ⏳ M3
  -> prompt: system rules + chat history + fenced chunks         ⏳ M4/M5
  -> LLM answer with [source] citations -> stream to UI          ⏳ M4
```

---

## Glossary

Terms in the order I learned them. Each has a comparison to something I already know.

| Term | Meaning | Like... |
|---|---|---|
| **LLM** | Large Language Model. Predicts the next piece of text. Has no access to our documents by itself. | A very good autocomplete |
| **RAG** | Retrieval-Augmented Generation: search first, then let the LLM answer from the search results. | A REST handler that queries the DB before building the response |
| **Token** | The unit of text a model reads. A word or part of a word. English ≈ 4 chars/token; Bangla needs many more tokens for the same meaning. Models have a max token limit. | A column size limit |
| **Chunk** | A small, self-contained piece of a document. Search works on chunks, not whole files. | A row in an indexed table |
| **Chunking** | Splitting documents into chunks. Too big: one chunk covers many topics and matches weakly. Too small: loses context. | Choosing the granularity of rows |
| **Context line** | A header we add to each chunk (`[Source \| title \| Section]`) so it says where it came from. | Denormalizing the parent's name into a child row |
| **Metadata** | Data about a chunk (source file, section, status), stored next to its text. | Extra columns |
| **Deterministic** | Same input always gives the same output (same chunk IDs every run). | A pure function |
| **Heuristic** | A rule that usually works but is not guaranteed (e.g. "bigger font = heading"). | |
| **Prompt injection** | Text inside data that tries to give the AI instructions ("ignore previous instructions..."). | SQL injection, but for prompts |

---

## M1 · Ingestion

**Goal:** turn all 12 files (11 markdown + 1 PDF) into clean, citable chunks.
**Status:** ✅ done · 50 chunks · 33 tests passing · Spec: [specs/M1.md](specs/M1.md)

### What we built

| File | Job |
|---|---|
| `src/ingest/parse.py` | File → `Document(source, title, text, status)`. Markdown read as-is; PDF converted to markdown. |
| `src/ingest/chunk.py` | `Document` → list of `Chunk`s. Split by headings, size limit, tables keep their header, context line. |
| `src/ingest/run.py` | CLI: runs both steps, prints every chunk, saves `artifacts/chunks.jsonl`. |
| `tests/ingest/` | 33 pytest tests (12 parse, 21 chunk). |

### How it works

```
data/*.md  --read utf-8------------------------------\
                                                      >-- markdown text --> split_sections()
data/*.pdf --pdfplumber: lines + font sizes + table--/       (by ## / ###, keep heading path)
                                                                   |
                                                             split_content()
                                                (only if > 1500 chars: blocks -> lines -> words;
                                                 tables by rows, header repeated)
                                                                   |
                                                    Chunk + context line --> chunks.jsonl
```

**Key idea:** the PDF is converted into the *same markdown format* as the other files, so one chunker
handles everything. One code path means fewer bugs.

A chunk looks like this (`embed_text` = what will be searched):

```
[Source: employee_handbook_v3.md | Nodi Tech Ltd. — Employee Handbook | Section: 3. Leave > Carry forward]
Up to 5 unused annual leave days can be carried forward to the next year. Carried-forward days must be
used by 31 March, otherwise they lapse. Casual and sick leave cannot be carried forward.
```

### Decisions and why

| Decision | Why | Alternative rejected |
|---|---|---|
| Chunk by headings (`##`, `###`) | The author already split topics with headings. Sections here are small (max 668 chars), so each is one clean topic. | Fixed-size chunks (e.g. 500 chars + overlap): can cut a sentence in half or separate a table from its header. |
| Max 1500 chars per chunk | Big enough that no real section is split; small enough to keep one topic per chunk. | — |
| Tables never split from header | A row like `\| ND-1042 \| 429 \|` means nothing without `\| Code \| HTTP status \|`. | — |
| `pdfplumber` for the PDF | Extracts tables as rows/cells and gives font sizes. | `pypdf` (text only, table jumbled), `PyMuPDF` (AGPL license). |
| PDF headings from **font size** | Title 17pt, headings 13pt, body 11pt. Sizes are computed per document, not hard-coded. | Regex like `^\d+\. ` (fragile, guesses from text). |
| Context line built from metadata | Tiny chunks like "Casual leave: 10 days" get meaning (which doc, which section). Free and deterministic. | LLM-written context per chunk (brief's stretch goal; slower, costs LLM calls). |
| `status: superseded` from filename | Lets later steps prefer the current handbook (18 days) over the old policy (15 days). | — |
| Prompt injection **not removed** | Ingestion should not edit data. Removing it would hide the problem; the real defence is fencing in M5. | Stripping "SYSTEM NOTE" lines (brittle, attacker just rewords). |
| Deterministic `chunk_id` = `source#n` | Re-running gives the same IDs, so re-indexing in Qdrant overwrites instead of duplicating. | Random UUIDs. |

### Problems we hit and how we solved them

1. **PDF table came out jumbled.** Plain text extraction reads by position, so wrapped cells interleave:
   ```
   BDT 6,000 per
   Hotel outside Dhaka Pre-approval from team lead required
   night
   ```
   **Fix:** pdfplumber's `find_tables()` returns proper rows; we join `\n` inside cells with a space →
   `| Hotel outside Dhaka | BDT 6,000 per night | Pre-approval from team lead required |`.
2. **Table would appear twice** (clean table + jumbled text of the same area).
   **Fix:** skip any text line that sits inside the table's bounding box (`bbox`).
3. **PDF footer glued onto the last paragraph.** **Fix:** start a new paragraph when the vertical gap
   between lines is bigger than the font size.
4. **Windows encoding.** Windows defaults to cp1252, which corrupts/crashes on Bangla.
   **Fix:** always `encoding="utf-8"` when reading/writing; `sys.stdout.reconfigure(encoding="utf-8")` when printing.
5. **Python was not installed.** Installed 3.12.10 with winget (user scope), created `.venv`.

### How we know it works

- 33 tests: every file chunked, PDF table intact with header, no duplicate table text, heading paths,
  Bangla preserved, `ND-1042`/`SEC-EXC` verbatim, status flags, determinism, and every source document
  in the eval CSV actually exists as a chunk.
- Splitting rules (large table, long section, giant line) tested with **synthetic input**, because no
  real section is long enough to trigger them.
- **Mutation check:** deliberately broke the code (no header repeat; flat heading path). Tests failed
  both times. That proves the tests check behaviour, not just "it runs".
- Eyeballed the full printout of all 50 chunks.

### Numbers

| Metric | Value |
|---|---|
| Files | 12 (11 md + 1 PDF) |
| Chunks | 50 |
| Chunk size (chars) | min 43 · max 668 · avg 204 |
| Sections split for size | 0 |
| Tests | 33 passed |

### Known limitations

- Font-size heading detection fails if a PDF uses the same size for headings and body.
- The PDF footer ("Questions: finance@...") lands inside section "4. Not reimbursable".
- The FAQ (6 Q&As, no headings) is one chunk covering 6 topics; may weaken retrieval.
- Superseded status comes from the filename, not the content.
- No real token counts yet (only characters); measured with the bge-m3 tokenizer in M2.

### Questions I should be able to answer

<details>
<summary><b>Why do we chunk documents at all?</b></summary>

Search returns whole units. If the unit is a whole document, a question about carry-forward returns the
entire handbook, mostly irrelevant text, which wastes the LLM's context and confuses it. Chunks let
search return just the relevant piece. Also, an embedding of a long text averages many topics together
and matches every question weakly.
</details>

<details>
<summary><b>Why heading-based chunking instead of fixed-size chunks with overlap?</b></summary>

Our documents are well structured and every section is small, so a section is a natural single-topic
chunk. Fixed-size chunking ignores structure: it can cut mid-sentence or separate a table row from its
header. Overlap is the usual patch for that, but it creates duplicate text. With structured docs,
following the structure is simpler and cleaner.
</details>

<details>
<summary><b>Why are tables tricky, and how did you handle the PDF table?</b></summary>

A table row only has meaning with its header. In a PDF there is no "table" at all, only text placed at
x/y positions, so plain extraction interleaves wrapped cells. We used pdfplumber's table detector to get
real rows and cells, joined wrapped cell text, rendered it as a markdown table, skipped the raw text
inside the table's bounding box to avoid duplicates, and made sure a table is never split from its
header (if split, the header is repeated in every piece).
</details>

<details>
<summary><b>How do you detect headings in the PDF?</b></summary>

By font size, measured per document: the largest size is the title, the most common size is body text,
anything in between is a section heading. It's a heuristic: it works for this PDF, and it doesn't depend
on hard-coded sizes, but it would fail for a PDF where headings and body share a size.
</details>

<details>
<summary><b>What is the context line, and why add it?</b></summary>

A one-line header on every chunk: source file, document title, section path, and SUPERSEDED if
relevant. A chunk like "Casual leave: 10 days. Sick leave: 14 days." doesn't say which company or
document it's from. The context line puts that into the text we search, so both vector and keyword
search can use it, and the LLM can see that a chunk is outdated.
</details>

<details>
<summary><b>Why didn't you remove the prompt injection during ingestion?</b></summary>

Ingestion shouldn't edit source data, and filtering by pattern is brittle because an attacker just
rewords. In real life we can't know every injected phrase. The robust defence is at prompt time
(M5): retrieved chunks are fenced and labelled as untrusted data, never instructions. Keeping it in
the index also lets us test that defence.
</details>

<details>
<summary><b>How will you handle the outdated leave policy (15 vs 18 days)?</b></summary>

M1 marks every chunk from that file `status: superseded`, and the context line says "Status:
SUPERSEDED". Later steps (prompt rules in M4/M5) use that to prefer the current handbook. The eval
requires the answer to be 18 days, never 15.
</details>

<details>
<summary><b>How do you know your chunker is correct?</b></summary>

33 pytest tests on real data and synthetic edge cases, a mutation check (we broke the code on purpose
and the tests caught it), and a manual review of all 50 printed chunks.
</details>

---

## M2 · Vector search

⏳ Not started.

---

## M3 · Hybrid search + rerank

⏳ Not started.

---

## M4 · Chat

⏳ Not started.

---

## M5 · Guardrails

⏳ Not started.

---

## M6 · Evaluation

⏳ Not started.

---

## Numbers at a glance

| Milestone | Key result |
|---|---|
| M1 | 50 chunks from 12 files · 33 tests passing |
| M2 | — |
| M3 | — |
| M4 | — |
| M5 | — |
| M6 | Retrieval hit rate: — (target ≥ 90%) · Answer correctness: — (target ≥ 85%) |
