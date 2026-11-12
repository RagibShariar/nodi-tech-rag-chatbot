# Project Notes · প্রজেক্ট নোট

### Nodi Tech RAG Chatbot

> **English:** My study guide for this project. For every milestone: what we built, how it works, why we
> chose it, what went wrong, the real numbers, and questions I should be able to answer.
> Read it before any demo, review or interview.
>
> **বাংলায়:** এটা আমার পড়ার গাইড। প্রতিটা ধাপে কী বানালাম, কীভাবে কাজ করে, কেন এভাবে করলাম,
> কী ভুল হয়েছিল, আসল সংখ্যা কত, আর কেউ প্রশ্ন করলে কী উত্তর দেব, সব এখানে।
> নিয়ম: **English অংশ মুখে বলার জন্য, বাংলা অংশ বোঝার জন্য।**

---

## Contents · সূচিপত্র

- [0. What was given vs what I built](#0-what-was-given-vs-what-i-built)
- [1. The project in 60 seconds](#1-the-project-in-60-seconds)
- [2. Architecture and status](#2-architecture-and-status)
- [3. Glossary · শব্দকোষ](#3-glossary--শব্দকোষ)
- [M1 · Ingestion](#m1--ingestion) ✅
- [M2 · Vector search](#m2--vector-search) ✅
- [M3 · Hybrid search + rerank](#m3--hybrid-search--rerank) ⏳
- [M4 · Chat](#m4--chat) ⏳
- [M5 · Guardrails](#m5--guardrails) ⏳
- [M6 · Evaluation](#m6--evaluation) ⏳
- [Numbers at a glance](#numbers-at-a-glance)

---

## 0. What was given vs what I built

| Given (assignment kit, unmodified) | Built by me |
|---|---|
| `rag_chatbot_project_brief.md`: the assignment | `src/`: all code |
| `data/`: 12 company documents (with deliberate traps) | `tests/`: all tests |
| `eval/eval_questions.csv`: 40 test questions | `docs/`: specs, decisions, experiments, these notes |
| `eval/conversation_tests.json`: 3 multi-turn chats | `README.md` (the kit's README was a copy of the brief; replaced with a project README), `PROGRESS.md`, `CLAUDE.md` |

**বাংলায়:** পরীক্ষার প্রশ্নপত্র আর বই দেওয়া হয়েছিল (brief, data, eval)। ওগুলো আমি বদলাইনি, কারণ
প্রশ্নপত্র বদলালে পরীক্ষাটাই মিথ্যা হয়ে যায়। উত্তরের খাতা, মানে কোড, টেস্ট আর ডকুমেন্টেশন, আমার নিজের।

---

## 1. The project in 60 seconds

**English (say this):**
A chatbot that answers questions about Nodi Tech Ltd., a fictional company, using only 12 internal
documents. It uses **RAG**: before answering, it searches the documents for the relevant pieces, then
the LLM answers **only from those pieces**, with a citation. If nothing relevant is found, it says
"I don't know". Everything runs locally: Qwen via Ollama, bge-m3 embeddings, Qdrant, Python + FastAPI.
The data has deliberate traps (exact error codes, tables, an outdated policy, a Bangla document, a hidden
prompt injection), and success is measured by a 40-question evaluation, not by impressions.

**সহজ বাংলায়:**
ভাবো, অফিসে একজন নতুন কর্মী। তাকে কোম্পানির ১২টা ফাইল দেওয়া হলো। কেউ কিছু জিজ্ঞেস করলে
সে **নিজের মাথা থেকে বানিয়ে বলে না**। আগে ফাইল ঘেঁটে সঠিক পাতাটা বের করে, তারপর সেখান থেকে
উত্তর দেয় আর বলে দেয় "এটা হ্যান্ডবুকের ৩ নম্বর অংশে আছে"। ফাইলে না থাকলে সোজা বলে
"আমি জানি না"। আমাদের চ্যাটবট ঠিক এই কর্মী।

---

## 2. Architecture and status

```
INDEXING (একবার, আগে থেকে)                                      status
data/ -> parse (md + PDF) -> chunk -> add context line           ✅ M1
      -> embed -> vector DB (Qdrant)                             ✅ M2
      -> BM25 index                                              ⏳ M3

CHAT (প্রতিটা প্রশ্নে)
user message -> rewrite follow-up into standalone question       ⏳ M4
  -> lexical search + vector search -> fuse (RRF) -> rerank      ⏳ M3
  -> prompt: system rules + chat history + fenced chunks         ⏳ M4/M5
  -> LLM answer with [source] citations -> stream to UI          ⏳ M4
```

**বাংলায়:** দুইটা আলাদা কাজ। **Indexing** হলো লাইব্রেরি সাজানো: বই এনে, তাকে তাকে রেখে,
সূচিপত্র বানানো। এটা একবারই করতে হয়। **Chat** হলো কেউ এসে প্রশ্ন করলে সেই সাজানো লাইব্রেরি
থেকে খুঁজে উত্তর দেওয়া। এটা প্রতিটা প্রশ্নে হয়।

---

## 3. Glossary · শব্দকোষ

যে ক্রমে শিখেছি, সেই ক্রমে সাজানো। প্রতিটা শব্দের আগে English সংজ্ঞা, তারপর বাংলায় বাস্তব উদাহরণ।

### LLM (Large Language Model)
**English:** A model that predicts the next piece of text. It has never seen our company documents,
and when it doesn't know, it can confidently make something up (**hallucination**).
**বাংলায়:** মোড়ের চায়ের দোকানের "সবজান্তা মামা"। দুনিয়ার সব বিষয়ে কথা বলে, খুব
আত্মবিশ্বাস নিয়ে। কিন্তু তোমার অফিসের ছুটির নিয়ম তিনি জানেন না, তবুও জিজ্ঞেস করলে একটা উত্তর
বানিয়ে দেবেন। এই বানিয়ে বলাটাই hallucination।

### RAG (Retrieval-Augmented Generation)
**English:** Search first (retrieval), then let the LLM write the answer (generation) only from the
search results.
**বাংলায়:** **Open-book পরীক্ষা।** মুখস্থ থেকে না লিখে বই খুলে সঠিক পাতা বের করে উত্তর লেখা,
আর পাশে পাতা নম্বর লিখে দেওয়া (এটাই citation)। বইয়ে না থাকলে লিখতে হয় "বইয়ে নেই", বানানো নিষেধ।

### Token
**English:** The unit of text a model reads: a word or part of a word. Every model has a maximum number
of tokens it can read at once. How many tokens a language needs depends on the model's **tokenizer**:
measured with bge-m3, the same bus sentence was 16 tokens in English and 22 in Bangla (~1.4×).
Tokenizers trained mostly on English split Bangla into far more pieces.
**বাংলায়:** **খুচরা পয়সা।** মডেল পুরো টাকার নোট নেয় না, ভাঙিয়ে খুচরা করে নেয়। কে কত
খুচরা দেবে, সেটা নির্ভর করে দোকানদারের (tokenizer) উপর। bge-m3 বাংলা ভালো চেনে, তাই বাংলায়
মাত্র ~১.৪ গুণ বেশি কয়েন লাগল। যে দোকানদার শুধু ইংরেজি চেনে, সে বাংলা ভাঙাতে অনেক বেশি কয়েন
ধরিয়ে দেবে। আর মডেলের মানিব্যাগে একসাথে কতগুলো কয়েন ধরে, তার একটা সীমা আছে (token limit)।

### Chunk / Chunking
**English:** A chunk is a small, self-contained piece of a document. Search works on chunks, not
whole files. Too big: one chunk covers many topics and matches every question weakly. Too small: it
loses its meaning.
**বাংলায়:** **কাঁঠালের কোয়া।** পুরো কাঁঠাল একসাথে খাওয়া যায় না, কোয়া আলাদা করতে হয়।
প্রকৃতি নিজেই কোয়া ভাগ করে রেখেছে, ঠিক যেমন লেখক heading দিয়ে বিষয় ভাগ করে রাখেন। আমরা
heading ধরে কাটি, মানে কোয়া ধরে আলাদা করি। ছুরি দিয়ে সমান মাপে কাটলে (fixed-size chunking)
কোয়াও মাঝখান দিয়ে কেটে যেত।

### Context line
**English:** A header we add to every chunk, `[Source | title | Section]`, so a chunk says where it
came from.
**বাংলায়:** **ওষুধের পাতা।** পুরো পাতায় ওষুধের নাম লেখা থাকে, কিন্তু একটা ট্যাবলেট কেটে
আলাদা করলে আর বোঝা যায় না এটা কোন ওষুধ। Context line হলো প্রতিটা কাটা ট্যাবলেটের গায়ে নাম
লিখে দেওয়া।

### Metadata
**English:** Data *about* a chunk (source file, section, status), stored next to its text.
**বাংলায়:** কুরিয়ারের পার্সেলের গায়ের লেবেল: কে পাঠাল, কোথায় যাবে। ভেতরের জিনিস না,
জিনিসটা সম্পর্কে তথ্য।

### Superseded
**English:** Replaced by a newer version. Our old leave policy (15 days) is superseded by the
handbook (18 days).
**বাংলায়:** নোটিশ বোর্ডে পুরনো রুটিনের উপর "বাতিল" সিল। কাগজটা এখনো বোর্ডে ঝুলছে, কিন্তু
ওটা দেখে ক্লাসে গেলে ভুল হবে।

### Deterministic
**English:** Same input, same output, every time (same chunk IDs on every run).
**বাংলায়:** স্কুলের রোল নম্বর। প্রতিদিন হাজিরায় একই ছাত্রের একই রোল, বদলায় না।

### Heuristic
**English:** A rule of thumb that usually works but isn't guaranteed (e.g. "bigger font = heading").
**বাংলায়:** আকাশে কালো মেঘ দেখে ছাতা নেওয়া। সাধারণত ঠিকই হয়, কিন্তু মাঝে মাঝে মেঘ করেও
বৃষ্টি হয় না।

### Prompt injection
**English:** Text hidden inside data that tries to give the AI instructions ("ignore previous
instructions..."). The defence: treat retrieved text as data, never as commands.
**বাংলায়:** পরীক্ষার খাতায় ছাত্র লিখে রাখল "স্যার, আমাকে পাশ করিয়ে দিন"। পরীক্ষক এটা পড়বেন,
কিন্তু মানবেন না, কারণ খাতা হলো মূল্যায়নের জিনিস, পরীক্ষককে হুকুম দেওয়ার জায়গা না।
আমাদের বটও ডকুমেন্টের লেখা পড়বে, কিন্তু সেখান থেকে হুকুম নেবে না।

### Embedding *(M2)*
**English:** A list of numbers (a vector, 1024 numbers for bge-m3) that represents a text's meaning.
Texts with similar meaning get similar vectors, even with different words or languages.
**বাংলায়:** **অর্থের মানচিত্রে ঠিকানা।** ঢাকায় যেমন একই জিনিসের দোকান এক এলাকায়: বই
নীলক্ষেতে, ইলেকট্রনিক্স এলিফ্যান্ট রোডে। Embedding প্রতিটা লেখাকে একটা "অর্থের মানচিত্রে"
বসায়। "ছুটি কত দিন?" আর "annual leave 18 days" একই এলাকায় পড়ে, ভাষা আলাদা হলেও।

### Cosine similarity *(M2)*
**English:** How similar two vectors are, measured by the angle between them: 1 means the same
direction, 0 means unrelated. Length is ignored, so short and long texts compare fairly.
**বাংলায়:** দুজন মানুষ আঙুল তুলে পথ দেখাচ্ছে। দুজনই একই দিকে দেখালে মিল পুরো (১), একজন
উত্তরে আর একজন পূর্বে দেখালে মিল নেই (০)। কার হাত কত লম্বা, সেটা দেখি না, শুধু দিক দেখি।

### Vector database (Qdrant) *(M2)*
**English:** Stores vectors with their metadata and quickly answers "give me the 5 nearest vectors".
Postgres table → collection, row → point, B-tree → HNSW index.
**বাংলায়:** নীলক্ষেতের অভিজ্ঞ বইওয়ালা। "এই ধরনের একটা বই চাই" বললেই হাজার বইয়ের মধ্য
থেকে কাছাকাছি ৫টা বের করে দেন। সূচিপত্রের মতো হুবহু নাম মেলাতে হয় না, ধরন মিললেই হয়।

### Embedding model vs LLM *(M2)*
**English:** Two different models with two jobs. The embedding model **finds** (turns text into
vectors); the LLM **writes** the answer. You don't embed with a chat LLM: it's trained to predict the
next word, not to place similar meanings close together.
**বাংলায়:** লাইব্রেরিতে **লাইব্রেরিয়ান** আর **লেখক** আলাদা মানুষ। লাইব্রেরিয়ান (embedding
model, যেমন bge-m3) ঠিক বইটা খুঁজে আনেন। লেখক (LLM, যেমন Qwen) সেই বই পড়ে উত্তর লেখেন।
ভালো লেখককে বই খোঁজার কাজে লাগালে কাজ হবে, কিন্তু ভালো হবে না।

### hit@k *(M2)*
**English:** For a question, is at least one correct source document among the top-k results? Reported
as a percentage over all questions. Our target: hit@5 ≥ 90%.
**বাংলায়:** বাজারে ৫টা দোকান ঘুরে চাওয়া জিনিসটা অন্তত একটায় পেলে "হিট"। কোন দোকানে
পেলাম সেটা এখানে দেখি না, শুধু পেলাম কি না।

### MRR (Mean Reciprocal Rank) *(M2)*
**English:** 1 / (rank of the first correct result), averaged over questions. 1st place = 1.0,
2nd = 0.5, 5th = 0.2, not found = 0. Rewards putting the right answer *first*.
**বাংলায়:** Google-এ সঠিক লিংক প্রথমেই পাওয়া আর পঞ্চম নম্বরে পাওয়া এক কথা না। প্রথম
দোকানেই জিনিস পেলে পুরো নম্বর, পঞ্চম দোকানে গিয়ে পেলে অল্প নম্বর।

### Ceiling effect *(M2)*
**English:** When a metric is already at its maximum (hit@5 = 100%), it can't show further
improvement, so you need a finer metric (MRR) to compare.
**বাংলায়:** ক্লাসের সবাই ১০০-তে ১০০ পেলে কে বেশি ভালো, পরীক্ষা দিয়ে আর বোঝা যায় না। তখন
কঠিন প্রশ্ন লাগে, বা দেখতে হয় কে কত তাড়াতাড়ি শেষ করল।

### Dependency injection / fake *(M2)*
**English:** Passing a function (here: the embedder) in as a parameter, so tests can pass a fake one and
run without the real model.
**বাংলায়:** গাড়ি চালানো শেখার সময় আসল রাস্তায় না নেমে সিমুলেটরে শেখা। স্টিয়ারিং
(আমাদের কোড) আসল, রাস্তা (মডেল) নকল, তাই দ্রুত আর নিরাপদে পরীক্ষা করা যায়।

---

## M1 · Ingestion

**Goal:** turn all 12 files (11 markdown + 1 PDF) into clean, citable chunks.
**লক্ষ্য:** ১২টা ফাইলকে পরিষ্কার, ছোট ছোট টুকরায় ভাগ করা, যাতে প্রতিটা টুকরা বলতে পারে সে কোথা থেকে এসেছে।
**Status:** ✅ done · 50 chunks · 33 tests passing · Spec: [specs/M1.md](specs/M1.md)

### What we built

| File | Job |
|---|---|
| `src/ingest/parse.py` | File → `Document(source, title, text, status)`. Markdown read as-is; PDF converted to markdown. |
| `src/ingest/chunk.py` | `Document` → `Chunk`s. Split by headings, size limit, tables keep their header, context line. |
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
handles everything.

**সহজ বাংলায়, রান্নাঘরের উদাহরণ:**
1. **বাজার করা (parse):** সব ফাইল পড়া। ১১টা markdown ফাইল যেন কাটা-বাছা মাছ, সরাসরি নেওয়া যায়।
   PDF-টা যেন আস্ত মাছ, আগে আঁশ ছাড়িয়ে পরিষ্কার করে বাকিগুলোর মতো বানিয়ে নিতে হয়
   (PDF থেকে markdown)। তারপর একটাই বঁটিতে সব কাটা যায়।
2. **কাটা (chunk):** heading ধরে টুকরা করা। কোনো টুকরা অনেক বড় হলে আরও ছোট করা।
3. **লেবেল লাগানো (context line):** ফ্রিজের প্রতিটা প্যাকেটে লেখা "রুই, পেটি, ১২ তারিখ"।
4. **ফ্রিজে রাখা (`chunks.jsonl`):** পরের ধাপ (M2) এখান থেকে নেবে।

A chunk looks like this (`embed_text` = what will be searched):

```
[Source: employee_handbook_v3.md | Nodi Tech Ltd. — Employee Handbook | Section: 3. Leave > Carry forward]
Up to 5 unused annual leave days can be carried forward to the next year. Carried-forward days must be
used by 31 March, otherwise they lapse. Casual and sick leave cannot be carried forward.
```

### Decisions and why

| Decision | Why | Alternative rejected |
|---|---|---|
| Chunk by headings (`##`, `###`) | The author already split topics with headings. Sections here are small (max 668 chars), so each is one clean topic. | Fixed-size chunks + overlap: can cut a sentence or separate a table from its header. |
| Max 1500 chars per chunk | Big enough that no real section is split; small enough to keep one topic per chunk. | — |
| Tables never split from header | A row like `\| ND-1042 \| 429 \|` means nothing without `\| Code \| HTTP status \|`. | — |
| `pdfplumber` for the PDF | Extracts tables as rows/cells and gives font sizes. | `pypdf` (text only, table jumbled), `PyMuPDF` (AGPL license). |
| PDF headings from **font size** | Title 17pt, headings 13pt, body 11pt, measured per document, not hard-coded. | Regex like `^\d+\. ` (guesses from text). |
| Context line from metadata | Tiny chunks get meaning. Free and deterministic. | LLM-written context per chunk (stretch goal; slower). |
| `status: superseded` from filename | Later steps can prefer the handbook (18 days) over the old policy (15 days). | — |
| Prompt injection **not removed** | Ingestion shouldn't edit data; the real defence is fencing in M5. | Stripping "SYSTEM NOTE" lines (brittle; attacker rewords). |
| Deterministic `chunk_id` = `source#n` | Re-indexing overwrites instead of duplicating. | Random UUIDs. |

**বাংলায় মূল "কেন"গুলো:**
- **Table কখনো header থেকে আলাদা না:** SSC-র রেজাল্ট শিটে একটা ঘরে লেখা "৮৫"। উপরের header
  না দেখলে বোঝার উপায় নেই এটা বাংলার নম্বর না অঙ্কের। টেবিলের প্রতিটা সারির মানে header-এর উপর নির্ভর করে।
- **Injection মুছে ফেলিনি:** চোর ধরতে হলে দরজায় তালা দিতে হয় (M5-এর fencing)। চোরের ছবি
  দেয়াল থেকে সরিয়ে ফেললে চুরি থামে না। আর injection index-এ থাকলে আমরা পরীক্ষা করতে পারব যে
  তালা আসলেই কাজ করছে কি না।

### Problems we hit and how we solved them

1. **PDF table came out jumbled.** Plain text extraction reads by position, so wrapped cells interleave:
   ```
   BDT 6,000 per
   Hotel outside Dhaka Pre-approval from team lead required
   night
   ```
   **Fix:** pdfplumber's `find_tables()` returns real rows; `\n` inside cells joined with a space →
   `| Hotel outside Dhaka | BDT 6,000 per night | Pre-approval from team lead required |`.

   **বাংলায়:** PDF আসলে ছাপানো কাগজের মতো, "কোন অক্ষর পাতার কোথায় বসবে" শুধু এই তথ্য থাকে।
   টেবিল বলে কিছু চেনে না। তাই সাধারণভাবে পড়লে মনে হয় পাশাপাশি দুই কলামের লেখা এক লাইনে
   মিশে গেছে, যেন দুই কলামের পত্রিকা এক লাইন করে পড়ছি। pdfplumber টেবিলের দাগ (ঘর) চিনে আলাদা
   করে দেয়।
2. **Table would appear twice** (clean table + jumbled text of the same area).
   **Fix:** skip text lines inside the table's bounding box (`bbox`).
   **বাংলায়:** একই খবর দুইবার ছাপা হলে পাঠক বিভ্রান্ত হয়। টেবিলের এলাকার লেখা একবারই নিই।
3. **PDF footer glued onto the last paragraph.** **Fix:** start a new paragraph when the vertical gap
   between lines is bigger than the font size.
   **বাংলায়:** দুই লাইনের মাঝে বড় ফাঁকা মানে নতুন অনুচ্ছেদ, ঠিক যেমন চিঠিতে নতুন কথা শুরুর আগে একটু জায়গা ছাড়ি।
4. **Windows encoding.** Windows defaults to cp1252, which corrupts Bangla.
   **Fix:** always `encoding="utf-8"`; `sys.stdout.reconfigure(encoding="utf-8")` when printing.
   **বাংলায়:** ভুল encoding মানে ভুল অভিধান দিয়ে চিঠি পড়া, বাংলা অক্ষর হিজিবিজি হয়ে যায়।
5. **Python was not installed.** Installed 3.12.10 via winget, created `.venv`.

### How we know it works

- **33 tests** on real data: every file chunked, PDF table intact with header, no duplicate table text,
  heading paths, Bangla preserved, `ND-1042`/`SEC-EXC` verbatim, status flags, determinism, and every
  source document in the eval CSV exists as a chunk.
- **Synthetic tests** for splitting rules, because no real section is long enough to trigger them.
- **Mutation check:** broke the code on purpose (no header repeat; flat heading path). Tests failed both
  times, which proves they check behaviour, not just "it runs".
- **Eyeballed** all 50 chunks.

**বাংলায়:** Mutation check মানে দারোয়ান জেগে আছে কি না দেখতে নিজেই চোর সেজে গেটে যাওয়া।
দারোয়ান ধরে ফেললে বোঝা যায় পাহারা আসল, শুধু চেয়ারে বসে থাকা না। আমাদের test-রা দুইবারই "চোর" ধরেছে।

### Numbers

| Metric | Value |
|---|---|
| Files | 12 (11 md + 1 PDF) |
| Chunks | 50 |
| Chunk size (chars) | min 43 · max 668 · avg 204 |
| Sections split for size | 0 |
| Tests | 33 passed |

### Known limitations · দুর্বলতা

- Font-size heading detection fails if a PDF uses the same size for headings and body.
  *(মেঘ দেখে ছাতা: এই PDF-এ কাজ করেছে, সব PDF-এ করবে এমন গ্যারান্টি নেই।)*
- PDF footer ("Questions: finance@...") lands inside section "4. Not reimbursable".
- FAQ (6 Q&As, no headings) is one chunk covering 6 topics; may weaken retrieval.
  *(একটা কোয়ায় ৬টা বিচি। M6-এর score দেখে ঠিক করব আলাদা করব কি না।)*
- Superseded status comes from the filename, not the content.
- No real token counts yet; measured in M2.

### Questions I should be able to answer

<details>
<summary><b>Why do we chunk documents at all?</b></summary>

**Answer:** Search returns whole units. If the unit is a whole document, a question about carry-forward
returns the entire handbook, mostly irrelevant text, which wastes the LLM's context and confuses it.
Also, one embedding of a long text averages many topics and matches every question weakly.

**বুঝে নাও:** কেউ "ছুটি কত দিন" জিজ্ঞেস করলে পুরো হ্যান্ডবুক হাতে ধরিয়ে দেওয়া আর সঠিক
অনুচ্ছেদটা আঙুল দিয়ে দেখিয়ে দেওয়ার মধ্যে পার্থক্য।
</details>

<details>
<summary><b>Why heading-based chunking instead of fixed-size chunks with overlap?</b></summary>

**Answer:** Our documents are well structured and every section is small, so a section is a natural
single-topic chunk. Fixed-size chunking ignores structure: it can cut mid-sentence or separate a table
row from its header. Overlap patches that but duplicates text. With structured docs, following the
structure is simpler and cleaner.

**বুঝে নাও:** কাঁঠালের কোয়া ধরে আলাদা করা বনাম ছুরি দিয়ে সমান মাপে কাটা। Overlap মানে প্রতিটা
টুকরায় পাশের টুকরার একটু করে রেখে দেওয়া, যাতে কাটা জায়গার ক্ষতি কম হয়। কিন্তু কোয়া ধরে কাটলে
কাটা জায়গাই থাকে না।
</details>

<details>
<summary><b>Why are tables tricky, and how did you handle the PDF table?</b></summary>

**Answer:** A table row only has meaning with its header. A PDF has no "table", only text placed at x/y
positions, so plain extraction interleaves wrapped cells. We used pdfplumber's table detector to get
real rows and cells, joined wrapped cell text, rendered a markdown table, skipped the raw text inside
the table's bounding box to avoid duplicates, and never split a table from its header (if split, the
header is repeated in every piece).

**বুঝে নাও:** রেজাল্ট শিটের "৮৫" header ছাড়া অর্থহীন। আর PDF হলো ছাপানো কাগজ, তার ভেতরে
টেবিলের কোনো কাঠামো নেই, তাই ঘরগুলো আলাদা করে চিনিয়ে দিতে হয়।
</details>

<details>
<summary><b>How do you detect headings in the PDF?</b></summary>

**Answer:** By font size, measured per document: the largest size is the title, the most common size is
body text, anything in between is a heading. It's a heuristic: no hard-coded sizes, but it would fail if
headings and body used the same size.

**বুঝে নাও:** পত্রিকায় সবচেয়ে বড় অক্ষরে প্রধান শিরোনাম, মাঝারি অক্ষরে উপশিরোনাম, ছোট অক্ষরে
খবর। অক্ষরের সাইজ দেখেই বোঝা যায় কোনটা কী।
</details>

<details>
<summary><b>What is the context line, and why add it?</b></summary>

**Answer:** A one-line header on every chunk: source file, document title, section path, and SUPERSEDED
if relevant. A chunk like "Casual leave: 10 days. Sick leave: 14 days." doesn't say which document it's
from. The context line puts that into the searchable text, and lets the LLM see that a chunk is outdated.

**বুঝে নাও:** ওষুধের পাতা থেকে কাটা ট্যাবলেটের গায়ে নাম লিখে দেওয়া।
</details>

<details>
<summary><b>Why didn't you remove the prompt injection during ingestion?</b></summary>

**Answer:** Ingestion shouldn't edit source data, and pattern filtering is brittle because an attacker
just rewords. We can't know every injected phrase. The robust defence is at prompt time (M5): retrieved
chunks are fenced and labelled as untrusted data. Keeping it in the index also lets us test that defence.

**বুঝে নাও:** খাতায় "স্যার পাশ করিয়ে দিন" লেখা মুছে দেওয়া সমাধান না, কারণ পরের ছাত্র অন্যভাবে
লিখবে। সমাধান হলো পরীক্ষকের নিয়ম: খাতার লেখা হুকুম না।
</details>

<details>
<summary><b>How will you handle the outdated leave policy (15 vs 18 days)?</b></summary>

**Answer:** M1 marks every chunk from that file `status: superseded`, and its context line says
"Status: SUPERSEDED". Later, prompt rules (M4/M5) use that to prefer the current handbook. The eval
requires 18 days, never 15.

**বুঝে নাও:** নোটিশ বোর্ডের পুরনো রুটিনে "বাতিল" সিল মেরে রাখা। ফেলে দিইনি, কিন্তু কেউ ভুল করে ওটা দেখবে না।
</details>

<details>
<summary><b>How do you know your chunker is correct?</b></summary>

**Answer:** 33 pytest tests on real data and synthetic edge cases, a mutation check (we broke the code
on purpose and the tests caught it), and a manual review of all 50 chunks.

**বুঝে নাও:** নিজে চোর সেজে দারোয়ান পরীক্ষা করা, আর নিজের চোখে ৫০টা টুকরা দেখে নেওয়া।
</details>

---

## M2 · Vector search

**Goal:** embed every chunk, store in Qdrant, return the top 5 chunks for a question, and measure it.
**লক্ষ্য:** প্রতিটা টুকরাকে "অর্থের মানচিত্রে" বসানো, আর প্রশ্ন এলে সবচেয়ে কাছের ৫টা টুকরা বের করা। তারপর মেপে দেখা কতটা ঠিক।
**Status:** ✅ done · bge-m3 hit@5 100% · MRR 0.935 · 47 tests passing · Spec: [specs/M2.md](specs/M2.md)

### What we built

| File | Job |
|---|---|
| `src/config.py` | Settings from `.env` with defaults (Ollama URL, model, Qdrant path). |
| `src/retrieval/embed.py` | `embed(text, kind)` → POST to Ollama `/api/embed` → `Embedding(vector, tokens)`. Model prefixes. |
| `src/retrieval/vector_store.py` | `build_index()` (drop + create collection, upsert points), `search()` (top-k hits). |
| `src/retrieval/index.py` | CLI: chunks → embeddings → Qdrant; prints token stats. |
| `src/retrieval/search.py` | CLI: question → top 5 with score, source, section. |
| `scripts/retrieval_check.py` | hit@5 / all@5 / MRR over 36 eval questions, per type, with misses. |
| `tests/retrieval/`, `tests/scripts/` | Fake embedder + in-memory Qdrant (fast); real Ollama (integration, auto-skip). |

### How it works

```
INDEX (once)
chunk.embed_text --POST /api/embed (bge-m3)--> [1024 numbers] --+
chunk fields (source, section, status, content) ----------------+--> Qdrant point
                                     id = uuid5(chunk_id) ------+    (collection chunks_bge-m3, cosine)

SEARCH (every question)
question --POST /api/embed--> [1024 numbers] --> Qdrant: "5 nearest by cosine" --> 5 hits + scores
```

**সহজ বাংলায়, লাইব্রেরির উদাহরণ:**
1. **Index = বই তাকে সাজানো।** প্রতিটা টুকরার জন্য লাইব্রেরিয়ান (bge-m3) একটা "ঠিকানা"
   ঠিক করেন: ১০২৪টা সংখ্যা। একই বিষয়ের টুকরা কাছাকাছি তাকে যায়। Qdrant হলো সেই তাক, যেখানে
   ঠিকানার সাথে বইয়ের পরিচয়পত্রও (source, section) রাখা থাকে।
2. **Search = লাইব্রেরিয়ানকে জিজ্ঞেস করা।** প্রশ্নটারও একটা ঠিকানা বের করা হয়, তারপর
   সেই ঠিকানার সবচেয়ে কাছের ৫টা তাক থেকে বই নামানো হয়।
3. **Embedding আসলে একটা REST call।** তুমি রোজ যে API কল করো, এটাও তাই: text পাঠাই, সংখ্যা ফেরত পাই।

### Decisions and why

| Decision | Why | Alternative rejected |
|---|---|---|
| **bge-m3** as embedding model | Measured best of three (E1): hit@5 100%, MRR 0.935, only model with zero misses. | nomic-embed-text (missed the Bangla cross-language question), qwen3-embedding:0.6b (missed 2 exact-code questions). |
| Ollama for embeddings, via plain `httpx` | One local runtime for embeddings now and the LLM later. A REST call is easy to understand and debug. | sentence-transformers (pulls in PyTorch). |
| Qdrant **local mode** | No Docker here; 50 vectors don't need a server. Same API; `QDRANT_URL` switches to a server. | Docker Desktop + Qdrant server (heavy setup for no gain at this size). |
| Cosine distance | Compares direction (meaning), ignores length, so short and long chunks compare fairly. | Euclidean distance. |
| `id = uuid5(chunk_id)` | Qdrant needs int/UUID; uuid5 is deterministic, so re-indexing overwrites instead of duplicating. | Random UUIDs. |
| Embed first, then drop + recreate the collection | If embedding fails halfway, the old index survives. Recreate means no stale chunks linger. | Upsert only (deleted chunks would stay forever). |
| One collection per model | Vector sizes differ (1024 vs 768); lets us compare models side by side. | One shared collection. |
| `embed_fn` passed in (dependency injection) | Unit tests use a fake embedder + in-memory Qdrant: 6 tests in ~2s, no model. | Tests that always need Ollama (slow, flaky). |
| Query/document **prefixes** per model | nomic and qwen3-embedding were trained with them; without them they work worse. | Same text for both. |

**বাংলায় মূল "কেন"গুলো:**
- **মডেল বেছেছি মেপে, পছন্দে না।** তিনটা মডেলকে একই ৩৬টা প্রশ্নে পরীক্ষা দিইয়েছি। ক্রিকেট
  দলে খেলোয়াড় নেওয়া হয় পারফরম্যান্স দেখে, নাম শুনে না।
- **আগে embed, পরে পুরনো collection মোছা:** বাসা বদলের সময় নতুন বাসার চাবি হাতে পাওয়ার
  আগে পুরনো বাসা ছেড়ে দিলে রাস্তায় থাকতে হয়।
- **Local mode:** ৫০টা বইয়ের জন্য গুদাম ভাড়া (server/Docker) লাগে না, ঘরের আলমারিই (folder) যথেষ্ট।
  বই বেশি হলে গুদামে নিতে শুধু ঠিকানা বদলাতে হবে (`QDRANT_URL`)।

### What the numbers taught us

1. **Similarity ≠ correctness.** For "How many days of annual leave do I get?", the **old policy
   (15 days) ranked #1** (0.716) and the current handbook (18 days) #2 (0.686). The old policy is a full
   sentence phrased like the question; the handbook row is terse. Retrieval found both; choosing the
   right one is the LLM's job in M4/M5, using the `SUPERSEDED` label.
   **বাংলায়:** লাইব্রেরিয়ান প্রশ্নের সাথে *কথায়* সবচেয়ে মেলে এমন বই দেন, *সঠিক* বই না।
   পুরনো রুটিন দেখতে নতুনটার মতোই, শুধু "বাতিল" সিলটা পড়ে বুঝতে হয়।
2. **Cross-language works with bge-m3.** English question → Bangla bus chunk at #1 (0.725).
   nomic-embed-text (mostly English) missed it entirely.
   **বাংলায়:** যে লাইব্রেরিয়ান দুই ভাষাই জানেন, তিনি ইংরেজিতে প্রশ্ন শুনে বাংলা বই এনে দিতে পারেন।
3. **Exact codes: small margins.** "ND-1042" → correct row 0.643 vs wrong codes 0.574 / 0.568 (in an
   isolated test). In the real index the whole error table is one chunk, so bge-m3 found it, but
   qwen3-embedding missed `SEC-EXC` and `ND-4000`. That fragility is why M3 adds keyword search.
   **বাংলায়:** "ND-1042" আর "ND-1001" লাইব্রেরিয়ানের কাছে প্রায় একই রকম শোনায়, যেমন দুইটা
   কাছাকাছি মোবাইল নম্বর। এখানে দরকার অক্ষরে অক্ষরে মেলানো (BM25, M3)।
4. **Ceiling effect.** bge-m3 already has hit@5 = 100%, so M3 can't improve hit@5. M3 will be judged by
   **MRR** and **all@5**.
5. **Tokens:** Bangla needed ~1.4× the tokens of English with bge-m3 (22 vs 16), less than I predicted.

### Problems we hit and how we solved them

1. **CPU is slow:** ~3s per chunk (161s to index 50). No supported GPU (AMD Vega 11 iGPU). Fine at this
   size; batching is the fix if we ever need speed.
2. **Fake embedder gave nonsense rankings.** Two causes, both found by measuring, not guessing:
   punctuation glued to words ("march," ≠ "march"), and only 64 dimensions → hash collisions.
   **Fix:** strip punctuation, 1024 dimensions. Lesson: test fakes must be correct too.
   **বাংলায়:** সিমুলেটরের রাস্তা ভুল বানালে শেখাটাও ভুল হয়।
3. **CSV `KeyError: 'id'`:** the eval CSV starts with an invisible BOM (`EF BB BF`), so the first column
   was `﻿id`. **Fix:** `encoding="utf-8-sig"`. Same bug class we handled for markdown in M1.
   **বাংলায়:** খাতার প্রথম পাতায় অদৃশ্য কালির দাগ, চোখে দেখা যায় না কিন্তু কলামের নাম বদলে দেয়।
4. **My prediction was wrong:** I said Bangla needs "many more" tokens; measured ~1.4× with bge-m3.
   Corrected in the glossary. Lesson: measure, don't assume.

### How we know it works

- **47 tests:** 6 vector-store unit tests (fake embedder, in-memory Qdrant: count, no duplicates on
  re-index, stale points removed, sorted top-k, deterministic IDs), 5 embed tests (prefixes, cosine math,
  + 3 integration with real bge-m3), 3 tests for the retrieval-check metrics, plus all 33 from M1.
- **Real searches** shown for 4 question types (HR, exact code, cross-language, version conflict).
- **Baseline** on 36 eval questions × 3 models, recorded in [experiments.md](experiments.md) (E1).

### Numbers

| Metric | Value |
|---|---|
| Vectors | 50 × 1024 dims (bge-m3) |
| Tokens per chunk | min 45 · max 257 · avg 94 (bge-m3 limit: 8192) |
| Index time | 161s on CPU (~3s/chunk) |
| bge-m3 | hit@5 **100%** (36/36) · all@5 34/36 · MRR **0.935** |
| nomic-embed-text | hit@5 97.2% · all@5 33/36 · MRR 0.898 |
| qwen3-embedding:0.6b | hit@5 94.4% · all@5 31/36 · MRR 0.880 |
| Tests | 47 passed |

### Known limitations · দুর্বলতা

- hit@5 is **document-level**: a chunk from the right file but the wrong section still counts. With
  only 50 chunks, top-5 is 10% of everything, so 100% is easier than it sounds.
  *(১২টা বইয়ের লাইব্রেরিতে ৫টা বই নামালে সঠিকটা থাকা সহজ। লাখ বইয়ের লাইব্রেরিতে কঠিন।)*
- 36 questions is a small test set: one question = 2.8 percentage points. Model gaps are 1–2 questions.
- The outdated policy can outrank the current one; retrieval alone can't fix that.
- Local mode: one process at a time (folder lock), no dashboard, exact search (no HNSW index).
- CPU only: slow indexing; every query costs ~2–3s of embedding.

### Questions I should be able to answer

<details>
<summary><b>What is an embedding, and how does vector search work?</b></summary>

**Answer:** An embedding model turns text into a fixed-length vector (1024 numbers for bge-m3) such
that texts with similar meaning get similar vectors. At index time we embed every chunk and store the
vectors in Qdrant with their metadata. At query time we embed the question and ask Qdrant for the 5
vectors with the highest cosine similarity. It matches meaning, not exact words, and works across
languages with a multilingual model.

**বুঝে নাও:** প্রতিটা টুকরাকে অর্থের মানচিত্রে একটা ঠিকানা দেওয়া। প্রশ্নেরও ঠিকানা বের করে
সবচেয়ে কাছের ৫টা টুকরা আনা, যেমন নীলক্ষেতে গিয়ে "এই ধরনের বই" চাওয়া।
</details>

<details>
<summary><b>Why cosine similarity?</b></summary>

**Answer:** It compares the direction of two vectors (the angle), not their length. Direction encodes
meaning; length tends to reflect things like text length. So a short chunk and a long chunk about the
same topic still score as similar. 1 = same direction, 0 = unrelated.

**বুঝে নাও:** দুজন একই দিকে আঙুল দেখালেই মিল, কার হাত কত লম্বা তা দেখি না।
</details>

<details>
<summary><b>Why bge-m3 and not another model?</b></summary>

**Answer:** We measured it. Three models on the same 36 eval questions: bge-m3 hit@5 100%, MRR 0.935;
nomic-embed-text 97.2%, 0.898; qwen3-embedding:0.6b 94.4%, 0.880. bge-m3 was best on every metric and
the only one with no misses. The misses were exactly the dataset's traps: nomic (mostly English) missed
the English question about the Bangla document; qwen3-embedding missed two exact-code questions. The
test set is small (gaps of 1–2 questions), so I'd re-check on a bigger set before claiming it in general.

**বুঝে নাও:** তিনজন খেলোয়াড়কে একই নেটে ব্যাট করিয়ে দেখেছি, যে সবচেয়ে ভালো খেলেছে তাকে নিয়েছি।
তবে মাত্র ৩৬ বলের হিসাব, তাই "সবসময় সেরা" বলব না।
</details>

<details>
<summary><b>Retrieval is at 100%. Why do you still need hybrid search (M3)?</b></summary>

**Answer:** Three reasons. First, hit@5 here is document-level on a tiny corpus (top-5 of 50 chunks is
10% of everything), so it's lenient. Second, it's at the ceiling, so it can't measure improvement; MRR
(0.935) and all@5 (34/36) still have room. Third, exact codes have small similarity margins, and a
different model (qwen3-embedding) already missed two of them, so vector-only is fragile. Keyword search
(BM25) handles exact codes reliably. M3 will be judged on MRR and all@5, and must not lower hit@5.

**বুঝে নাও:** পরীক্ষায় সবাই ১০০ পেলে প্রশ্ন সহজ ছিল কি না সেটাও ভাবতে হয়। আর মোবাইল নম্বরের মতো
জিনিস "শুনতে কাছাকাছি" দিয়ে খোঁজা ঝুঁকির, অক্ষরে অক্ষরে মেলানো দরকার।
</details>

<details>
<summary><b>The outdated policy scored higher than the handbook. Why, and what will you do?</b></summary>

**Answer:** The old policy says "Employees receive 15 days of annual leave per year", a full sentence
phrased like the question. The handbook has it in a terse table row. Embeddings measure how similar the
wording and topic are, not which fact is current. Both are in the top 5, so retrieval did its job. The
fix belongs in generation: chunks carry `status: superseded` and a "Status: SUPERSEDED" context line,
and the prompt (M4/M5) will tell the LLM to prefer current documents. The eval checks for 18, never 15.

**বুঝে নাও:** পুরনো নোটিশটা ভাষায় প্রশ্নের সাথে বেশি মেলে, তাই উপরে এসেছে। "বাতিল" সিল দেখে
বাদ দেওয়ার কাজ লেখকের (LLM), লাইব্রেরিয়ানের না।
</details>

<details>
<summary><b>How do you test this without running the model?</b></summary>

**Answer:** Dependency injection: `build_index` and `search` take an embed function. Unit tests pass a
fake bag-of-words embedder and an in-memory Qdrant, so they test our logic (counts, no duplicates,
stale points removed, sorting) in about two seconds. Separate integration tests call the real bge-m3,
are marked `integration`, and skip automatically when Ollama isn't running.

**বুঝে নাও:** সিমুলেটরে গাড়ি চালানো শেখা, তারপর অল্প কিছুক্ষণ আসল রাস্তায় চালিয়ে দেখা।
</details>

<details>
<summary><b>Why Qdrant local mode instead of a server?</b></summary>

**Answer:** There's no Docker on this machine and we have 50 vectors. Local mode runs Qdrant inside the
Python process and saves to a folder, like SQLite vs a Postgres server. Same client API, so switching
to a server is one setting (`QDRANT_URL`). Trade-offs: one process at a time, no dashboard, and exact
search instead of an HNSW index, which at 50 vectors is instant and actually more accurate.

**বুঝে নাও:** ৫০টা বইয়ের জন্য গুদাম ভাড়া না করে ঘরের আলমারি। বই বাড়লে শুধু ঠিকানা বদলে গুদামে যাওয়া যাবে।
</details>

<details>
<summary><b>Why do some models need "query" and "document" prefixes?</b></summary>

**Answer:** Some models (nomic-embed-text, qwen3-embedding) were trained with a marker saying whether
the text is a search query or a document, because questions and answers are phrased differently. Using
the same marker at inference matches their training; leaving it out degrades results. bge-m3 needs none.
We store prefixes per model in one dict.

**বুঝে নাও:** কিছু লাইব্রেরিয়ানকে শেখানো হয়েছে "এটা প্রশ্ন" আর "এটা বই" আলাদা করে বলে দিতে হবে।
না বললেও কাজ করেন, তবে একটু খারাপ।
</details>

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
| M2 | bge-m3 (chosen by score): hit@5 100% · all@5 34/36 · MRR 0.935 · 47 tests passing |
| M3 | — |
| M4 | — |
| M5 | — |
| M6 | Retrieval hit rate: — (target ≥ 90%) · Answer correctness: — (target ≥ 85%) |

Full history of every measured score: [experiments.md](experiments.md).
