# Chat Bot - PDF RAG

A local-first PDF RAG chatbot: upload many PDFs, preview them, ask questions
across one, several, or all of them, and get answers grounded in the retrieved
text with citations back to the file and page.

## Design goals

The system was built against five constraints, and every design decision below
is traceable to one of them:

| Goal | How it is met |
| --- | --- |
| **Fast** | No embedding model to load or run. Retrieval is pure-Python BM25 over an in-memory index — sub-millisecond on the bundled benchmark. |
| **Light** | Dependencies are FastAPI + PyMuPDF + httpx. No PyTorch, no vector DB, no model weights on disk. |
| **Accurate** | BM25 (IDF + length normalisation) instead of overlap counting, plus an evidence gate that refuses instead of guessing. |
| **Many files at once** | Batch upload (up to 20 PDFs/request, parallel extract), multi-document retrieval with a coverage guarantee, and per-question document scoping in the UI. |
| **No hallucination** | Answers are built only from retrieved chunks; a second verification pass checks the answer against those chunks; when retrieval finds nothing the assistant says so instead of inventing an answer. |

## Features

- Upload one or many PDFs; drag-free multi-select with per-file success/failure reporting
- Streaming upload with a hard size guard (`max_upload_mb`, default 50 MB)
- PDF text extraction with PyMuPDF, chunking with page metadata
- Optional vision fallback: pages with almost no extractable text are rendered
  and sent to a vision model (`enable_gemini_vision_fallback`)
- **BM25 lexical retrieval** over the chunk index — no embeddings, no ChromaDB
- Vietnamese-aware tokenisation: diacritic folding (`nhân viên` ≡ `nhan vien`)
  and English plural folding (`layers` ≡ `layer`)
- Multi-document retrieval with a **coverage guarantee**: when you select N
  documents, the top-N slots go to one best chunk per document, then remaining
  slots are filled by global score
- **Evidence gate**: retrieval returns nothing when the best chunk covers less
  than `min_query_coverage` of the question's content words, so the assistant
  refuses instead of answering from an incidental keyword match
- Follow-up rescue: conversational follow-ups ("còn cái kia thì sao?") are
  retried against the previous turn without spending an extra LLM call
- Answer verification pass that checks the answer against the retrieved context
- Chat history in SQLite, with the last `history_turns` turns fed back to the model
- React UI with chat, PDF preview, document selection, and resizable panels

## Tech stack

- **Frontend**: React, Vite, KaTeX, marked
- **Backend**: FastAPI, Uvicorn
- **Retrieval**: BM25 implemented in `app/services/vector_store.py` (pure Python, no ML)
- **LLM**: any OpenRouter model (OpenAI-compatible API), optional Gemini vision fallback
- **PDF**: PyMuPDF (text extraction + page rendering)
- **Database**: SQLite (chat history)

## Project structure

```text
backend/
  app/
    main.py                     FastAPI app, endpoints, ingest orchestration
    config.py                   Settings (env-driven, see Configuration)
    schemas.py                  Pydantic request/response models
    services/
      pdf_processor.py          PyMuPDF extraction, chunking, page rendering
      vector_store.py           BM25 index, evidence gate, multi-doc fan-out
      rag_tools.py              tool planning (search/summarize/compare/list)
      llm_client.py             OpenRouter calls, prompt building, verification
      chat_history.py           SQLite chat sessions and messages
  tests/
    test_multi_pdf_quality.py       retrieval + API quality tests
    test_multi_file_workflow.py     multi-file, gating, history, persistence tests
    eval_corpus.py                  Vietnamese benchmark corpus + labelled cases
    eval_retrieval.py               offline benchmark harness (see Verification)
  storage/                      uploads/, index/, chat_history.sqlite3 (gitignored)
frontend/
  src/main.jsx                  chat UI, multi-file upload, document selection
  src/styles.css
```

## Setup

### Backend

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Create `backend/.env`:

```env
OPENROUTER_API_KEY=your_openrouter_api_key
OPENROUTER_MODEL=google/gemini-2.5-flash-lite
```

Run it:

```powershell
uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### Frontend

```powershell
cd frontend
npm install
npm run dev
```

Open <http://127.0.0.1:3000>.

### One command for both

```powershell
powershell -ExecutionPolicy Bypass -File .\start_local.ps1
```

This checks the venv, installs frontend deps if missing, warns when `.env` has no
API key, then starts the backend and frontend and opens the browser.

## API

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Liveness probe |
| `GET` | `/health` | Config summary (upload limit, context budget, verification flag) |
| `GET` | `/api/documents` | List indexed documents |
| `GET` | `/api/documents/status` | Indexed count plus PDFs on disk that are not indexed |
| `POST` | `/api/documents/reindex` | Index every stored PDF missing from the index (idempotent) |
| `POST` | `/api/documents` | Upload one PDF |
| `POST` | `/api/documents/batch` | Upload up to 20 PDFs in one request |
| `GET` | `/api/documents/{id}/file` | Stream the stored PDF (used by the preview pane) |
| `DELETE` | `/api/documents/{id}` | Delete a document and its index entries |
| `POST` | `/api/chat` | Ask a question, optionally scoped to `document_ids` |
| `GET` | `/api/chat/sessions` | List chat sessions |
| `POST` | `/api/chat/sessions` | Create a chat session |
| `GET` | `/api/chat/sessions/{id}/messages` | List messages in a session |
| `DELETE` | `/api/chat/sessions/{id}` | Delete a session |

`POST /api/chat` request:

```json
{
  "question": "Quy trình hoàn tiền mất bao nhiêu ngày?",
  "document_ids": ["<doc-id>", "<doc-id>"],
  "session_id": "<optional>"
}
```

`document_ids` omitted or empty means **search all documents**. `document_id`
(singular) is still accepted for backward compatibility and is merged into the
scope.

Response highlights:

| Field | Meaning |
| --- | --- |
| `sources` | Cited chunks (document, page, snippet) |
| `documents_used` | Document ids that actually contributed to the answer |
| `documents_skipped` | Selected documents that produced no chunk within the context budget |
| `query_rewritten` | `true` when the follow-up rescue re-ran retrieval with prior context |
| `verification` | `supported` / `revised` / `no_evidence` / `disabled` / `skipped` |

## Configuration

All settings are environment variables (or fields in `backend/.env`).

| Variable | Default | Notes |
| --- | --- | --- |
| `OPENROUTER_API_KEY` | — | Required for LLM answers. Alias: `GEMINI_API_KEY` |
| `OPENROUTER_MODEL` | `google/gemini-2.5-flash-lite` | Alias: `GEMINI_MODEL` |
| `OPENROUTER_FALLBACK_MODEL` | `""` | Retried when the primary model errors |
| `ENABLE_GEMINI_VISION_FALLBACK` | `false` | Render + OCR low-text pages via a vision model |
| `VISION_MIN_TEXT_CHARS` | `80` | Below this many characters a page counts as low-text |
| `VISION_MAX_PAGES` | `20` | Cap on vision calls per document; `0` = no cap |
| `ENABLE_TOOL_PLANNING` | `false` | Let the LLM pick the tool; when off, keyword rules decide |
| `CHUNK_SIZE` | `1100` | Characters per chunk |
| `CHUNK_OVERLAP` | `180` | Overlap between adjacent chunks |
| `TOP_K` | `6` | Retrieved chunks per question |
| `MAX_CONTEXT_CHUNKS` | `24` | Hard cap on chunks entering the LLM context in one turn |
| `MAX_UPLOAD_MB` | `50` | Reject larger single PDFs with HTTP 413 |
| `MIN_QUERY_COVERAGE` | `0.25` | Evidence gate threshold; `0` disables the gate |
| `EVIDENCE_METRIC` | `coverage` | `coverage` or `idf_coverage` |
| `INGEST_WORKERS` | `4` | Parallel PDF extract/chunk workers during batch upload |
| `HISTORY_TURNS` | `4` | Previous turns handed to the LLM |
| `ENABLE_ANSWER_VERIFICATION` | `true` | Second pass that validates the answer against the context |
| `STORAGE_DIR` | `backend/storage` | Root for uploads, index, and chat DB (`/tmp` on Vercel) |
| `BACKEND_CORS_ORIGINS` | localhost + GitHub Pages | Comma-separated allowed origins |

## How retrieval works

1. **Ingest** — each page's text is chunked, and every chunk is stored with
   `document_id`, `filename`, `page`, and `chunk_index`. Term frequencies are
   cached per chunk and document frequencies are maintained incrementally, so
   adding a document never requires re-scoring the corpus.
2. **Tokenise** — text is lowercased, Vietnamese diacritics are folded
   (`NFD` decomposition plus an explicit `đ → d` mapping), trailing English
   plurals are stripped, and stopwords are removed for coverage maths.
3. **Score** — BM25 with `k1 = 1.5`, `b = 0.75`:
   `score = Σ IDF(term) · tf·(k1+1) / (tf + k1·(1 − b + b·len/avgdl))`
4. **Gate** — if the best chunk covers less than `min_query_coverage` of the
   question's content words, retrieval returns nothing. Queries that contain an
   identifier-like token (`HR-01`, `ISO 27001`) bypass the gate, because
   identifiers are evidence by themselves.
5. **Fan out (multi-document)** — for each selected document, take its best
   chunk; sort those by score; then fill the remaining budget with the next-best
   chunks across all documents. Scores are normalised **once, after merging** —
   normalising per document would make every document's best chunk tie at 1.0.
6. **Answer** — the retrieved chunks go into the prompt, the model answers from
   them only, and an optional verification pass checks the answer against the
   same chunks. If retrieval came back empty, the assistant replies
   `Chưa tìm thấy nội dung liên quan trong tài liệu.`

## Recovering an existing library

PDFs are stored as `backend/storage/uploads/<document_id>.pdf` and the BM25 index
lives in `backend/storage/index/`. The two can drift apart — most notably when
upgrading from an older build that used a vector database, which leaves every
PDF on disk but nothing in the index. In that state the UI shows an empty
library even though the files are still there.

The app detects this: `GET /api/documents/status` compares the two, and the UI
shows an **Index N file còn thiếu** banner with a one-click repair.

From the command line:

```powershell
cd backend
.\.venv\Scripts\python.exe rebuild_index.py --check   # report only
.\.venv\Scripts\python.exe rebuild_index.py           # index what is missing
.\.venv\Scripts\python.exe rebuild_index.py --force   # wipe and re-index everything
```

`--force` also rebuilds documents whose chunking changed, for example after
editing `CHUNK_SIZE`. Re-indexing **never deletes a stored PDF** — a failed
re-index leaves the file untouched so it can be retried.

Re-indexing skips the vision fallback by default (pass `--vision` to enable it).
Repairing an index should be a fast, offline operation, not hundreds of calls to
a vision model — see the next section.

### Repairing stored text without re-reading any PDF

Older builds could persist a vision model's safety verdict as page content, so a
stray `"User Safety: safe"` line became a retrievable chunk that the assistant
then quoted back as if it came from your document. Ingest no longer does this,
but text already in the index stays wrong until it is cleaned:

```powershell
.\.venv\Scripts\python.exe rebuild_index.py --clean-text            # dry run
.\.venv\Scripts\python.exe rebuild_index.py --clean-text --apply    # write, keeps a .bak
```

This edits the persisted JSON directly rather than re-reading PDFs, so it is
instant and cannot lose anything: it only removes lines that are provably model
bookkeeping. The BM25 caches are rebuilt from the cleaned text on the next load.

## Where the time actually goes

Retrieval is not the bottleneck; **PDF ingest is**. Measured on this project's
own library (4 PDFs, 78 MB, 1514 pages):

| Stage | Cost |
| --- | --- |
| Text extraction (PyMuPDF) | 4.8 s for 682 pages |
| Chunking | < 0.1 s |
| BM25 indexing | 0.3 s for 1251 chunks |
| **Vision fallback** | **~15 s per low-text page, sequential** |

The vision fallback is the only stage whose cost is unbounded, because each page
is a separate network round trip. A 682-page textbook had 78 figure pages; with
vision enabled, indexing the whole library made **112 sequential calls and ran
for over 13 minutes with no output**, which is indistinguishable from a hang.
That is why `VISION_MAX_PAGES` exists (default 20, with a warning logged when it
truncates) and why re-indexing leaves vision off.

The same library now indexes in **11.5 s** with vision off. Enable it only for
documents that are genuinely scans — a 5-page scanned exam takes 74 s to read as
images, and is the case the feature exists for.

`get_text(..., sort=True)` is also worth knowing about: it costs ~14 ms/page
versus ~2 ms/page unsorted. It is kept because it fixes reading order on
multi-column layouts, which matters far more for retrieval quality than the ~3%
of ingest time it adds.

## Verification

The project ships an offline benchmark instead of asking you to trust the
scoring. `tests/eval_corpus.py` holds six Vietnamese business documents that
deliberately share vocabulary (approval workflows, leave policy, refunds,
security, software guide, incident handling) and 32 labelled cases split across
`factual`, `identifier`, `unaccented`, `paraphrase`, `compare`, and `refusal`.

Run it:

```powershell
cd backend
.\.venv\Scripts\python.exe -m tests.eval_retrieval                 # full report
.\.venv\Scripts\python.exe -m tests.eval_retrieval --sweep-coverage # tune the gate
.\.venv\Scripts\python.exe -m tests.eval_retrieval --scale-bench   # index size vs latency
.\.venv\Scripts\python.exe -m tests.eval_retrieval --analyze-gate  # why presence gating fails
.\.venv\Scripts\python.exe -m tests.eval_retrieval --end-to-end    # answer-level refusal (needs API key)
.\.venv\Scripts\python.exe -m tests.eval_retrieval --strict        # exit 1 on any miss
```

When a question comes back refused and you need to know **who** refused, use the
diagnostic. It wraps the model client, logs every prompt and response, and
reports whether the draft itself refused (a model limit) or a good draft was
destroyed by the verification pass (a pipeline bug) — a distinction that the
refusal *rate* alone cannot make:

```powershell
.\.venv\Scripts\python.exe -u tests/diag_verifier.py
```

Running it settled the question here: every false refusal was the draft itself,
zero were the verifier, so the fix is a stronger model rather than more
threshold tuning.

### Measured results

Default configuration (`min_query_coverage=0.25`, `evidence_metric=coverage`),
32 cases:

```text
top1_accuracy        0.957
recall@k             1.000
mrr                  0.978
compare_coverage     1.000
citation_rate        1.000
refusal_recall       0.333
false_refusal_rate   0.000
latency p50          0.9 ms
latency p95          2.2 ms
```

Per question type:

```text
kind         cases   top1   keyword   refusal
compare          3  1.000         -         -
factual         10  1.000     1.000         -
identifier       4  1.000     1.000         -
paraphrase       5  0.800     0.800         -
refusal          6      -         -     0.333
unaccented       4  1.000     1.000         -
```

**The `identifier` row does not test the identifier escape.** All four of those
cases reach coverage 0.59–0.73 against a 0.25 threshold, so the gate never
blocks them and the escape's return value is irrelevant. Two real bugs lived in
that escape (hyphenated codes like `MUA-07` never matched; only the top-ranked
chunk was inspected) while this row reported a clean 1.000 the whole time.

No question built from this corpus can trip the gate, and that is structural
rather than accidental: a query that is just a code (`HR-01`) is inherently
high-coverage, because the code *is* a content word and the matching chunk
contains it. The escape is therefore covered by a unit test —
`test_identifier_escape_scans_past_the_top_chunk`, which sets
`min_query_coverage=0.9` to force the gate to matter. The benchmark now prints a
note on every run saying so, rather than letting the row look like coverage.

**Read the refusal numbers carefully — they measure two different layers.**

`refusal_recall 0.333` above is the **evidence gate**: the share of off-topic
questions for which *retrieval* returned nothing. It is not the property users
care about. "Does the assistant make things up?" is decided by the gate *and* the
verification pass together, so it has to be measured on the final answer:

```powershell
.\.venv\Scripts\python.exe -m tests.eval_retrieval --end-to-end
```

Measured end to end (retrieval → draft → verify) with `liquid/lfm-2.5-2.6b:free`,
all 32 cases:

```text
refusal_correct      1.000   (6/6 off-topic questions refused)
false_refusal        0.115   (3/26 answerable questions refused)
```

Six of six off-topic questions end in a correct refusal — for four of them
retrieval *did* return chunks, and the model read them and answered "Tài liệu
không cung cấp thông tin...". So the gate is the cheap first line of defence, not
the last one, and a low gate number does not mean the assistant hallucinates.

The 4 false refusals are the honest cost, and they are not random — they
concentrate in the kinds that need reasoning rather than lookup:

```text
kind          cases   wrongly refused
factual          10   0
unaccented        4   0
identifier        4   1   (25 %)
compare           3   0   (0 %)
paraphrase        5   2   (40 %)
```

A direct question ("Nhân viên được bao nhiêu ngày phép?") never trips it. An
indirect one ("Tôi bị cảm nhẹ thì cần làm thủ tục gì để được nghỉ?") sometimes
does, because connecting a paraphrase to the source text is exactly what a 2.6B
model is weakest at. That is a model-capability limit, not a pipeline bug — the
right fix is a stronger model, not more threshold tuning, which would trade these
3 false refusals for missed real answers.

(Re-run on 2026-09-21 with the same model brought the rate to 3/26: the `compare`
case that previously refused by mistake answered correctly this run — model
variance, not a code change. All 3 remaining false refusals are still `no_evidence`
labels, i.e. the *draft* refused and the verifier correctly caught it, so none are
pipeline bugs.)

Three caveats, stated rather than buried:

- **n=6.** Six off-topic cases is enough to catch a broken system, not enough to
  claim a precise rate.
- **The verifier's label was unreliable even when its outcome was right.** On 3 of
  those 6 questions the model returned `is_supported: true` for an answer that
  said the documents did not cover the question. The answer was correct; the
  label contradicted it. `verify_answer` now overrides that case with
  `no_evidence`, because the label is shown to the user.
- **Measuring this is easy to get wrong, and I got it wrong twice.** First I
  counted only `unsupported` labels and missed refusals the verifier had rewritten
  (`revised`). Then I matched refusal phrases *anywhere* in the answer, which
  flagged three long comparison answers that merely noted a gap mid-text. The
  predicate now lives in `app.services.llm_client.looks_like_refusal`, checks only
  the first 120 characters, and is imported by the benchmark so the two cannot
  drift apart.

**A gate idea that did not work.** Gating on corpus-level vocabulary presence —
"the question names things the corpus has never heard of, so refuse" — sounds
right and fails. A legitimate paraphrase ("Tôi bị cảm nhẹ thì cần làm thủ tục gì
để được nghỉ?", presence 0.59) looks exactly like an off-topic question ("Công ty
có bán cà phê rang xay không?", presence 0.73). The distributions overlap across
0.54–0.73, so no threshold separates them:

```powershell
.\.venv\Scripts\python.exe -m tests.eval_retrieval --analyze-gate
```

```text
refusal      n=6   presence min=0.35  max=0.73
paraphrase   n=5   presence min=0.54  max=0.88   <- overlaps refusal entirely
factual     n=10   presence min=0.88  max=0.92
```

Caveat: this corpus is small (24 chunks, ~580 terms), so "thủ tục" is missing as
an artifact of size, not because the topic is out of domain. Presence gating
might separate cleanly on thousands of documents — but that is an assumption, and
this benchmark cannot confirm it.

### Threshold trade-off

Chosen with `--sweep-coverage` rather than by guessing:

```text
metric         min_cov   top1   recall   refusal   false_refusal   p50 ms
coverage          0.25  0.957    1.000     0.333           0.000      1.2
coverage          0.30  0.870    0.913     0.333           0.077      1.0
idf_coverage      0.20  0.870    0.913     0.667           0.077     13.9
idf_coverage      0.30  0.826    0.826     1.000           0.154     13.9
```

`coverage` at `0.25` was chosen because it lifts refusal recall from 0.00 to
0.33 at **zero** cost to answerable recall. `idf_coverage` refuses more
aggressively but trades recall (1.00 → 0.87) and is an order of magnitude
slower, so it is available as an opt-in rather than the default.

**Known weakness, stated plainly:** the *retrieval gate* refuses only 33 % of
off-topic questions, because four of the six share words like *Công ty* or
*Chính sách* with the corpus. Raising the threshold or switching metrics costs
real answers, so the cheap gate is kept and the gap is documented instead of
hidden. The end-to-end refusal rate is 100 % because the verification pass
catches what the gate misses — but that costs an extra LLM call, so the gate
still matters for latency and cost.

### Scale

`--scale-bench` replicates a real 15.4 MB, 718-page PDF (1538 chunks per copy)
and measures ingest plus query latency:

```text
copies   chunks   peak RAM   p50      p95      max
     1     1538     17.1 MB   12.7 ms  24.0 ms  24.4 ms
     5     7690     84.5 MB   30.2 ms  49.7 ms  72.2 ms
    10    15380    168.9 MB   34.0 ms  78.7 ms  83.9 ms
    20    30760    337.7 MB   72.4 ms 160.3 ms 164.9 ms
```

Worst case here is a *global* search across all 30 760 chunks — scoping the
question to a subset cuts latency proportionally.

### Live end-to-end check

Green tests do not prove the program runs. This does — a real `uvicorn` server
against the project's own PDF library (4 documents, 1 514 pages, 3 111 chunks):

```text
$ curl http://127.0.0.1:8011/health
{"status":"ok","max_upload_mb":50,"max_context_chunks":24,"answer_verification":true}

$ curl http://127.0.0.1:8011/api/documents/status
{"indexed":4,"unindexed":[],"unindexed_count":0}
```

A question spanning **two different documents** — `"What is gradient descent and
how does it work?"`:

```text
verification:  supported: Câu trả lời được hỗ trợ bởi CONTEXT...
docs_used:     ['95f72d89...', 'b27d26aa...']    <- two distinct documents
docs_skipped:  []
n_sources:     2
pages:         Hands-On ML... p.327  +  Hands-On ML... p.506
```

Two separate PDFs contributed evidence to one answer, and the answer cited both
page numbers. That is multi-file fan-out working in the real app, not just in a
fixture.

Scope restriction — asking about ML while limited to the calculus exam:

```text
POST /api/chat  {"question":"What is gradient descent?",
                 "document_ids":["c04b8dec..."]}

verification:  skipped
docs_used:     []
n_sources:     0
answer:        "Chưa tìm thấy nội dung liên quan trong tài liệu."
```

Correct behaviour: the scope is honoured, and when the in-scope documents do not
contain the answer the system refuses rather than reaching for out-of-scope
material.

### Two more latent bugs fixed this session (2026-09-21)

Both were found by reading the code and confirmed with regression tests — they
never surfaced in the offline benchmark because the benchmark's corpus is small.

- **Evidence loss in `dedupe_sources` (Bug 10).** The dedupe key used
  `preview_page` instead of the true chunk `page`. For procedure questions
  `context_start_page()` rewinds `preview_page` to the start of the section, so two
  genuinely different chunks (e.g. step 3 on page 10 and step 7 on page 14) could
  both be rewound to `preview_page=10` and collapse into one entry, silently
  dropping real evidence. Fixed by keying on `(document_id, page, hash(text))`; a
  long page split into several chunks still stays distinct because its text
  differs. Regression test: `TestDedupeKeepsDistinctProcedureChunks`.
- **`DELETE /api/documents/{id}` could 500 on Windows (Bug 11).** The endpoint
  used a bare `pdf_path.unlink()`. A PDF still open in a viewer raises
  `PermissionError`, which turned a successful vector delete into a 500. Routed
  through the existing `_safe_unlink` (already used elsewhere for exactly this on
  Windows); a locked file is now left on disk for the user to retry instead of
  taking down the request.

### Test suite

```powershell
cd backend
.\.venv\Scripts\python.exe -m pytest tests -q
```

92 tests across two files. Last full run: **92 passed**.

One earlier run showed `test_oversized_upload_is_rejected` failing. That was the
agent sandbox, not the app: its bulk-delete guard raises `SystemExit` from inside
`path.unlink`, which `except OSError` cannot catch. The guard is cumulative and
turn-scoped, so it fires intermittently — which is why the same suite passed
cleanly on the next run. The test passes in a normal terminal.

`tests/test_multi_file_workflow.py` covers:

- **fan-out coverage** — more documents than the context budget, relevance order
  beating caller order, and near-duplicates not being misreported as skipped
- **the evidence gate** — refusal on off-topic questions, identifiers bypassing
  the gate, the gate being disableable, hyphenated codes like `MUA-07` being
  recognised, and the identifier escape scanning past the top-ranked chunk
- **token normalisation** — unaccented Vietnamese queries matching accented text,
  English plurals matching singulars
- **batched persistence** — `save=False` + `flush`, and deleting a document
  actually clearing it from the index
- **multi-file upload over the real API** — batch indexing, partial failures,
  oversized uploads rejected with 413, non-PDF rejection
- **document scoping** — answers confined to the selected documents, and
  documents that could not fit the budget being reported
- **conversation history** — history reaching the LLM, and the follow-up rescue
  ladder
- **index reconciliation** — orphan PDFs detected and indexed, idempotency,
  corrupt PDFs surviving without being deleted
- **vision budget** — the per-document cap actually bounding network calls, and
  re-indexing spending none
- **refusal labelling** — a model calling its own refusal `supported` being
  overridden, and a gap noted mid-answer *not* being counted as a refusal
- **safety preambles never reaching the user** — a preamble-only response
  becoming a real message instead of being echoed back, and a preamble-only
  rewrite being treated as no rewrite

`tests/test_multi_pdf_quality.py` covers end-to-end retrieval quality on a real
PDF (`precision(top1)=1.00`, `citation_valid=1.00`).

The frontend production build is also verified: `cd frontend && npm run build`
→ `✓ built in 792ms`.

## Deployment

- **Render** (`render.yaml`): `pip install -r backend/requirements.txt`, then
  `cd backend && uvicorn app.main:app --host 0.0.0.0 --port $PORT`. Set
  `OPENROUTER_API_KEY` and `BACKEND_CORS_ORIGINS` in the dashboard.
- **Vercel** (`vercel.json` + `api/index.py`): functions have a 60 s budget and
  the storage root moves to `/tmp` automatically when `VERCEL` is set. The
  `/tmp` filesystem is ephemeral, so on Vercel the index and chat history do not
  survive a cold start — use Render (or any host with a persistent disk) if you
  need durable documents.

## Notes

- Uploaded PDFs, the BM25 index, and chat history all live under
  `backend/storage/`. Deleting that directory resets the system.
- There is no embedding model to download: the first run is ready immediately.
- `numpy` remains a dependency of the PDF/vision path, not of retrieval.
