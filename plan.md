# Build Plan — Agentic RAG with Hybrid Retrieval, Reranking & Abstention

**Author:** Ninad Anil Kawade
**Created:** 10 Nov 2026
**Target:** Working, deployed, evaluated project on CV before **22 Nov 2026** (Thomson Reuters deadline)
**Working name:** `evidence-rag` (rename to something domain-specific once you pick a corpus — e.g. `lexquery` for legal, `filingsqa` for SEC)

---

## 1. What we are building (and why it's not generic)

A question-answering system over a document corpus that a reviewer can **click and use live**. It is *not* "chat with my PDFs." The difference is four deliberate engineering choices, each of which is a hiring signal:

| Feature | What it does | Why it matters |
|---|---|---|
| **Hybrid retrieval** | Combines keyword search (BM25) + semantic search (vector embeddings) via Reciprocal Rank Fusion | Pure vector search misses exact terms (names, codes); pure keyword misses paraphrase. Hybrid is what production systems actually use. |
| **Cross-encoder reranking** | Re-scores the top candidates with a more accurate (slower) model | Shows you understand the retrieve-then-rerank two-stage pattern, not just top-k cosine. |
| **Self-grading loop (Corrective RAG)** | Grades its own retrieved docs; if they're weak, rewrites the query and retries | This is the "agentic" part — decision-making over tools, not a single prompt. |
| **Abstention** | Says *"I don't have enough evidence to answer"* instead of hallucinating | Maturity signal hiring managers explicitly call out. A system that knows what it doesn't know. |

Plus the thing most ML students skip and you can do easily: **it ships as a live FastAPI service with a frontend and an evaluation harness reporting real numbers.**

### The one-sentence pitch (for your README and CV)
> An agentic RAG system over [corpus] using hybrid retrieval + cross-encoder reranking + a self-correcting retrieval loop, with evidence-based abstention, served via FastAPI and evaluated with RAGAS (faithfulness 0.XX across 50 questions).

---

## 2. Pick your corpus first (decide before you write any code)

The corpus determines how much the project resonates with each employer. Pick **one** you can speak about in an interview. Options, best-fit first:

1. **Legal / regulatory documents** (e.g. UK legislation from legislation.gov.uk, or EUR-Lex) → best fit for **Thomson Reuters** (Legal is their core segment). Strong, differentiated, and the domain is "information retrieval over law" — exactly their Labs team's work.
2. **SEC 10-K filings** (via SEC EDGAR API, free) → best fit for **finance roles** (Wise, Monzo, HRT, Moody's, JPMorgan). Rich, structured, real-world messy.
3. **arXiv ML papers** (via arXiv API, free) → neutral, easy to get, good if you want a research flavour (G-Research, Amazon).

**Recommendation:** Go with **(1) legal/regulatory** because your nearest deadline (Thomson Reuters, 22 Nov) is the strongest-fit LLM role and a legal corpus speaks directly to it. ~200–500 documents is plenty.

> **Decision to record in DECISIONS.md:** which corpus, why, how many docs, where you got them.

---

## 3. Architecture

```
                    ┌─────────────────────────────────────────────┐
   User question →  │  FastAPI  /ask                              │
                    └───────────────────┬─────────────────────────┘
                                        ▼
                         ┌──────────────────────────────┐
                         │   LangGraph agent (the loop)  │
                         └──────────────────────────────┘
                                        │
      ┌─────────────────────────────────┼───────────────────────────────┐
      ▼                                 ▼                                 ▼
┌───────────┐   hybrid        ┌──────────────────┐   grade      ┌──────────────────┐
│ RETRIEVE  │ ──────────────▶ │ RERANK           │ ───────────▶ │ GRADE DOCS        │
│ BM25 +    │   RRF fusion    │ cross-encoder    │   top-k      │ relevant enough?  │
│ vector    │                 │                  │              └─────────┬────────┘
└───────────┘                 └──────────────────┘                        │
      ▲                                                     yes ┌──────────┴────────┐ no
      │                                                         ▼                   ▼
      │                                               ┌──────────────┐    ┌──────────────────┐
      │                                               │ GENERATE     │    │ REWRITE QUERY     │
      │                                               │ answer +     │    │ + retry (max 2)   │
      │                                               │ citations    │    └─────────┬────────┘
      │                                               └──────┬───────┘              │
      └──────────────────────────────────────────────────────────────────────────┘
                                                              │ still weak after retries
                                                              ▼
                                                      ┌──────────────┐
                                                      │  ABSTAIN     │
                                                      │ "insufficient│
                                                      │  evidence"   │
                                                      └──────────────┘
```

**Ingestion (offline, run once):** load docs → chunk → embed → store in ChromaDB + build BM25 index.

---

## 4. Tech stack (exact choices)

| Layer | Choice | Notes |
|---|---|---|
| Language | Python 3.11+ | |
| Agent orchestration | **LangGraph** | The graph = the diagram above. This is the "framework you built around." |
| Vector DB | **ChromaDB** | Local, zero-config, persists to disk. |
| Keyword search | **rank_bm25** | Simple BM25 implementation. |
| Embeddings | **sentence-transformers** `BAAI/bge-small-en-v1.5` | Free, local, good quality. |
| Reranker | **sentence-transformers** `cross-encoder/ms-marco-MiniLM-L-6-v2` | Free, local. |
| LLM | **OpenAI `gpt-4o-mini`** (cheap) or **Groq** free tier (Llama 3.x) | gpt-4o-mini is ~pennies for this whole project. Groq is free if you want zero cost. |
| Structured output | **Pydantic** | Answer + citations as a validated schema. |
| Evaluation | **RAGAS** | context precision, faithfulness, answer relevancy. |
| API | **FastAPI + uvicorn** | Your existing strength — lean on it. |
| Frontend | **Streamlit** (fast) or minimal **React** | Streamlit if time-pressed; React if you want to show frontend skill. |
| Packaging | **Docker + docker-compose** | Production signal. |
| CI (optional layer) | **GitHub Actions** | Runs RAGAS eval, fails PR on metric drop. |
| Deploy | **Hugging Face Spaces** or **Render** free tier | Gives you the live URL. |

> **Cost note:** with gpt-4o-mini and local embeddings/reranker, the entire project (including eval runs) costs well under £5. Groq's free tier makes it £0.

---

## 5. Repository structure

```
evidence-rag/
├── README.md                 # problem → live demo → result → how to run (first 200 words matter)
├── DECISIONS.md              # why each choice, what you gave up, known failure modes
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
├── .github/workflows/eval.yml   # optional CI layer
├── data/
│   └── raw/                  # downloaded corpus
├── ingest/
│   ├── load.py               # load + parse documents
│   ├── chunk.py              # chunking with overlap
│   └── build_index.py        # embed → ChromaDB + BM25 index
├── app/
│   ├── retrieval.py          # hybrid retrieval + RRF
│   ├── rerank.py             # cross-encoder reranker
│   ├── graph.py              # LangGraph agent (retrieve→rerank→grade→generate/retry/abstain)
│   ├── schemas.py            # Pydantic models (Answer, Citation)
│   └── main.py               # FastAPI app (/ask, /health, /metrics)
├── frontend/
│   └── app.py                # Streamlit (or a React app/ folder)
├── eval/
│   ├── testset.jsonl         # 50 curated questions + expected answers
│   ├── run_eval.py           # RAGAS metrics
│   └── results.md            # the numbers + failure analysis
└── demo.gif                  # short screen recording
```

---

## 6. Phase-by-phase plan (scoped for ~2 weekends + weekday evenings)

Each phase has an **acceptance criterion** — do not move on until it's met.

### Phase 0 — Setup (Mon/Tue eve, ~2h)
- Create repo, Python venv, install dependencies.
- Pick corpus, download ~200–500 docs into `data/raw/`.
- Commit the skeleton structure.
- **Acceptance:** repo runs, dependencies install, corpus is on disk.

### Phase 1 — Ingestion + baseline retrieval (Weekend 1, Sat, ~6h)
- `load.py`: parse docs to plain text + metadata (source, title, page/section).
- `chunk.py`: split into ~800-char chunks with ~150-char overlap, keep metadata on each chunk.
- `build_index.py`: embed chunks with bge-small → ChromaDB; build a BM25 index over the same chunks.
- Write a quick script to query both and eyeball results.
- **Acceptance:** you can ask a question and get back top-k chunks from *both* vector and BM25 search.

### Phase 2 — Hybrid retrieval + reranking (Weekend 1, Sun, ~6h)
- `retrieval.py`: run BM25 + vector in parallel, fuse with **Reciprocal Rank Fusion** (RRF):
  `score(d) = Σ 1 / (k + rank_i(d))` with `k=60`. Return top ~20.
- `rerank.py`: cross-encoder re-scores those 20 against the query; keep top ~5.
- Compare hybrid+rerank vs. naive vector on 5–10 questions. Note where it's better.
- **Acceptance:** hybrid+rerank visibly returns better top-5 than naive vector on your test questions (write down an example).

### Phase 3 — The agentic loop (weekday eves, ~5h)
- `schemas.py`: Pydantic `Answer { text, citations: list[Citation], confidence }`, `Citation { source, snippet }`.
- `graph.py` with LangGraph nodes:
  - **retrieve** → **rerank** → **grade_docs** (LLM grades each top doc: relevant? yes/no)
  - if enough relevant docs → **generate** (answer strictly from context, with citations)
  - if not → **rewrite_query** → retry retrieve (max 2 retries)
  - if still weak after retries → **abstain** (return "insufficient evidence" with what it searched)
- Key rule in the generation prompt: *"Answer only from the provided context. If the context does not contain the answer, say you cannot answer from the available documents."*
- **Acceptance:** (a) a normal question returns an answer with correct citations; (b) an out-of-corpus question (e.g. "what's the weather today?") triggers abstention, not a hallucination.

### Phase 4 — Serve it (weekday eve, ~3h)
- `main.py`: FastAPI with:
  - `POST /ask` → `{answer, citations, confidence, abstained: bool}`
  - `GET /health` → status
  - `GET /metrics` → request count, avg latency, abstention rate
  - input validation (empty/over-long query → safe 4xx, not a crash)
- `frontend/app.py`: Streamlit box → question in, answer + citations + "abstained" badge out.
- **Acceptance:** you can use the whole thing from a browser locally.

### Phase 5 — Evaluation harness (Weekend 2, Sat, ~6h) ← the differentiator
- Build `eval/testset.jsonl`: **50 questions** you curate by hand. Include deliberately:
  - ~35 answerable questions (with a reference answer you write)
  - ~10 out-of-corpus questions (correct behaviour = abstain)
  - ~5 edge cases (answer spans multiple documents, ambiguous phrasing)
  - State *why* you chose them (edge + regression cases) — this itself is a signal.
- `run_eval.py` with **RAGAS**: context precision, faithfulness, answer relevancy on the answerable set; abstention accuracy on the out-of-corpus set.
- `results.md`: the numbers + a **failure analysis** — list the 5 ways it still breaks (e.g. "fails on multi-hop questions needing 3+ chunks").
- **Acceptance:** you have real metrics and a written, honest list of failure modes.

### Phase 6 — Package, deploy, document (Weekend 2, Sun, ~4h)
- `Dockerfile` (python:3.12-slim + uvicorn) and `docker-compose.yml`.
- Deploy to Hugging Face Spaces or Render → **get the live URL**.
- Write `README.md` (problem → live link → result number → how to run, in that order).
- Write `DECISIONS.md` (corpus choice, chunk size, model, what you gave up).
- Record a ~60s `demo.gif`.
- **Acceptance:** a stranger can click your live URL and use it; README states the problem and result in the first 200 words.

### Phase 7 (optional, if time) — CI eval gate
- `.github/workflows/eval.yml`: on push, run a small RAGAS suite; fail the build if faithfulness drops below a threshold.
- This upgrades the project toward **Playbook Project 1** (the rarest signal). Only do it once everything above works.

---

## 7. If you run short on time (scope cuts, in order)

Cut from the bottom up — protect the core + eval:
1. Drop the GitHub Actions CI (Phase 7). *(First to go.)*
2. Use Streamlit instead of React for the frontend.
3. Skip deployment; provide a **one-command local run** (`docker compose up`) + the demo.gif instead of a live URL.
4. Shrink the test set to 30 questions (still report metrics — never cut the eval entirely; it's the differentiator).

**Never cut:** hybrid retrieval, abstention, and the evaluation. Those three are the whole reason the project stands out.

---

## 8. What "done" looks like (definition of done)

- [ ] Live URL (or one-command run) a stranger can use
- [ ] Hybrid retrieval + cross-encoder reranking working
- [ ] Self-correcting retrieval loop (query rewrite + retry)
- [ ] Abstains on weak evidence instead of hallucinating
- [ ] Answers carry citations to source documents
- [ ] RAGAS metrics reported on a 50-question curated test set
- [ ] `results.md` with honest failure analysis (the 5 ways it breaks)
- [ ] README leads with problem → demo → result; DECISIONS.md explains tradeoffs
- [ ] 60-second demo.gif

---

## 9. CV bullet (fill in your real numbers)

> **Agentic RAG system over [legal/SEC] documents** — Built a retrieval-augmented QA system with hybrid search (BM25 + dense vectors, RRF fusion), cross-encoder reranking, and a self-correcting retrieval loop (LangGraph); added evidence-based abstention to prevent hallucination. Served via FastAPI with a live demo. Evaluated with RAGAS across 50 curated questions: faithfulness 0.XX, context precision 0.XX, 9X% correct abstention on out-of-corpus queries. *(Python, LangGraph, ChromaDB, sentence-transformers, RAGAS, FastAPI, Docker)*

---

## 10. Reference links

- awesome-llm-apps (CRAG, Agentic Typed RAG patterns): https://github.com/Shubhamsaboo/awesome-llm-apps
- RAGAS (evaluation): https://github.com/explodinggradients/ragas
- LangGraph docs: https://github.com/langchain-ai/langgraph
- DeepEval (if you prefer it for the CI layer): https://github.com/confident-ai/deepeval
- SEC EDGAR API: https://www.sec.gov/edgar/sec-api-documentation
- UK legislation data: https://www.legislation.gov.uk/developer
- Production ML checklist: https://datatalksclub.github.io/podwiki/wiki/production-ml-project-checklist/

---

*Build order summary: corpus → ingest → hybrid+rerank → agentic loop → serve → **evaluate** → deploy → document. Protect the eval and abstention above all else — they are what make this read as real engineering, not a tutorial.*
