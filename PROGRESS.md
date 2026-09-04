# RAG Project — Progress Tracker

> Last updated: 2026-09-02

---

## Completed Phases

### Phase 1 — Project Skeleton & Document Loader ✅

**What we built:**
- Project folder structure (`src/`, `data/`)
- A document loader that reads PDF and `.txt` files and returns a uniform `{"source", "page", "text"}` dict per page
- Config files: `.env` (API key placeholder), `.gitignore`, `requirements.txt`
- Sample test document (`data/sample_ml.txt`)

**Key concept:** Uniform data contract — both PDF and TXT loaders output the same dict shape so all downstream phases are format-agnostic.

---

### Phase 2 — Chunking & Embeddings ✅

**What we built:**
- Token-based chunker that splits page text into 400-token windows with 50-token overlap
- Embedding module that converts text chunks into 384-dimensional vectors using `all-MiniLM-L6-v2`
- Demo script that prints chunk details, a real embedding vector, and a 3-sentence similarity comparison

**Key concept:** Embeddings encode *meaning* as numbers. Texts with similar meaning produce vectors with high dot-product similarity; unrelated texts produce near-zero similarity.

**Dependency fix during this phase:** `tiktoken` has no pre-built wheel for Python 3.14 (requires Rust compiler). Replaced with `transformers.AutoTokenizer("bert-base-uncased")`, which is already pulled in by `sentence-transformers` and uses the same tokenizer family as our embedding model.

---

## Files in the Project

| File | Phase | Purpose |
|---|---|---|
| [`src/loader.py`](file:///c:/Users/bhagy/OneDrive/Desktop/RAG/src/loader.py) | 1 | Loads PDF/TXT files, returns list of page dicts |
| [`src/chunker.py`](file:///c:/Users/bhagy/OneDrive/Desktop/RAG/src/chunker.py) | 2 | Splits page text into overlapping token-window chunks |
| [`src/embedder.py`](file:///c:/Users/bhagy/OneDrive/Desktop/RAG/src/embedder.py) | 2 | Embeds chunks/queries into 384-d vectors with MiniLM |
| [`src/phase2_demo.py`](file:///c:/Users/bhagy/OneDrive/Desktop/RAG/src/phase2_demo.py) | 2 | End-to-end demo: load -> chunk -> embed -> similarity |
| [`data/sample_ml.txt`](file:///c:/Users/bhagy/OneDrive/Desktop/RAG/data/sample_ml.txt) | 1 | Sample document about machine learning (for testing) |
| [`requirements.txt`](file:///c:/Users/bhagy/OneDrive/Desktop/RAG/requirements.txt) | 1 | All project dependencies with comments |
| [`.env`](file:///c:/Users/bhagy/OneDrive/Desktop/RAG/.env) | 1 | Anthropic API key (placeholder, not committed) |
| [`.gitignore`](file:///c:/Users/bhagy/OneDrive/Desktop/RAG/.gitignore) | 1 | Ignores `.env`, `__pycache__/`, `chroma_db/` |

---

## Installed Dependencies

**Python version:** 3.14.7 (via `uv` venv at `.venv/`)

| Package | Version | Why |
|---|---|---|
| `pypdf` | 6.16.2 | PDF text extraction (page-by-page) |
| `sentence-transformers` | 3.0.1 | Loads all-MiniLM-L6-v2 embedding model |
| `transformers` | 4.57.6 | AutoTokenizer for token counting (replaces tiktoken) |
| `torch` | 2.13.0 | PyTorch backend for sentence-transformers |
| `numpy` | 2.5.2 | Array operations for embeddings |
| `python-dotenv` | 1.0.1 | Loads `.env` file for API keys |

*(Plus transitive deps: `tokenizers`, `huggingface-hub`, `scikit-learn`, `scipy`, `tqdm`, etc.)*

**Not installed (yet, needed for later phases):**
- `chromadb` — Phase 3 (vector store)
- `anthropic` — Phase 4 (Claude API)
- `streamlit` — Phase 7 (UI)

---

## Next Step: Phase 3 — Vector Store

**What we'll build:**
1. Store chunks + embeddings in ChromaDB (a local embedded vector database)
2. Write a `query()` function that takes a question string, embeds it, and returns the top-k most similar chunks
3. Run a test query against our sample document and print the results

**Files to create/modify:**
- `src/vector_store.py` — NEW: ChromaDB init, upsert, and query functions
- `src/phase3_demo.py` — NEW: demo that indexes the sample doc and runs a test query

**Dependency to install:**
```bash
uv pip install chromadb==0.5.5
```

**To resume, say:** `phase 3`
