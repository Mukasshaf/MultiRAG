# Data Directory & Document Storage

This directory serves as the persistent local document store for the MultiRAG system. Source documents placed here are processed, parsed, chunked, and indexed into Pinecone for semantic and keyword retrieval.

---

## Directory Layout

```
data/
├── README.md                  # This documentation file
├── documents/                 # Default directory for raw document uploads
├── parent_store/              # Saved parent documents for hierarchical parent-child chunking
└── bm25_model.json            # Statistical TF-IDF vocabulary weights for sparse keyword search
```

---

## 📄 Supported Document Formats

MultiRAG supports heterogeneous multi-format ingestion via dedicated loaders and strategy routers:

| Extension | Format | Parser / Loader | Routing Strategy |
| :--- | :--- | :--- | :--- |
| **`.pdf`** | PDF Documents | `PyPDFLoader` | Hierarchical Parent-Child (textbooks >12 pages) or Recursive Sentence-Boundary (<12 pages) |
| **`.docx`** | Microsoft Word | `Docx2txtLoader` | Header-Ancestor Hierarchy (`H1 > H2 > H3`) |
| **`.pptx`** | Presentations | `python-pptx` | Slide-Node Chunking (Slide titles, body bullets & table text) |
| **`.md`** | Markdown Files | `TextLoader` / UTF-8 | Header-Ancestor Hierarchy preserving markdown section outlines |
| **`.txt`** | Plain Text | `TextLoader` (Auto-detect) | Recursive Sentence-Boundary with NLTK grammatical punctuation |
| **`.json`** | JSON Datasets | Native JSON parser | Schema-Aware chunking by record/object |

> [NOTE]
> PDFs must contain a selectable text layer. Scanned, image-only PDFs require OCR preprocessing before ingestion.

---

## How Ingestion Works

1. **Automatic Web Ingestion:**
   - When a file is uploaded through the web browser UI (`http://localhost:8000`), the file is stored in `data/documents/` and immediately processed through the ingestion pipeline.
2. **Deduplication & State Tracking:**
   - MultiRAG tracks ingested documents in an internal registry (`data/.ingestion_registry.json`).
   - If a file has already been indexed and its file modification time hasn't changed, the ingestion coordinator skips it to prevent duplicate vector upserts.
3. **Parent Document Persistence:**
   - For long documents using hierarchical chunking, the complete section text is saved locally in `parent_store/`. 
   - When small child chunks match a search query, the retriever fetches the full parent text from `parent_store/` so the LLM has complete context.
4. **BM25 Sparse Model:**
   - After batch ingestion, `bm25_model.json` stores the computed inverse document frequency (IDF) statistics across all your chunks.

---

## Manual CLI Commands

You can drop files directly into `data/documents/` and run ingestion from the command line:

```bash
# Ingest all new/untracked documents in data/
uv run python scripts/ingest.py

# Force re-ingestion and vector re-indexing of all documents
uv run python scripts/ingest.py --force

# Ingest from a custom subfolder
uv run python scripts/ingest.py --dir data/custom_folder

# Re-fit the BM25 keyword model after changing the corpus
uv run python scripts/fit_bm25.py
```
