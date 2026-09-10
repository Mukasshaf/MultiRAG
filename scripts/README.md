# MultiRAG Maintenance & Evaluation Scripts

This directory contains standalone CLI tools for offline ingestion, BM25 model training, and RAG evaluation benchmarking.

---

## Available Scripts

| Script | Command | Purpose |
| :--- | :--- | :--- |
| **`ingest.py`** | `uv run python scripts/ingest.py` | Scans documents, applies chunking strategies, creates dense embeddings, and upserts vectors to Pinecone. |
| **`fit_bm25.py`** | `uv run python scripts/fit_bm25.py` | Fits BM25 vocabulary on all indexed chunks and exports `bm25_model.json` for hybrid sparse search. |
| **`evaluate.py`** | `uv run python scripts/evaluate.py` | Runs end-to-end RAG accuracy evaluation against a benchmark dataset (calculates Hit Rate, MRR, and latency). |

---

## 1. Document Ingestion — `scripts/ingest.py`

Coordinates the entire document ingestion pipeline from the command line without opening the web browser.

### Features:
- Automatically routes each file to its optimal chunking strategy based on file extension and page count.
- Computes SHA-256 / modification timestamps to skip already-ingested files.
- Generates 384-dimensional dense vectors using `all-MiniLM-L6-v2`.
- Batches vector upserts to Pinecone Serverless.

### Usage:
```bash
# Ingest any new files in the default data directory (configured in .env)
uv run python scripts/ingest.py

# Force re-ingest all files (even if previously tracked)
uv run python scripts/ingest.py --force

# Ingest from a specific custom folder
uv run python scripts/ingest.py --dir path/to/my_documents

# Ingest a single file
uv run python scripts/ingest.py --file path/to/my_documents/report.pdf
```

### CLI Arguments:
| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--dir` | `str` | `.env: DATA_DIR` | Target directory containing documents to ingest |
| `--file` | `str` | `None` | Path to a single specific document to ingest |
| `--force` | `flag` | `False` | Bypasses the ingestion registry and re-chunks all files |

---

## 2. BM25 Model Fitting — `scripts/fit_bm25.py`

Computes the term frequency and inverse document frequency (TF-IDF) statistics over your indexed documents to build the BM25 sparse vocabulary.

### When to run this:
- **After first ingesting documents into Pinecone.**
- **Whenever you add a substantial batch of new files or re-ingest with `--force`.**

### Usage:
```bash
# Fit BM25 on the default data directory
uv run python scripts/fit_bm25.py

# Fit BM25 on a custom folder
uv run python scripts/fit_bm25.py --dir path/to/my_documents
```

### Output:
Generates `data/bm25_model.json` (or the path defined in `BM25_MODEL_PATH`), containing the token weights needed for hybrid search alpha blending during live queries.

---

## 3. RAG Evaluation Benchmark — `scripts/evaluate.py`

Measures the retrieval quality and answer fidelity of the MultiRAG system against a golden set of questions and expected answers.

### Metrics Computed:
- **Hit Rate @ K:** Percentage of queries where the true relevant document is present in the retrieved candidates.
- **Mean Reciprocal Rank (MRR):** Measures how high up the correct document appears after BAAI cross-encoder reranking.
- **Latency / Response Time:** Average query condensation, retrieval, and LLM synthesis duration.

### Usage:
```bash
# Run benchmark with default settings
uv run python scripts/evaluate.py

# Run benchmark with custom sample size
uv run python scripts/evaluate.py --samples 20

# Run evaluation and save JSON report
uv run python scripts/evaluate.py --output eval_report.json
```

---

##  Recommended End-to-End Workflow

When setting up or updating your MultiRAG knowledge base:

```bash
# Step 1: Copy documents into data/
cp path/to/files/* data/documents/

# Step 2: Ingest documents into Pinecone
uv run python scripts/ingest.py --force

# Step 3: Train and save the BM25 model weights
uv run python scripts/fit_bm25.py

# Step 4: Run evaluation to verify retrieval precision
uv run python scripts/evaluate.py

# Step 5: Start the web application
uv run python main.py
```
