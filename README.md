# MultiRAG v2 — Multilingual Document Intelligence

A production-ready, **fully local** RAG (Retrieval-Augmented Generation) system that lets you ask questions across multiple documents in **any language**.


## Prerequisites

| Tool | Install |
|------|---------|
| Python 3.10+ | [python.org](https://python.org) |
| uv | `pip install uv` |
| Ollama | [ollama.ai](https://ollama.ai) |
| Pinecone account | [pinecone.io](https://pinecone.io) (free tier works) |

## Quick Start

### 1. Clone & install dependencies

```bash
cd "C:\My Drive\codes\rag 2"
uv sync
```

### 2. Configure environment

```bash
copy .env.example .env
```

Edit `.env` and fill in:
```env
PINECONE_API_KEY=your_key_here
PINECONE_INDEX_NAME=multilingual-rag   # will be auto-created
PINECONE_REGION=us-east-1
OLLAMA_MODEL=mistral
```

### 3. Pull the Ollama model

```bash
ollama pull mistral
ollama serve          # keep this running in a separate terminal
```

### 4. Start the web server

```bash
uv run python main.py
```

Open **http://localhost:8000** in your browser.

### 5. Upload documents & ask questions

- Drag & drop files into the sidebar, or use the upload button.
- Type your question in any language — the system will answer using your documents.
- Click **Sources** below any answer to see exactly which document + page was used.

---

## Manual Ingestion (CLI)

```bash
# Ingest all new files in data/
uv run python scripts/ingest.py

# Force re-ingest all files
uv run python scripts/ingest.py --force

# Custom directory
uv run python scripts/ingest.py --dir path/to/docs
```

---

## API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET`  | `/` | Web UI |
| `GET`  | `/health` | Ollama + Pinecone status |
| `POST` | `/chat/full` | Query → JSON `{answer, sources}` |
| `POST` | `/upload` | Upload files (multipart) |
| `GET`  | `/documents` | List ingested docs |
| `DELETE` | `/documents/{filename}` | Delete doc + Pinecone vectors |
| `GET`  | `/index/stats` | Pinecone index statistics |
| `GET`  | `/docs` | FastAPI Swagger UI |

---

## Customization

All settings are in `.env`:

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `mistral` | Ollama model to use |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `TOP_K_RESULTS` | `5` | Pinecone results per query |
| `PINECONE_REGION` | `us-east-1` | Pinecone serverless region |

---

## Supported Languages (embedding)

The `paraphrase-multilingual-MiniLM-L12-v2` model supports **50+ languages** including:
English · French · German · Spanish · Portuguese · Italian · Dutch · Polish · Russian · Chinese · Japanese · Korean · Arabic · Hindi · and many more.

---

*Built on: LangChain · sentence-transformers · Pinecone · Ollama · FastAPI*
