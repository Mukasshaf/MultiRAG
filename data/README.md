# Data Directory

Drop your documents here. The RAG system will automatically pick them up.

## Supported Formats

| Extension | Description |
|-----------|-------------|
| `.pdf`    | PDF documents (text-based, not scanned images) |
| `.txt`    | Plain text files (any encoding, auto-detected) |
| `.docx`   | Microsoft Word documents |
| `.csv`    | Comma-separated values |
| `.json`   | JSON files (all content extracted) |
| `.xlsx`   | Microsoft Excel spreadsheets |

## How Ingestion Works

1. **Upload** files via the web UI, or drop them directly into this folder.
2. **Run the ingestor** (web uploads do this automatically):
   ```bash
   python scripts/ingest.py
   ```
3. Files are **skipped if already ingested** — only new or changed files are processed.
4. To force re-ingestion of all files:
   ```bash
   python scripts/ingest.py --force
   ```

## Notes

- `.ingested_registry.json` is auto-created here — do not delete it, or files will be re-ingested.
- Scanned PDFs (image-only) are not supported without OCR. Use text-layer PDFs.
- Very large files (>100 MB) may be slow to process.
