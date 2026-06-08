from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from config import settings
from src.document_loader import load_file, SUPPORTED_EXTENSIONS
from src.chunker import chunk_documents
from src.embedder import embed_texts
from src.pinecone_store import get_index, upsert_chunks
from src.ingestion_tracker import is_ingested, mark_ingested

logger = logging.getLogger(__name__)


def ingest_file(file_path: Path, index=None, force: bool = False) -> dict:
    """
    Ingest a single file into Pinecone.

    Args:
        file_path: Path to the document.
        index:     Reuse an existing Pinecone index handle.
        force:     If True, re-ingest even if already tracked.

    Returns:
        dict with keys: filename, status, chunks_upserted
    """
    result = {"filename": file_path.name, "status": "skipped", "chunks_upserted": 0}

    if not force and is_ingested(file_path):
        logger.info(f"[INGEST] Already ingested, skipping: {file_path.name}")
        return result

    documents = load_file(file_path)
    if not documents:
        result["status"] = "error_no_docs"
        return result

    chunks = chunk_documents(documents)
    if not chunks:
        result["status"] = "error_no_chunks"
        return result

    texts = [c.page_content for c in chunks]
    embeddings = embed_texts(texts, show_progress=True)

    if index is None:
        index = get_index()
    n = upsert_chunks(chunks, embeddings, index=index)

    mark_ingested(file_path)

    result["status"] = "success"
    result["chunks_upserted"] = n
    logger.info(f"[INGEST] ✓ {file_path.name} → {n} chunks upserted")
    return result


def ingest_directory(data_dir: Optional[str] = None, force: bool = False) -> List[dict]:
    """
    Scan the data directory and ingest all new/changed supported files.

    Args:
        data_dir: Path override (defaults to settings.data_dir).
        force:    Re-ingest all files regardless of registry.

    Returns:
        List of result dicts (one per file).
    """
    dir_path = Path(data_dir or settings.data_dir).resolve()
    logger.info(f"[INGEST] Scanning directory: {dir_path}")

    index = get_index()

    results = []
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in sorted(dir_path.rglob(f"*{ext}")):
            if file_path.name.startswith("."):
                continue
            result = ingest_file(file_path, index=index, force=force)
            results.append(result)

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"].startswith("error"))
    total_chunks = sum(r["chunks_upserted"] for r in results)

    logger.info(
        f"[INGEST] Done. success={success}, skipped={skipped}, errors={errors}, "
        f"total_chunks_upserted={total_chunks}"
    )
    return results
