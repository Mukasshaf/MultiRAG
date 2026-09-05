from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from config import settings
from src.document_loader import load_file, SUPPORTED_EXTENSIONS
from chunking.router import chunk_document
from chunking.models import Chunk
from src.embedder import embed_texts
from src.pinecone_store import get_index, upsert_chunks
from src.ingestion_tracker import is_ingested, mark_ingested
from src.language_detector import tag_language

logger = logging.getLogger(__name__)

def ingest_file(file_path: Path, index=None, force: bool = False) -> dict:
    
    result = {"filename": file_path.name, "status": "skipped", "chunks_upserted": 0}

    if not force and is_ingested(file_path):
        logger.info(f"[INGEST] Already ingested, skipping: {file_path.name}")
        return result

    content, file_type, page_count = load_file(file_path)
    if content is None:
        result["status"] = "error_no_docs"
        return result

    doc_id = file_path.name
    chunks = chunk_document(file_type, content, doc_id, page_count)
    if not chunks:
        result["status"] = "error_no_chunks"
        return result

    for c in chunks:
        c.metadata["doc_id"] = doc_id
        c.metadata["source_file"] = file_path.name
        
    tag_language(chunks)

    texts = [c.embed_text for c in chunks]
    embeddings = embed_texts(texts, show_progress=True)

    if index is None:
        index = get_index()
    n = upsert_chunks(chunks, embeddings, index=index, use_sparse=False)

    mark_ingested(file_path)

    result["status"] = "success"
    result["chunks_upserted"] = n
    logger.info(f"[INGEST] ? {file_path.name} -> {n} chunks upserted")
    return result

def ingest_directory(data_dir: Optional[str] = None, force: bool = False) -> List[dict]:
    dir_path = Path(data_dir or settings.data_dir).resolve()
    logger.info(f"[INGEST] Scanning directory: {dir_path}")

    index = get_index()

    try:
        stats = index.describe_index_stats()
        if stats.total_vector_count == 0 and not force:
            logger.info("[INGEST] Pinecone index has 0 vectors - forcing re-ingestion.")
            force = True
    except Exception as e:
        logger.debug(f"[INGEST] Could not inspect index stats: {e}")

    all_chunk_texts: List[str] = []
    results = []

    for ext in SUPPORTED_EXTENSIONS:
        for file_path in sorted(dir_path.rglob(f"*{ext}")):
            if (
                file_path.name.startswith(".")
                or file_path.name == "bm25_model.json"
                or "eval" in file_path.parts
            ):
                continue

            if not force and is_ingested(file_path):
                logger.info(f"[INGEST] Already ingested, skipping: {file_path.name}")
                results.append({"filename": file_path.name, "status": "skipped", "chunks_upserted": 0})
                continue

            content, file_type, page_count = load_file(file_path)
            if content is None:
                results.append({"filename": file_path.name, "status": "error_no_docs", "chunks_upserted": 0})
                continue

            doc_id = file_path.name
            chunks = chunk_document(file_type, content, doc_id, page_count)
            if not chunks:
                results.append({"filename": file_path.name, "status": "error_no_chunks", "chunks_upserted": 0})
                continue

            for c in chunks:
                c.metadata["doc_id"] = doc_id
                c.metadata["source_file"] = file_path.name

            texts = [c.embed_text for c in chunks]
            all_chunk_texts.extend(texts)

            embeddings = embed_texts(texts, show_progress=True)
            n = upsert_chunks(chunks, embeddings, index=index, use_sparse=False)
            mark_ingested(file_path)

            results.append({"filename": file_path.name, "status": "success", "chunks_upserted": n})
            logger.info(f"[INGEST] ? {file_path.name} -> {n} chunks upserted")

    if all_chunk_texts:
        logger.info(f"[INGEST] Fitting BM25 on {len(all_chunk_texts)} chunks and re-upserting sparse vectors...")
        try:
            from src.bm25_encoder import fit_and_save, reset_encoder, encode_document as _enc_doc
            fit_and_save(all_chunk_texts)
            reset_encoder()
            _reindex_sparse(index, dir_path, results)
        except Exception as e:
            logger.warning(f"[INGEST] BM25 fitting failed (sparse vectors skipped): {e}")
    else:
        logger.info("[INGEST] No new documents - BM25 refit skipped.")

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors = sum(1 for r in results if r["status"].startswith("error"))
    total_chunks = sum(r["chunks_upserted"] for r in results)

    logger.info(
        f"[INGEST] Done. \nsuccess={success}, skipped={skipped}, errors={errors}, "
        f"total_chunks_upserted={total_chunks}"
    )
    return results

def _reindex_sparse(index, dir_path: Path, results: List[dict]) -> None:
    from src.bm25_encoder import encode_document
    from src.embedder import embed_texts

    successful_files = {r["filename"] for r in results if r["status"] == "success"}
    if not successful_files:
        return

    logger.info(f"[INGEST] Re-upserting sparse vectors for {len(successful_files)} file(s)...")
    for ext in SUPPORTED_EXTENSIONS:
        for file_path in sorted(dir_path.rglob(f"*{ext}")):
            if (
                file_path.name.startswith(".")
                or file_path.name == "bm25_model.json"
                or "eval" in file_path.parts
            ):
                continue
            if file_path.name not in successful_files:
                continue
                
            content, file_type, page_count = load_file(file_path)
            if content is None:
                continue
                
            doc_id = file_path.name
            chunks = chunk_document(file_type, content, doc_id, page_count)
            if not chunks:
                continue
                
            for c in chunks:
                c.metadata["doc_id"] = doc_id
                c.metadata["source_file"] = file_path.name
                
            texts = [c.embed_text for c in chunks]
            embeddings = embed_texts(texts, show_progress=False)
            upsert_chunks(chunks, embeddings, index=index, use_sparse=True)
            
    logger.info("[INGEST] Sparse re-indexing complete.")
