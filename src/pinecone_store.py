from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from config import settings

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100  # Pinecone upsert batch limit


def _make_id(source_file: str, chunk_index: int) -> str:
    raw = f"{source_file}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_index():
    """Return a live Pinecone Index object, creating the index if needed."""
    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info(f"[PINECONE] Creating index '{index_name}' (dim={settings.embedding_dim}, cosine)")
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dim,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region=settings.pinecone_region),
        )
       
        import time
        while not pc.describe_index(index_name).status["ready"]:
            logger.info("[PINECONE] Waiting for index to be ready...")
            time.sleep(2)
        logger.info(f"[PINECONE] Index '{index_name}' is ready.")
    else:
        logger.info(f"[PINECONE] Using existing index '{index_name}'.")

    return pc.Index(index_name)


def upsert_chunks(
    chunks: List[Any],           # List[langchain_core.documents.Document]
    embeddings: List[List[float]],
    index=None,
) -> int:
    """
    Upsert chunk vectors + metadata into Pinecone.

    Returns:
        Number of vectors upserted.
    """
    if index is None:
        index = get_index()

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        meta = chunk.metadata
        source_file = meta.get("source_file", "unknown")
        chunk_index = meta.get("chunk_index", 0)
        vec_id = _make_id(source_file, chunk_index)

        pinecone_meta: Dict[str, Any] = {
            "text": chunk.page_content[:1000],  
            "source_file": source_file,
            "file_type": meta.get("file_type", "unknown"),
            "page": int(meta.get("page", 0)),
            "language": meta.get("language", "unknown"),
            "chunk_index": chunk_index,
        }
        vectors.append({"id": vec_id, "values": embedding, "metadata": pinecone_meta})

    total = 0
    for i in range(0, len(vectors), _BATCH_SIZE):
        batch = vectors[i : i + _BATCH_SIZE]
        index.upsert(vectors=batch)
        total += len(batch)
        logger.info(f"[PINECONE] Upserted batch {i // _BATCH_SIZE + 1}: {len(batch)} vectors")

    logger.info(f"[PINECONE] Total upserted: {total} vectors")
    return total


def query_index(
    query_embedding: List[float],
    top_k: int = 5,
    index=None,
) -> List[Dict[str, Any]]:
    """
    Query Pinecone for top-k similar vectors.

    Returns:
        List of dicts with keys: id, score, text, source_file, file_type, page, language, chunk_index
    """
    if index is None:
        index = get_index()

    response = index.query(
        vector=query_embedding,
        top_k=top_k,
        include_metadata=True,
    )

    results = []
    for match in response.matches:
        meta = match.metadata or {}
        results.append({
            "id": match.id,
            "score": round(float(match.score), 4),
            "text": meta.get("text", ""),
            "source_file": meta.get("source_file", "unknown"),
            "file_type": meta.get("file_type", "unknown"),
            "page": meta.get("page", 0),
            "language": meta.get("language", "unknown"),
            "chunk_index": meta.get("chunk_index", 0),
        })

    return results


def delete_by_source(source_file: str, index=None) -> None:
    if index is None:
        index = get_index()

    index.delete(filter={"source_file": {"$eq": source_file}})
    logger.info(f"[PINECONE] Deleted all vectors for source_file='{source_file}'")


def get_index_stats(index=None) -> Dict[str, Any]:
    """Return index stats (total vector count, dimension, etc.)."""
    if index is None:
        index = get_index()
    stats = index.describe_index_stats()
    return {
        "total_vector_count": stats.total_vector_count,
        "dimension": stats.dimension,
        "index_fullness": stats.index_fullness,
    }
