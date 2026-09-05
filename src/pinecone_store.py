from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional

from config import settings
from chunking.models import Chunk

logger = logging.getLogger(__name__)

_BATCH_SIZE = 100

def _make_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]

def get_index():

    from pinecone import Pinecone, ServerlessSpec

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index_name = settings.pinecone_index_name

    existing = [idx.name for idx in pc.list_indexes()]
    if index_name not in existing:
        logger.info(f"[PINECONE] Creating index '{index_name}' (dim={settings.embedding_dim}, dotproduct)")
        pc.create_index(
            name=index_name,
            dimension=settings.embedding_dim,
            metric="dotproduct",
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
    chunks: List[Chunk],         
    embeddings: List[List[float]],
    index=None,
    use_sparse: bool = True,
) -> int:
    
    if index is None:
        index = get_index()

    if use_sparse:
        try:
            from src.bm25_encoder import encode_document as _encode_doc
            _sparse_available = True
        except Exception as e:
            logger.warning(f"[PINECONE] BM25 sparse encoding unavailable: {e}. Falling back to dense-only.")
            _sparse_available = False
    else:
        _sparse_available = False

    vectors = []
    for chunk, embedding in zip(chunks, embeddings):
        meta = chunk.metadata
        
        doc_id = meta.get("doc_id", "unknown")
        chunk_index = meta.get("chunk_index", 0)
        source_type = meta.get("source_type")
        content_type = meta.get("content_type")
        parent_id = meta.get("parent_id")
        
        if not source_type or not content_type:
            logger.warning(f"[PINECONE] Skipping chunk {doc_id}:{chunk_index} - missing source_type or content_type")
            continue
            
        vec_id = _make_id(doc_id, chunk_index)

        pinecone_meta: Dict[str, Any] = {
            "source_file": meta.get("source_file", doc_id),
            "doc_id": doc_id,
            "source_type": source_type,
            "content_type": content_type,
            "page": int(meta.get("page", 0)),
            "chunk_index": chunk_index,
            "payload_text": chunk.payload_text[:2000],
            "language": meta.get("language", "unknown")
        }
        
        if parent_id:
            pinecone_meta["parent_id"] = parent_id

        vec: Dict[str, Any] = {
            "id": vec_id,
            "values": embedding,
            "metadata": pinecone_meta,
        }

        if _sparse_available:
            try:
                sparse_vec = _encode_doc(chunk.embed_text)
                if sparse_vec.get("indices"):
                    vec["sparse_values"] = sparse_vec
            except Exception as e:
                logger.debug(f"[BM25] Sparse encoding failed for chunk {vec_id}: {e}")

        vectors.append(vec)

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
    sparse_vector: Optional[Dict] = None,
    alpha: Optional[float] = None,
    filters: Optional[Dict] = None,
) -> List[Dict[str, Any]]:
   
    if index is None:
        index = get_index()

    query_kwargs: Dict[str, Any] = {
        "top_k": top_k,
        "include_metadata": True,
    }

    if sparse_vector and sparse_vector.get("indices"):
        _alpha = alpha if alpha is not None else settings.hybrid_alpha
        scaled_dense = [v * _alpha for v in query_embedding]
        scaled_sparse_vals = [v * (1 - _alpha) for v in sparse_vector.get("values", [])]
        query_kwargs["vector"] = scaled_dense
        query_kwargs["sparse_vector"] = {
            "indices": sparse_vector["indices"],
            "values": scaled_sparse_vals,
        }
        logger.info(f"[PINECONE] Hybrid query (alpha={_alpha:.2f}) top_k={top_k}")
    else:
        query_kwargs["vector"] = query_embedding
        logger.info(f"[PINECONE] Dense-only query top_k={top_k}")

    if filters:
        query_kwargs["filter"] = filters
        logger.info(f"[PINECONE] Applying metadata filter: {filters}")

    response = index.query(**query_kwargs)

    results = []
    for match in response.matches:
        meta = match.metadata or {}
        results.append({
            "id": match.id,
            "score": round(float(match.score), 4),
            "doc_id": meta.get("doc_id", meta.get("source_file", "unknown")),
            "source_type": meta.get("source_type", meta.get("file_type", "unknown")),
            "source_file": meta.get("source_file", meta.get("doc_id", "unknown")),
            "file_type": meta.get("file_type", meta.get("source_type", "unknown")),
            "language": meta.get("language", "unknown"),
            "content_type": meta.get("content_type", "unknown"),
            "page": meta.get("page", 0),
            "chunk_index": meta.get("chunk_index", 0),
            "payload_text": meta.get("payload_text", meta.get("text", "")),
            "parent_id": meta.get("parent_id")
        })

    return results

def delete_by_source(source_file: str, index=None) -> None:
    if index is None:
        index = get_index()

    index.delete(filter={"source_file": {"": source_file}})
    logger.info(f"[PINECONE] Deleted all vectors for source_file='{source_file}'")

def get_index_stats(index=None) -> Dict[str, Any]:
    if index is None:
        index = get_index()
    stats = index.describe_index_stats()
    return {
        "total_vector_count": stats.total_vector_count,
        "dimension": stats.dimension,
        "index_fullness": stats.index_fullness,
    }
