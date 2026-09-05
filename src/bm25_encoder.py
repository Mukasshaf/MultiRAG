from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

_bm25_instance = None

def _get_encoder():
    from pinecone_text.sparse import BM25Encoder
    global _bm25_instance
    if _bm25_instance is None:
        from config import settings
        model_path = Path(settings.bm25_model_path)
        if model_path.exists():
            logger.info(f"[BM25] Loading model from {model_path}")
            _bm25_instance = BM25Encoder()
            _bm25_instance.load(str(model_path))
        else:
            logger.warning(
                f"[BM25] No saved model at {model_path}. "
                "Returning default encoder (unfitted). Run scripts/fit_bm25.py after ingestion."
            )
            _bm25_instance = BM25Encoder().default()
    return _bm25_instance

def reset_encoder() -> None:
    global _bm25_instance
    _bm25_instance = None

def fit_and_save(corpus_texts: List[str]) -> None:
    from pinecone_text.sparse import BM25Encoder
    from config import settings

    model_path = Path(settings.bm25_model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"[BM25] Fitting on {len(corpus_texts)} documents...")
    encoder = BM25Encoder()
    encoder.fit(corpus_texts)
    encoder.dump(str(model_path))
    logger.info(f"[BM25] Model saved to {model_path}")

    global _bm25_instance
    _bm25_instance = encoder

def encode_document(text: str) -> Dict:
    encoder = _get_encoder()
    result = encoder.encode_documents(text)
    if isinstance(result, list):
        result = result[0] if result else {"indices": [], "values": []}
    return result

def encode_query(text: str) -> Dict:
    encoder = _get_encoder()
    result = encoder.encode_queries(text)
    if isinstance(result, list):
        result = result[0] if result else {"indices": [], "values": []}
    return result
