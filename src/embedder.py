from __future__ import annotations

import logging
from typing import List

import numpy as np

logger = logging.getLogger(__name__)

_MODEL_NAME = "paraphrase-multilingual-MiniLM-L12-v2"
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        logger.info(f"[EMBED] Loading embedding model: {_MODEL_NAME}")
        _model = SentenceTransformer(_MODEL_NAME)
        logger.info(f"[EMBED] Model loaded. Output dim: {_model.get_sentence_embedding_dimension()}")
    return _model


def embed_texts(texts: List[str], batch_size: int = 64, show_progress: bool = False) -> List[List[float]]:
    """
    Embed a list of strings.

    Args:
        texts:         Input strings.
        batch_size:    Batch size for encoding (default 64).
        show_progress: Show tqdm progress bar.

    Returns:
        List of float vectors (each of length 384).
    """
    model = _get_model()
    if not texts:
        return []
    embeddings: np.ndarray = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    logger.info(f"[EMBED] Embedded {len(texts)} texts → shape {embeddings.shape}")
    return embeddings.tolist()


def embed_query(query: str) -> List[float]:
    """Embed a single query string."""
    return embed_texts([query])[0]
