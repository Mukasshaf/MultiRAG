from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_reranker = None

def _get_reranker(model_name: str):
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        logger.info(f"[RERANKER] Loading model '{model_name}'...")
        _reranker = CrossEncoder(model_name, max_length=512)
        logger.info("[RERANKER] Model loaded.")
    return _reranker

def rerank(
    query: str,
    candidates: List[Dict[str, Any]],
    model_name: str,
    top_n: int = 5,
) -> List[Dict[str, Any]]:
    if not candidates:
        return candidates

    reranker = _get_reranker(model_name)
    pairs = [(query, c.get("text", "")) for c in candidates]
    scores = reranker.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    ranked = sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)
    logger.info(
        f"[RERANKER] Reranked {len(candidates)} candidates -> top {top_n}. "
        f"Top score: {ranked[0]['rerank_score']:.4f}"
    )
    return ranked[:top_n]
