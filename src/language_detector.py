from __future__ import annotations

import logging
from typing import List

from chunking.models import Chunk

logger = logging.getLogger(__name__)

_MIN_DETECT_LEN = 20

def detect_language(text: str) -> str:
    if len(text.strip()) < _MIN_DETECT_LEN:
        return "unknown"
    try:
        from langdetect import detect, LangDetectException
        return detect(text)
    except Exception:
        return "unknown"

def tag_language(chunks: List[Chunk]) -> List[Chunk]:
    lang_counts: dict[str, int] = {}
    for chunk in chunks:
        lang = detect_language(chunk.payload_text)
        chunk.metadata["language"] = lang
        lang_counts[lang] = lang_counts.get(lang, 0) + 1

    logger.info(f"[LANG] Language distribution across {len(chunks)} chunks: {lang_counts}")
    return chunks
