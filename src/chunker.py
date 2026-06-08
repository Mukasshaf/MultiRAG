from __future__ import annotations

import logging
from typing import List

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import settings
from src.language_detector import tag_language

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> List[Document]:
    """
    Split documents into chunks and enrich metadata.

    Args:
        documents:     Raw list of LangChain Documents from document_loader.
        chunk_size:    Override default from settings.
        chunk_overlap: Override default from settings.

    Returns:
        List of chunk Documents with full metadata.
    """
    cs = chunk_size or settings.chunk_size
    co = chunk_overlap or settings.chunk_overlap

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=cs,
        chunk_overlap=co,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_documents(documents)
    logger.info(f"[CHUNK] {len(documents)} documents → {len(chunks)} chunks (size={cs}, overlap={co})")

    file_counters: dict[str, int] = {}
    for chunk in chunks:
        src = chunk.metadata.get("source_file", "unknown")
        idx = file_counters.get(src, 0)
        chunk.metadata["chunk_index"] = idx
        file_counters[src] = idx + 1

    chunks = tag_language(chunks)

    return chunks
