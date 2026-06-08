from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List

from config import settings
from src.embedder import embed_query
from src.pinecone_store import get_index, query_index

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful, multilingual AI assistant. 
You answer questions based ONLY on the provided context documents.
If the context doesn't contain the answer, say so clearly do not make things up.
Always cite the source document and page number when you use information from it.
Answer in the same language as the user's question."""

_PROMPT_TEMPLATE = """{system}

## Context Documents
{context}

## User Question
{query}

## Answer
"""


def _build_context(hits: List[Dict[str, Any]]) -> str:
    """Format Pinecone hits into a readable context block."""
    lines = []
    for i, hit in enumerate(hits, 1):
        src = hit.get("source_file", "unknown")
        page = hit.get("page", "?")
        lang = hit.get("language", "?")
        score = hit.get("score", 0.0)
        text = hit.get("text", "").strip()
        lines.append(
            f"[{i}] Source: {src} | Page: {page} | Language: {lang} | Score: {score:.3f}\n"
            f"{text}\n"
        )
    return "\n---\n".join(lines)


def ask(query: str, top_k: int | None = None) -> Dict[str, Any]:
    """
    Full RAG pipeline (blocking / non-streaming).

    Returns:
        {
          "answer": str,
          "sources": List[dict],
          "query": str,
        }
    """
    import ollama

    k = top_k or settings.top_k

    logger.info(f"[RAG] Query: {query!r}")
    q_embedding = embed_query(query)

    index = get_index()
    hits = query_index(q_embedding, top_k=k, index=index)
    logger.info(f"[RAG] Retrieved {len(hits)} chunks from Pinecone")

    if not hits:
        return {
            "answer": "I couldn't find any relevant information in the uploaded documents.",
            "sources": [],
            "query": query,
        }

    context = _build_context(hits)
    prompt = _PROMPT_TEMPLATE.format(
        system=_SYSTEM_PROMPT,
        context=context,
        query=query,
    )

    logger.info(f"[RAG] Calling Ollama model: {settings.ollama_model}")
    response = ollama.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
    )
    answer = response["message"]["content"].strip()

    return {
        "answer": answer,
        "sources": hits,
        "query": query,
    }


def ask_stream(query: str, top_k: int | None = None) -> Generator[str, None, None]:
    import ollama

    k = top_k or settings.top_k

    q_embedding = embed_query(query)
    index = get_index()
    hits = query_index(q_embedding, top_k=k, index=index)

    if not hits:
        yield "I couldn't find any relevant information in the uploaded documents."
        return

    context = _build_context(hits)
    prompt = _PROMPT_TEMPLATE.format(
        system=_SYSTEM_PROMPT,
        context=context,
        query=query,
    )

    stream = ollama.chat(
        model=settings.ollama_model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0.1},
        stream=True,
    )
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            yield token
