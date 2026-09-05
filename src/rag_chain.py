from __future__ import annotations

import logging
from typing import Any, Dict, Generator, List, Optional

from config import settings
from chunking.parent_store import read_parent
from src.embedder import embed_query
from src.pinecone_store import get_index, query_index
from src.reranker import rerank
from src.memory import add_turn, get_history

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are a helpful, multilingual AI assistant. 
You answer questions based ONLY on the provided context documents.
If the context doesn't contain the answer, say so clearly do not make things up or create any information.
Always cite the source document and page number when you use information from it.
Answer in the same language as the user's question."""

_PROMPT_TEMPLATE = """## Context Documents
{context}

{query}

Given the following conversation history (if any) and the user's latest question, rephrase the latest question into a self-contained, standalone question or statement.

Rules:
1. Resolve Pronouns: If the latest question contains pronouns (e.g., "it", "they", "this") or implicitly refers to previous topics/sentences, replace them with the actual specific entities from the history.
2. Remove Meta-Talk: Strip out conversational filler like "briefly explain", "tell me", "what is", but keep the core grammatical structure intact. Do not aggressively shrink it to just keywords.
3. Keep Context: The result should be a clear, grammatically correct phrase/question that makes complete sense on its own.
4. Do NOT answer the question. Only output the rewritten question/phrase/statement.
5. Do NOT wrap the output in quotes.

Conversation History:
{history_str}

Latest Question: {query}

Standalone Query:"""

def _condense_query(query: str, history: List[Dict]) -> str:
    recent_history = history[-settings.memory_turn_window:] if history else []
    
    history_lines = []
    for turn in recent_history:
        role = turn.get("role", "user").capitalize()
        content = turn.get("content", "")
        history_lines.append(f"{role}: {content}")
        
    history_str = "\n".join(history_lines)
    prompt = _REWRITE_PROMPT.format(history_str=history_str, query=query)
    
    import ollama
    try:
        response = ollama.chat(
            model=settings.ollama_model,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.1},
        )
        condensed = response["message"]["content"].strip()
        logger.info(f"[RAG] Condensed query: '{query}' -> '{condensed}'")
        return condensed
    except Exception as e:
        logger.error(f"[RAG] Query condensation failed: {e}")
        return query

def _build_context(hits: List[Dict[str, Any]]) -> str:
    seen_parents = set()
    deduped_hits = []
    
    for hit in hits:
        parent_id = hit.get("parent_id")
        if parent_id:
            if parent_id in seen_parents:
                continue
            seen_parents.add(parent_id)
        deduped_hits.append(hit)
        
    lines = []
    for i, hit in enumerate(deduped_hits, 1):
        src = hit.get("doc_id", hit.get("source_file", "unknown"))
        page = hit.get("page", "?")
        lang = hit.get("language", "?")
        rerank_score = hit.get("rerank_score")
        dense_score = hit.get("score", 0.0)
        score_str = f"rerank={rerank_score:.4f}" if rerank_score is not None else f"dense={dense_score:.4f}"
        
        parent_id = hit.get("parent_id")
        if parent_id:
            text = read_parent(parent_id)
            if not text:
                text = hit.get("payload_text", "").strip()
        else:
            text = hit.get("payload_text", "").strip()
            
        lines.append(
            f"[{i}] Source: {src} | Page: {page} | Language: {lang} | {score_str}\n"
            f"{text}\n"
        )
    return "\n---\n".join(lines)

def _build_sparse_vector(query: str) -> Dict | None:
    try:
        from src.bm25_encoder import encode_query
        sparse = encode_query(query)
        if sparse and sparse.get("indices"):
            logger.info(f"[RAG] Sparse vector built ({len(sparse['indices'])} non-zero terms)")
            return sparse
    except Exception as e:
        logger.debug(f"[RAG] Sparse encoding skipped: {e}")
    return None

def ask(session_id: str, query: str, top_k: int | None = None, filters: dict | None = None) -> Dict[str, Any]:
    
    import ollama

    k = top_k or settings.top_k
    candidates_k = settings.reranker_candidates

    history = get_history(session_id)
    search_query = _condense_query(query, history)
    
    add_turn(session_id, "user", query, condensed_content=search_query if search_query != query else None)

    logger.info(f"[RAG] Search Query: {search_query!r}")
    q_embedding = embed_query(search_query)
    sparse_vec = _build_sparse_vector(search_query)

    index = get_index()
    candidates = query_index(
        q_embedding,
        top_k=candidates_k,
        index=index,
        sparse_vector=sparse_vec,
        alpha=settings.hybrid_alpha,
        filters=filters,
    )
    logger.info(f"[RAG] Retrieved {len(candidates)} candidates from Pinecone")

    hits = rerank(search_query, candidates, model_name=settings.reranker_model, top_n=k)
    logger.info(f"[RAG] Reranked to top {len(hits)} chunks")

    if not hits:
        answer = "I couldn't find any relevant information in the uploaded documents."
        add_turn(session_id, "assistant", answer, sources=[])
        return {
            "answer": answer,
            "sources": [],
            "query": search_query,
            "session_id": session_id,
        }

    context = _build_context(hits)
    user_prompt = _PROMPT_TEMPLATE.format(context=context, query=query)
    
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})

    logger.info(f"[RAG] Calling Ollama model: {settings.ollama_model}")
    response = ollama.chat(
        model=settings.ollama_model,
        messages=messages,
        options={"temperature": 0.1},
    )
    answer = response["message"]["content"].strip()
    add_turn(session_id, "assistant", answer, sources=hits)

    return {
        "answer": answer,
        "sources": hits,
        "query": search_query,
        "session_id": session_id,
    }

def ask_stream(session_id: str, query: str, top_k: int | None = None, filters: dict | None = None) -> Generator[str, None, None]:
    import ollama

    k = top_k or settings.top_k
    candidates_k = settings.reranker_candidates

    history = get_history(session_id)
    search_query = _condense_query(query, history)
    
    add_turn(session_id, "user", query, condensed_content=search_query if search_query != query else None)
    
    yield f"[SESSION_ID:{session_id}]\n"

    q_embedding = embed_query(search_query)
    sparse_vec = _build_sparse_vector(search_query)

    index = get_index()
    candidates = query_index(
        q_embedding,
        top_k=candidates_k,
        index=index,
        sparse_vector=sparse_vec,
        alpha=settings.hybrid_alpha,
        filters=filters,
    )
    hits = rerank(search_query, candidates, model_name=settings.reranker_model, top_n=k)

    if not hits:
        answer = "I couldn't find any relevant information in the uploaded documents."
        add_turn(session_id, "assistant", answer, sources=[])
        yield answer
        return

    context = _build_context(hits)
    user_prompt = _PROMPT_TEMPLATE.format(context=context, query=query)
    
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]
    for turn in history:
        messages.append({"role": turn["role"], "content": turn["content"]})
    messages.append({"role": "user", "content": user_prompt})

    stream = ollama.chat(
        model=settings.ollama_model,
        messages=messages,
        options={"temperature": 0.1},
        stream=True,
    )
    
    full_answer = []
    for chunk in stream:
        token = chunk["message"]["content"]
        if token:
            full_answer.append(token)
            yield token
            
    add_turn(session_id, "assistant", "".join(full_answer), sources=hits)
