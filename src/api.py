from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from config import settings
from src.ingestion_tracker import list_ingested, unmark_ingested
from src.ingestor import ingest_file
from src.pinecone_store import delete_by_source, get_index, get_index_stats
from src.rag_chain import ask_stream
from src.memory import get_or_create_session

logger = logging.getLogger(__name__)
app = FastAPI(
    title="MultiRag",
    description="RAG system"
    
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_static_dir = Path(__file__).parent.parent / "static"
if _static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")

class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    top_k: int = 5
    filters: Optional[dict] = None

@app.get("/", include_in_schema=False)
async def serve_ui():
    html_path = Path(__file__).parent.parent / "static" / "index.html"
    if html_path.exists():
        return FileResponse(str(html_path))
    return {"message": "Multilingual RAG API v2 — see /docs"}

@app.get("/health")
async def health_check():
    status = {"ollama": "unknown", "pinecone": "unknown"}

    try:
        import ollama
        models = ollama.list()
        available = [m["model"] for m in models.get("models", [])]
        status["ollama"] = "ok" if settings.ollama_model in available else f"model '{settings.ollama_model}' not found"
        status["ollama_models"] = available
        status["active_model"] = settings.ollama_model
    except Exception as e:
        status["ollama"] = f"error: {e}"

    try:
        settings.validate()
        stats = get_index_stats()
        status["pinecone"] = "ok"
        status["pinecone_stats"] = stats
    except Exception as e:
        status["pinecone"] = f"error: {e}"

    return status

@app.post("/chat")
async def chat(request: ChatRequest):
    
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    session_id = get_or_create_session(request.session_id)

    def generate():
        try:
            for token in ask_stream(session_id, request.query, top_k=request.top_k, filters=request.filters):
                yield f"data: {token}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            logger.error(f"[API] Chat error: {e}")
            yield f"data: [ERROR] {str(e)}\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@app.post("/chat/full")
async def chat_full(request: ChatRequest):
    
    from src.rag_chain import ask
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
        
    session_id = get_or_create_session(request.session_id)
        
    try:
        result = ask(session_id, request.query, top_k=request.top_k, filters=request.filters)
        return result
    except Exception as e:
        logger.error(f"[API] Chat/full error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload")
async def upload_documents(files: List[UploadFile] = File(...)):
   
    data_path = Path(settings.data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    ALLOWED_SUFFIXES = {".pdf", ".txt", ".docx", ".csv", ".json", ".xlsx"}

    results = []
    index = get_index()

    for file in files:
        suffix = Path(file.filename).suffix.lower()
        if suffix not in ALLOWED_SUFFIXES:
            results.append({
                "filename": file.filename,
                "status": "rejected",
                "detail": f"Unsupported file type: {suffix}",
            })
            continue

        save_path = data_path / file.filename
        try:
            with open(save_path, "wb") as f:
                shutil.copyfileobj(file.file, f)
            logger.info(f"[API] Saved upload: {save_path}")
        except Exception as e:
            results.append({"filename": file.filename, "status": "save_error", "detail": str(e)})
            continue
        finally:
            await file.close()

        try:
            result = ingest_file(save_path, index=index, force=True)
            results.append(result)
        except Exception as e:
            logger.error(f"[API] Ingestion error for {file.filename}: {e}")
            results.append({"filename": file.filename, "status": "ingest_error", "detail": str(e)})

    return {"uploaded": len(files), "results": results}

@app.get("/documents")
async def list_documents():
    
    registry = list_ingested()
    data_path = Path(settings.data_dir)

    documents = []
    for filename, meta in registry.items():
        file_path = data_path / filename
        documents.append({
            "filename": filename,
            "file_type": Path(filename).suffix.lstrip(".").lower(),
            "size_bytes": meta.get("size_bytes", 0),
            "ingested_at": meta.get("ingested_at", ""),
            "exists_on_disk": file_path.exists(),
        })

    documents.sort(key=lambda d: d["ingested_at"], reverse=True)
    return {"total": len(documents), "documents": documents}

@app.delete("/documents/{filename}")
async def delete_document(filename: str):
    data_path = Path(settings.data_dir)
    file_path = data_path / filename

    try:
        index = get_index()
        delete_by_source(filename, index=index)
    except Exception as e:
        logger.warning(f"[API] Pinecone delete failed for '{filename}': {e}")

    unmark_ingested(filename)

    if file_path.exists():
        file_path.unlink()
        logger.info(f"[API] Deleted file from disk: {file_path}")

    return {"status": "deleted", "filename": filename}

@app.get("/index/stats")
async def index_stats():
    try:
        return get_index_stats()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
