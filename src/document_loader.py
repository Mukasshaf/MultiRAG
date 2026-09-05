from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Any

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".json", ".md", ".pptx"}

def _load_pdf(path: Path) -> tuple[str, int]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    content = "\n\n".join(doc.page_content for doc in docs)
    return content, len(docs)

def _load_txt(path: Path) -> str:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    return docs[0].page_content if docs else ""

def _load_docx(path: Path) -> str:
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    return docs[0].page_content if docs else ""

def _load_md(path: Path) -> str:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    return docs[0].page_content if docs else ""

def _load_json(path: Path) -> list | dict:
    import json
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data
    except Exception as e:
        logger.error(f"[LOAD] Failed to load JSON {path.name}: {e}")
        return []

def _load_pptx(path: Path):
    from pptx import Presentation
    return Presentation(str(path))

_LOADER_MAP = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
    ".docx": _load_docx,
    ".md": _load_md,
    ".json": _load_json,
    ".pptx": _load_pptx,
}

def load_file(path: Path) -> tuple[Any, str, int]:
    extention = path.suffix.lower()
    loader_fn = _LOADER_MAP.get(extention)
    if loader_fn is None:
        logger.warning(f"Unsupported file type '{extention}' for {path.name} - skipping.")
        return None, "", 0
    try:
        content = loader_fn(path)
        file_type = extention[1:]
        
        page_count = 0
        if file_type == "pdf":
            content, page_count = content
            
        logger.info(f"[LOAD] {path.name} loaded [{file_type}]")
        return content, file_type, page_count
    except Exception as e:
        logger.error(f"[LOAD] Failed to load {path.name}: {e}")
        return None, "", 0

def load_directory(data_dir: str) -> list[tuple[Any, str, int, Path]]:
  
    data_path = Path(data_dir).resolve()
    all_files = []

    for extention in SUPPORTED_EXTENSIONS:
        for file_path in sorted(data_path.rglob(f"*{extention}")):
            if file_path.name.startswith("."):
                continue
            content, ftype, pages = load_file(file_path)
            if content is not None:
                all_files.append((content, ftype, pages, file_path))

    logger.info(f"[LOAD] Total documents loaded from '{data_dir}': {len(all_files)}")
    return all_files
