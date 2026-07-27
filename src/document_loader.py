from __future__ import annotations

import logging
from pathlib import Path
from typing import List

from langchain_core.documents import Document

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".docx", ".csv", ".json", ".xlsx"}


def _load_pdf(path: Path) -> List[Document]:
    from langchain_community.document_loaders import PyPDFLoader
    loader = PyPDFLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = "pdf"
    return docs


def _load_txt(path: Path) -> List[Document]:
    from langchain_community.document_loaders import TextLoader
    loader = TextLoader(str(path), encoding="utf-8", autodetect_encoding=True)
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = "txt"
        doc.metadata["page"] = 0
    return docs


def _load_docx(path: Path) -> List[Document]:
    from langchain_community.document_loaders import Docx2txtLoader
    loader = Docx2txtLoader(str(path))
    docs = loader.load()
    for doc in docs:
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = "docx"
        doc.metadata["page"] = 0
    return docs


def _load_csv(path: Path) -> List[Document]:
    from langchain_community.document_loaders.csv_loader import CSVLoader
    loader = CSVLoader(str(path), encoding="utf-8")
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = "csv"
        doc.metadata.setdefault("page", i)
    return docs


def _load_json(path: Path) -> List[Document]:
    from langchain_community.document_loaders import JSONLoader
    loader = JSONLoader(
        file_path=str(path),
        jq_schema=".",
        text_content=False,
    )
    docs = loader.load()
    for i, doc in enumerate(docs):
        doc.metadata["source_file"] = path.name
        doc.metadata["file_type"] = "json"
        doc.metadata.setdefault("page", i)
    return docs


def _load_excel(path: Path) -> List[Document]:
    import openpyxl
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    docs = []
    for i, sheet_name in enumerate(wb.sheetnames):
        ws = wb[sheet_name]
        rows = []
        for row in ws.iter_rows(values_only=True):
            row_text = "\t".join(str(cell) if cell is not None else "" for cell in row)
            if row_text.strip():
                rows.append(row_text)
        if rows:
            content = f"[Sheet: {sheet_name}]\n" + "\n".join(rows)
            docs.append(Document(
                page_content=content,
                metadata={
                    "source_file": path.name,
                    "file_type": "xlsx",
                    "page": i,
                    "sheet": sheet_name,
                }
            ))
    wb.close()
    return docs


_LOADER_MAP = {
    ".pdf": _load_pdf,
    ".txt": _load_txt,
    ".docx": _load_docx,
    ".csv": _load_csv,
    ".json": _load_json,
    ".xlsx": _load_excel,
}


def load_file(path: Path) -> List[Document]:
    extention = path.suffix.lower()
    loader_fn = _LOADER_MAP.get(extention)
    if loader_fn is None:
        logger.warning(f"Unsupported file type '{extention}' for {path.name} — skipping.")
        return []
    try:
        docs = loader_fn(path)
        logger.info(f"[LOAD] {path.name} → {len(docs)} document(s) [{extention}]")
        return docs
    except Exception as e:
        logger.error(f"[LOAD] Failed to load {path.name}: {e}")
        return []


def load_directory(data_dir: str) -> List[Document]:
  
    data_path = Path(data_dir).resolve()
    all_docs: List[Document] = []

    for extention in SUPPORTED_EXTENSIONS:
        for file_path in sorted(data_path.rglob(f"*{extention}")):
            if file_path.name.startswith("."):
                continue
            docs = load_file(file_path)
            all_docs.extend(docs)

    logger.info(f"[LOAD] Total documents loaded from '{data_dir}': {len(all_docs)}")
    return all_docs
