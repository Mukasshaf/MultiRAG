from typing import List, Dict, Any
from .models import Chunk
from .parent_store import write_parent
import json

def recursive_fallback_chunk(text: str, doc_id: str, source_type: str) -> List[Chunk]:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    texts = splitter.split_text(text)
    chunks = []
    for i, t in enumerate(texts):
        meta = {
            "chunk_index": i,
            "source_type": source_type,
            "content_type": "text"
        }
        chunks.append(Chunk(embed_text=t, payload_text=t, metadata=meta))
    return chunks

def header_ancestor_chunk(content: str, doc_id: str, file_type: str) -> List[Chunk]:
    from langchain_text_splitters import MarkdownHeaderTextSplitter
    
    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    md_header_splits = markdown_splitter.split_text(content)
    
    chunks = []
    for i, split in enumerate(md_header_splits):
        header_path = []
        for key in ["Header 1", "Header 2", "Header 3"]:
            if key in split.metadata:
                header_path.append(split.metadata[key])
        
        path_str = " > ".join(header_path) if header_path else "Document"
        embed_text = f"Section: {path_str}\n\n{split.page_content}"
        payload_text = embed_text
        
        meta = {
            "chunk_index": i,
            "source_type": file_type,
            "content_type": "text"
        }
        chunks.append(Chunk(embed_text=embed_text, payload_text=payload_text, metadata=meta))
    
    return chunks

def slide_node_chunk(prs: Any, doc_id: str) -> List[Chunk]:
    chunks = []
    for i, slide in enumerate(prs.slides):
        title = ""
        if slide.shapes.title and slide.shapes.title.has_text_frame:
            title = slide.shapes.title.text

        body = []
        for shape in slide.shapes:
            if shape == slide.shapes.title:
                continue
            if shape.has_text_frame:
                body.append(shape.text)
            elif shape.has_table:
                for row in shape.table.rows:
                    row_data = [cell.text for cell in row.cells]
                    body.append(" | ".join(row_data))
        
        body_text = "\n".join(body).strip()
        if not title and not body_text:
            continue
            
        embed_text = f"Slide {i+1}: {title}\n\n{body_text}".strip()
        meta = {
            "chunk_index": i,
            "source_type": "pptx",
            "content_type": "text",
            "page": i + 1
        }
        chunks.append(Chunk(embed_text=embed_text, payload_text=embed_text, metadata=meta))
    return chunks

def schema_aware_chunk(data: Any, doc_id: str) -> List[Chunk]:
    chunks = []
    
    if isinstance(data, list):
        items = data
    elif isinstance(data, dict):
        if len(data) == 1 and isinstance(list(data.values())[0], list):
            items = list(data.values())[0]
        else:
            items = [data]
    else:
        items = [data]
        
    for i, item in enumerate(items):
        json_str = json.dumps(item, indent=2, ensure_ascii=False)
        meta = {
            "chunk_index": i,
            "source_type": "json",
            "content_type": "json"
        }
        chunks.append(Chunk(embed_text=json_str, payload_text=json_str, metadata=meta))
        
    return chunks

def hierarchical_parent_child_chunk(text: str, doc_id: str) -> List[Chunk]:
    parent_id = write_parent(doc_id, text)
    
    from langchain_text_splitters import RecursiveCharacterTextSplitter
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=50,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    texts = splitter.split_text(text)
    
    chunks = []
    for i, t in enumerate(texts):
        meta = {
            "chunk_index": i,
            "source_type": "pdf",
            "content_type": "text",
            "parent_id": parent_id
        }
        chunks.append(Chunk(embed_text=t, payload_text=t, metadata=meta))
        
    return chunks
