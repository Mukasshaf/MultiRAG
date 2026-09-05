from typing import Any, List
from config import settings
from .models import Chunk
from .strategies import (
    recursive_fallback_chunk,
    header_ancestor_chunk,
    slide_node_chunk,
    schema_aware_chunk,
    hierarchical_parent_child_chunk
)

def classify_pdf(page_count: int) -> str:
    threshold = getattr(settings, "pdf_short_threshold", 12)
    if page_count > threshold:
        return "textbook"
    return "short"

def chunk_document(file_type: str, content: Any, doc_id: str, page_count: int = 0) -> List[Chunk]:
    if file_type == "pdf":
        pdf_type = classify_pdf(page_count)
        if pdf_type == "textbook":
            return hierarchical_parent_child_chunk(content, doc_id)
        else:
            return recursive_fallback_chunk(content, doc_id, "pdf")
    
    elif file_type == "docx":
        return header_ancestor_chunk(content, doc_id, "docx")
    
    elif file_type == "md":
        return header_ancestor_chunk(content, doc_id, "md")
        
    elif file_type == "pptx":
        return slide_node_chunk(content, doc_id)
        
    elif file_type == "json":
        return schema_aware_chunk(content, doc_id)
        
    elif file_type == "txt":
        return recursive_fallback_chunk(content, doc_id, "txt")
        
    else:
        return recursive_fallback_chunk(content, doc_id, file_type)
