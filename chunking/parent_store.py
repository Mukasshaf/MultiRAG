import json
import logging
from pathlib import Path
from typing import Dict
import hashlib
from config import settings

logger = logging.getLogger(__name__)

def _get_store_path() -> Path:
    path = Path(settings.parent_store_path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def _load() -> Dict[str, str]:
    path = _get_store_path()
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"[PARENT_STORE] Error loading: {e}")
        return {}

def _save(data: Dict[str, str]) -> None:
    path = _get_store_path()
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"[PARENT_STORE] Error saving: {e}")

def write_parent(doc_id: str, text: str) -> str:
    parent_id = hashlib.sha256(f"{doc_id}:{text}".encode()).hexdigest()[:16]
    store = _load()
    if parent_id not in store:
        store[parent_id] = text
        _save(store)
    return parent_id

def read_parent(parent_id: str) -> str:
    store = _load()
    return store.get(parent_id, "")
