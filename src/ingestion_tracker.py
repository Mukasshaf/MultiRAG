from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any

from config import settings

logger = logging.getLogger(__name__)


def _load_registry() -> Dict[str, Any]:
    reg_path = Path(settings.registry_file)
    if reg_path.exists():
        try:
            with open(reg_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            logger.warning("[REGISTRY] Registry file corrupted — starting fresh.")
    return {}


def _save_registry(registry: Dict[str, Any]) -> None:
    reg_path = Path(settings.registry_file)
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    with open(reg_path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


def is_ingested(file_path: Path) -> bool:
    """
    Return True if the file has already been ingested and hasn't changed.
    Compares filename + mtime.
    """
    registry = _load_registry()
    key = file_path.name
    if key not in registry:
        return False
    try:
        stored_mtime = registry[key]["mtime"]
        current_mtime = file_path.stat().st_mtime
        return abs(stored_mtime - current_mtime) < 1.0
    except OSError:
        return False


def mark_ingested(file_path: Path) -> None:
    """Record the file as ingested in the registry."""
    registry = _load_registry()
    registry[file_path.name] = {
        "mtime": file_path.stat().st_mtime,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "size_bytes": file_path.stat().st_size,
    }
    _save_registry(registry)
    logger.info(f"[REGISTRY] Marked as ingested: {file_path.name}")


def unmark_ingested(filename: str) -> None:
    """Remove a file from the registry (used when deleting a document)."""
    registry = _load_registry()
    if filename in registry:
        del registry[filename]
        _save_registry(registry)
        logger.info(f"[REGISTRY] Removed from registry: {filename}")


def list_ingested() -> Dict[str, Any]:
    """Return the full registry as a dict."""
    return _load_registry()
