import time
import uuid
from typing import Dict, List, Optional
import logging

from config import settings

logger = logging.getLogger(__name__)

_SESSIONS: Dict[str, Dict] = {}

def _evict_stale_sessions():
    now = time.time()
    expiry_seconds = settings.session_expiry_minutes * 60
    stale_keys = [
        sid for sid, data in _SESSIONS.items()
        if (now - data["last_accessed"]) > expiry_seconds
    ]
    for sid in stale_keys:
        del _SESSIONS[sid]
    if stale_keys:
        logger.info(f"[MEMORY] Evicted {len(stale_keys)} stale session(s).")

def get_or_create_session(session_id: Optional[str]) -> str:
    _evict_stale_sessions()
    
    if not session_id or session_id not in _SESSIONS:
        new_id = str(uuid.uuid4())
        _SESSIONS[new_id] = {"last_accessed": time.time(), "history": []}
        logger.info(f"[MEMORY] Created new session: {new_id}")
        return new_id
    
    _SESSIONS[session_id]["last_accessed"] = time.time()
    return session_id

def get_history(session_id: str) -> List[Dict]:
    _evict_stale_sessions()
    session = _SESSIONS.get(session_id)
    if session:
        session["last_accessed"] = time.time()
        return session["history"]
    return []

def add_turn(session_id: str, role: str, content: str, condensed_content: Optional[str] = None, sources: Optional[List] = None):
    if session_id not in _SESSIONS:
        logger.warning(f"[MEMORY] Cannot add turn to unknown session: {session_id}")
        return
        
    turn = {
        "role": role,
        "content": content,
        "timestamp": time.time()
    }
    if condensed_content is not None:
        turn["condensed_content"] = condensed_content
    if sources is not None:
        turn["sources"] = sources
        
    _SESSIONS[session_id]["history"].append(turn)
    _SESSIONS[session_id]["last_accessed"] = time.time()
