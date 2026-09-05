from dataclasses import dataclass, field
from typing import Dict, Any, Optional

@dataclass
class Chunk:
    embed_text: str
    payload_text: str
    metadata: Dict[str, Any] = field(default_factory=dict)
