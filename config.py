from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv(override=True)

@dataclass
class Settings:
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_index_name: str = field(default_factory=lambda: os.getenv("PINECONE_INDEX_NAME", "multirag"))
    pinecone_region: str = field(default_factory=lambda: os.getenv("PINECONE_REGION", "us-east-1"))

    ollama_base_url: str = field(default_factory=lambda: os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"))
    ollama_model: str = field(default_factory=lambda: os.getenv("OLLAMA_MODEL", "qwen3:8b"))

    embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2"
    embedding_dim: int = 384

    reranker_model: str = field(default_factory=lambda: os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"))
    reranker_candidates: int = field(default_factory=lambda: int(os.getenv("RERANKER_CANDIDATES", "25")))


    pdf_short_threshold: int = field(default_factory=lambda: int(os.getenv("PDF_SHORT_THRESHOLD", "12")))
    parent_store_path: str = field(default_factory=lambda: os.getenv("PARENT_STORE_PATH", "data/parent_store.json"))

    top_k: int = field(default_factory=lambda: int(os.getenv("TOP_K_RESULTS", "5")))

    hybrid_alpha: float = field(default_factory=lambda: float(os.getenv("HYBRID_ALPHA", "0.75")))
    bm25_model_path: str = field(default_factory=lambda: os.getenv("BM25_MODEL_PATH", "data/bm25_model.json"))

    memory_turn_window: int = field(default_factory=lambda: int(os.getenv("MEMORY_TURN_WINDOW", "5")))
    session_expiry_minutes: int = field(default_factory=lambda: int(os.getenv("SESSION_EXPIRY_MINUTES", "60")))

    data_dir: str = field(default_factory=lambda: os.getenv("DATA_DIR", "data"))
    registry_file: str = field(init=False)

    def __post_init__(self):
        self.registry_file = os.path.join(self.data_dir, ".ingested_registry.json")

    def validate(self) -> None:
        missing = []
        if not self.pinecone_api_key:
            missing.append("PINECONE_API_KEY")
        if missing:
            raise EnvironmentError(
                f"Missing required environment variables: {', '.join(missing)}\n"
                "Copy .env.example to .env and fill in the values."
            )

settings = Settings()
