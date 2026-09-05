"""
scripts/fit_bm25.py — Fit and save BM25 model on current Pinecone corpus.

Run this after re-ingesting with --force, or whenever you want to refresh the
BM25 vocabulary from the ingestion tracker registry.

Usage:
    uv run python scripts/fit_bm25.py
    uv run python scripts/fit_bm25.py --dir data/documents
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("fit_bm25")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit BM25 model on ingested corpus")
    parser.add_argument("--dir", default=None, help="Data directory (default: from .env)")
    args = parser.parse_args()

    from config import settings
    settings.validate()

    from src.document_loader import load_file, SUPPORTED_EXTENSIONS
    from src.chunker import chunk_documents
    from src.bm25_encoder import fit_and_save, reset_encoder

    dir_path = Path(args.dir or settings.data_dir).resolve()
    logger.info(f"Scanning: {dir_path}")

    all_texts: list[str] = []
    for ext in SUPPORTED_EXTENSIONS:
        for fp in sorted(dir_path.rglob(f"*{ext}")):
            if fp.name.startswith(".") or "eval" in fp.parts:
                continue
            docs = load_file(fp)
            if not docs:
                continue
            chunks = chunk_documents(docs)
            all_texts.extend(c.page_content for c in chunks)
            logger.info(f"  {fp.name}: {len(chunks)} chunks")

    if not all_texts:
        logger.error("No text found. Make sure documents are in the data directory.")
        sys.exit(1)

    logger.info(f"Total chunks collected: {len(all_texts)}")
    reset_encoder()
    fit_and_save(all_texts)
    logger.info(f"BM25 model saved to: {settings.bm25_model_path}")
    logger.info("Done. Re-ingest with --force to add sparse vectors to Pinecone.")


if __name__ == "__main__":
    main()
