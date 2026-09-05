import argparse
import logging
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ingest")


def main():
    parser = argparse.ArgumentParser(description="Ingest documents into Pinecone")
    parser.add_argument("--dir", default=None, help="Data directory (default: from .env)")
    parser.add_argument("--force", action="store_true", help="Re-ingest all files")
    args = parser.parse_args()

    from config import settings
    settings.validate()

    from src.ingestor import ingest_directory

    logger.info("═" * 50)
    logger.info("  MultiRAG : Document Ingestion Pipeline")
    logger.info("═" * 50)

    results = ingest_directory(data_dir=args.dir, force=args.force)
    print("\n" + "-" * 60)
    print(f"{'FILE':<35} {'STATUS':<15} {'CHUNKS':>8}")
    print("-" * 60)
    for r in results:
        print(f"{r['filename'][:34]:<35} {r['status']:<15} {r['chunks_upserted']:>8}")
    print("-" * 60)

    success = sum(1 for r in results if r["status"] == "success")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    errors  = len(results) - success - skipped
    total   = sum(r["chunks_upserted"] for r in results)

    print(f"\nSuccess: {success} , Skipped: {skipped} , Errors: {errors}")
    print(f"  Total chunks upserted: {total}")
    print()


if __name__ == "__main__":
    main()
