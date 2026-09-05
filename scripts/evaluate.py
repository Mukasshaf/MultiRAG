"""
Usage:
    uv run python scripts/evaluate.py                          # use default judge from .env
    uv run python scripts/evaluate.py --judge ollama           # local Ollama
    uv run python scripts/evaluate.py --judge openrouter       # OpenRouter
    uv run python scripts/evaluate.py --judge nvidia           # NVIDIA NIM
    uv run python scripts/evaluate.py --judge omnirouter       # OmniRouter
    uv run python scripts/evaluate.py --limit 5 --no-save      # quick smoke test
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ragas.embeddings.base import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.run_config import RunConfig
from langchain_community.embeddings import HuggingFaceEmbeddings as LCHuggingFaceEmbeddings
from config import settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")



eval_embeddings = LangchainEmbeddingsWrapper(
    LCHuggingFaceEmbeddings(model_name=settings.embedding_model)
)


def _build_llm_judge(provider: str) -> LangchainLLMWrapper:
    import os

    provider = provider.lower().strip()
    logger.info(f"[JUDGE] Building LLM judge using provider: '{provider}'")

    if provider == "ollama":
        from langchain_community.llms import Ollama
        model = os.getenv("RAGAS_JUDGE_MODEL") or settings.ollama_model
        logger.info(f"[JUDGE] Ollama model: {model}")
        return LangchainLLMWrapper(Ollama(model=model))

    from langchain_openai import ChatOpenAI

    if provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY", "")
        model   = os.getenv("RAGAS_JUDGE_MODEL") or os.getenv("OPENROUTER_JUDGE_MODEL", "google/gemini-flash-1.5")
        base_url = "https://openrouter.ai/api/v1"
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set in .env")
        logger.info(f"[JUDGE] OpenRouter model: {model}")
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
            default_headers={"HTTP-Referer": "https://github.com/Mukasshaf/MultiRAG"},
        )

    elif provider == "nvidia":
        api_key = os.getenv("NVIDIA_API_KEY", "")
        model   = os.getenv("RAGAS_JUDGE_MODEL") or os.getenv("NVIDIA_JUDGE_MODEL", "nvidia/nemotron-3.5-lightning-30b-a3b")
        base_url = "https://integrate.api.nvidia.com/v1"
        if not api_key:
            raise ValueError("NVIDIA_API_KEY not set in .env")
        logger.info(f"[JUDGE] NVIDIA NIM model: {model}")
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    elif provider == "omnirouter":
        api_key  = os.getenv("OMNIROUTER_API_KEY", "")
        model    = os.getenv("RAGAS_JUDGE_MODEL") or os.getenv("OMNIROUTER_JUDGE_MODEL", "gpt-4o-mini")
        base_url = os.getenv("OMNIROUTER_BASE_URL", "https://api.omnirouter.ai/v1")
        if not api_key:
            raise ValueError("OMNIROUTER_API_KEY not set in .env")
        logger.info(f"[JUDGE] OmniRouter model: {model} @ {base_url}")
        llm = ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=base_url,
        )

    else:
        raise ValueError(
            f"Unknown judge provider: '{provider}'. "
            "Choose one of: ollama, openrouter, nvidia, omnirouter"
        )

    return LangchainLLMWrapper(llm)


def _build_run_config(provider: str) -> RunConfig:
    """Return RunConfig tuned for the provider type."""
    if provider == "ollama":
        return RunConfig(timeout=600, max_retries=2, max_workers=1, max_wait=60)
    else:
        return RunConfig(timeout=600, max_retries=5, max_workers=4, max_wait=120)


def _safe_score(val) -> float | None:
    if val is None:
        return None
    if isinstance(val, list):
        nums = [float(x) for x in val if x is not None and not (isinstance(x, float) and math.isnan(x))]
        return round(sum(nums) / len(nums), 4) if nums else None
    try:
        f = float(val)
        return None if math.isnan(f) else round(f, 4)
    except (TypeError, ValueError):
        return None


def run_rag(question: str) -> tuple[str, list[str]]:
    from src.rag_chain import ask
    result = ask(question)
    answer = result.get("answer", "")
    contexts = [s.get("text", "") for s in result.get("sources", []) if s.get("text")]
    return answer, contexts


def main() -> None:
    import os
    default_judge = os.getenv("RAGAS_JUDGE", "ollama")

    parser = argparse.ArgumentParser(description="RAGAS evaluation for MultiRAG")
    parser.add_argument(
        "--golden",
        default="data/eval/golden_set.json",
        help="Path to golden set JSON",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only run the first N entries (for quick testing)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not append results to the history CSV",
    )
    parser.add_argument(
        "--judge",
        default=default_judge,
        choices=["ollama", "openrouter", "nvidia", "omnirouter"],
        help=f"LLM judge provider (default from .env RAGAS_JUDGE={default_judge!r})",
    )
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        logger.error(f"Golden set not found: {golden_path}")
        sys.exit(1)

    with open(golden_path, encoding="utf-8") as f:
        golden = json.load(f)

    if args.limit:
        golden = golden[: args.limit]

    logger.info(f"Loaded {len(golden)} eval entries from {golden_path}")

    try:
        from ragas import evaluate
        from ragas.metrics import (
            answer_relevancy,
            context_precision,
            context_recall,
            faithfulness,
        )
        from datasets import Dataset
    except ImportError:
        logger.error("ragas and datasets are required: uv add ragas datasets")
        sys.exit(1)

    try:
        llm_judge = _build_llm_judge(args.judge)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    run_config = _build_run_config(args.judge)

    questions, answers, contexts, ground_truths = [], [], [], []

    for i, entry in enumerate(golden, 1):
        q = entry["question"]
        gt = entry["ground_truth"]
        logger.info(f"[{i}/{len(golden)}] Running: {q[:70]}...")
        try:
            answer, ctx = run_rag(q)
        except Exception as e:
            logger.warning(f"  RAG failed for entry {i}: {e}")
            answer = ""
            ctx = []

        questions.append(q)
        answers.append(answer)
        contexts.append(ctx if ctx else [""])
        ground_truths.append(gt)

    logger.info(f"Running RAGAS evaluation (judge={args.judge}, workers={run_config.max_workers})...")
    eval_dataset = Dataset.from_dict(
        {
            "question": questions,
            "answer": answers,
            "contexts": contexts,
            "ground_truth": ground_truths,
        }
    )

    results = evaluate(
        eval_dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm_judge,
        embeddings=eval_embeddings,
        run_config=run_config,
    )

    scores = {
        "faithfulness":      _safe_score(results["faithfulness"]),
        "answer_relevancy":  _safe_score(results["answer_relevancy"]),
        "context_precision": _safe_score(results["context_precision"]),
        "context_recall":    _safe_score(results["context_recall"]),
    }

    print("\n" + "=" * 58)
    print(f"  RAGAS Evaluation Results  [judge: {args.judge}]")
    print("=" * 58)
    for metric, score in scores.items():
        label = metric.replace("_", " ").title().ljust(22)
        value = f"{score:.4f}" if score is not None else "N/A (timed out)"
        print(f"  {label}: {value}")
    print("=" * 58)

    timed_out = [k for k, v in scores.items() if v is None]
    if timed_out:
        logger.warning(
            f"Metrics that timed out: {timed_out}. "
            "Try a faster judge with --judge openrouter or increase timeout."
        )

    if not args.no_save:
        history_path = Path("data/eval/results_history.csv")
        history_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not history_path.exists()
        with open(history_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp", "judge", "num_entries",
                    "faithfulness", "answer_relevancy",
                    "context_precision", "context_recall", "notes",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": datetime.utcnow().isoformat(),
                    "judge": args.judge,
                    "num_entries": len(golden),
                    **scores,
                    "notes": "",
                }
            )
        logger.info(f"Results saved to {history_path}")


if __name__ == "__main__":
    main()
