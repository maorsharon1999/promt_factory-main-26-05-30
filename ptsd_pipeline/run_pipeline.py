"""
run_pipeline.py
===============
Unified entry point for the full Project Sasha NLP pipeline.

Sequential stages
-----------------
  1. Data Generation   (src/data_generation.py)  → data/dataset.json
  2. Quality Judge     (src/quality_judge.py)     → data/dataset.clean.json + quality_metrics.json
  3. EDA               (src/eda.py)               → visuals/*.png + eda_tables.json
  4. Stratified Split  (src/splitting.py)         → data/train_dataset.json + test_dataset.json + split_manifest.json
  5. ML Baseline       (src/modeling.py)          → TF-IDF + LR + Zero-shot evaluation
  6. Report            (src/report.py)            → slide3_summary.md + README_eda.md

Usage
-----
    python run_pipeline.py                    # run all stages end-to-end
    python run_pipeline.py --skip-generation  # skip stage 1 (data already exists)
    python run_pipeline.py --skip-judge       # skip stage 2 (clean data already exists)
    python run_pipeline.py --no-zero-shot     # skip zero-shot LLM baseline in stage 5
    python run_pipeline.py --stages 3 4 5     # run only EDA + split + modeling
    python run_pipeline.py --mock             # use MockLLMClient (no Ollama / API required)

All paths are centralised in src/config.py.
Logging is configured once here (console INFO + pipeline_run.log DEBUG).
Each stage failure aborts the run with a logged exception.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to sys.path so gpu_check.py is importable
_HERE = Path(__file__).parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from src.logging_setup import configure_logging
from src.config import LOG_PATH, DATASET_OUTPUT_PATH, CLEAN_DATASET_PATH, QUALITY_METRICS_PATH, JUDGE_CACHE_PATH, JUDGE_FEEDBACK_PATH

configure_logging(LOG_PATH)
logger = logging.getLogger(__name__)


def _banner(text: str) -> None:
    sep = "=" * 62
    logger.info(sep)
    logger.info("  %s", text)
    logger.info(sep)


def run_stage_1(mock: bool = False) -> None:
    """Stage 1 — Generate synthetic Hebrew dataset via LLM."""
    _banner("STAGE 1 — Synthetic Data Generation")
    from src.data_generation import (
        LLMProvider, LLMClient,
        create_llm_client, MockLLMClient, generate_dataset,
        DATASET_OUTPUT_PATH as _OUT,
    )
    import os

    if mock:
        logger.warning("--mock flag set — using MockLLMClient (deterministic fake Hebrew).")
        llm: LLMClient = MockLLMClient()
    elif os.environ.get("GEMINI_API_KEY"):
        logger.info("GEMINI_API_KEY found — using Gemini model (gemini-1.5-flash).")
        llm = create_llm_client(
            provider=LLMProvider.GEMINI,
            model_name="gemini-1.5-flash",
        )
    elif os.environ.get("OPENROUTER_API_KEY"):
        logger.info("Gemini key not found — falling back to OpenRouter (free tier).")
        llm = create_llm_client(
            provider=LLMProvider.OPENROUTER,
            model_name="mistralai/mistral-7b-instruct:free",
        )
    else:
        logger.warning(
            "No LLM provider available. Falling back to MockLLMClient. "
            "Set GEMINI_API_KEY to generate real data."
        )
        llm = MockLLMClient()

    _OUT.parent.mkdir(parents=True, exist_ok=True)
    examples = generate_dataset(llm, output_path=str(_OUT))
    logger.info("Stage 1 complete — %d examples written to %s.", len(examples), _OUT)


def run_stage_2(mock: bool = False) -> None:
    """Stage 2 — G-Eval LLM-as-a-Judge quality filter."""
    _banner("STAGE 2 — Quality Judge (G-Eval)")
    from src.quality_judge import run_judge_pipeline

    if not DATASET_OUTPUT_PATH.exists():
        raise FileNotFoundError(
            f"Stage 2 requires '{DATASET_OUTPUT_PATH}'. Run Stage 1 first."
        )

    run_judge_pipeline(
        input_path=DATASET_OUTPUT_PATH,
        output_path=CLEAN_DATASET_PATH,
        metrics_path=QUALITY_METRICS_PATH,
        cache_path=JUDGE_CACHE_PATH,
        feedback_path=JUDGE_FEEDBACK_PATH,
        mock=mock,
    )
    logger.info("Stage 2 complete.")


def run_stage_3() -> None:
    """Stage 3 — EDA charts + numeric tables."""
    _banner("STAGE 3 — Exploratory Data Analysis")
    from src.eda import run_eda_pipeline
    from src.config import CLEAN_DATASET_PATH as _INPUT, EDA_TABLES_PATH as _TABLES

    run_eda_pipeline(input_path=_INPUT, tables_path=_TABLES)
    logger.info("Stage 3 complete.")


def run_stage_4() -> None:
    """Stage 4 — Iterative stratification split."""
    _banner("STAGE 4 — Iterative Stratification Split")
    from src.splitting import run_split_pipeline
    from src.config import (
        CLEAN_DATASET_PATH as _IN,
        TRAIN_DATASET_PATH as _TRAIN,
        TEST_DATASET_PATH as _TEST,
        SPLIT_MANIFEST_PATH as _MANIFEST,
    )

    if not _IN.exists():
        raise FileNotFoundError(
            f"Stage 4 requires '{_IN}'. Run Stages 1 + 2 first."
        )

    run_split_pipeline(
        input_path=_IN,
        train_path=_TRAIN,
        test_path=_TEST,
        manifest_path=_MANIFEST,
    )
    logger.info("Stage 4 complete.")


def run_stage_5(device=None) -> None:
    """Stage 5 — TF-IDF + LR baseline + Zero-shot evaluation."""
    _banner("STAGE 5 — ML Baseline Training & Evaluation")
    from src.modeling import run_baseline

    run_baseline(device=device)
    logger.info("Stage 5 complete.")


def run_stage_6() -> None:
    """Stage 6 — Generate Slide 3 summary and README_eda.md."""
    _banner("STAGE 6 — Report Generation")
    from src.report import run_report_pipeline

    # EDA already ran in Stage 3; pass run_eda=False to skip re-running charts
    run_report_pipeline(run_eda=False)
    logger.info("Stage 6 complete.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Project Sasha — Unified PTSD Hebrew NLP Pipeline"
    )
    parser.add_argument(
        "--stages", nargs="+", type=int, choices=range(1, 7), metavar="N",
        help="Run only specified stages (e.g. --stages 3 4 5)",
    )
    parser.add_argument(
        "--skip-generation", action="store_true",
        help="Skip Stage 1 (assume data/dataset.json already exists)",
    )
    parser.add_argument(
        "--skip-judge", action="store_true",
        help="Skip Stage 2 (assume data/dataset.clean.json already exists)",
    )
    parser.add_argument(
        "--mock", action="store_true",
        help="Use MockLLMClient for stages 1 and 2 (no Ollama / API required)",
    )
    args = parser.parse_args()

    # Determine which stages to run
    if args.stages:
        active_stages = set(args.stages)
    else:
        active_stages = {1, 2, 3, 4, 5, 6}
        if args.skip_generation:
            active_stages.discard(1)
        if args.skip_judge:
            active_stages.discard(2)

    # Hardware check — printed once at the very beginning
    _banner("HARDWARE CHECK")
    from gpu_check import get_device
    device = get_device()

    stage_fns = {
        1: lambda: run_stage_1(mock=args.mock),
        2: lambda: run_stage_2(mock=args.mock),
        3: run_stage_3,
        4: run_stage_4,
        5: lambda: run_stage_5(device=device),
        6: run_stage_6,
    }

    for stage_num in sorted(active_stages):
        try:
            stage_fns[stage_num]()
        except Exception:
            logger.exception("Pipeline aborted at Stage %d.", stage_num)
            sys.exit(1)

    _banner("PIPELINE COMPLETE")
    logger.info("All stages finished. Log: %s", LOG_PATH)


if __name__ == "__main__":
    main()
