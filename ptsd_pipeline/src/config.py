"""
config.py
=========
Centralised path and hyperparameter constants for the unified Project Sasha pipeline.

Every stage imports its paths and numeric constants from here so they stay consistent
across the full run_pipeline.py orchestration and when any stage is re-run standalone.

All comments and docstrings are in English.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Directory layout
# ---------------------------------------------------------------------------

_SRC_DIR = Path(__file__).parent
PROJECT_ROOT = _SRC_DIR.parent

DATA_DIR      = PROJECT_ROOT / "data"
VISUALS_DIR   = PROJECT_ROOT / "visuals"
REPORTS_DIR   = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
LOGS_DIR      = PROJECT_ROOT / "logs"

# Ensure output directories exist whenever config is imported
for _d in (DATA_DIR, VISUALS_DIR, REPORTS_DIR, ARTIFACTS_DIR, LOGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

LOG_PATH = LOGS_DIR / "pipeline_run.log"

# ---------------------------------------------------------------------------
# Stage 1 — Data Generation
# ---------------------------------------------------------------------------

DATASET_OUTPUT_PATH: Path = DATA_DIR / "dataset.json"

# ---------------------------------------------------------------------------
# Stage 2 — Quality Judge (G-Eval)
# ---------------------------------------------------------------------------

CLEAN_DATASET_PATH: Path  = DATA_DIR      / "dataset.clean.json"
QUALITY_METRICS_PATH: Path = ARTIFACTS_DIR / "quality_metrics.json"
JUDGE_CACHE_PATH: Path     = ARTIFACTS_DIR / "judge_cache.json"
JUDGE_FEEDBACK_PATH: Path  = ARTIFACTS_DIR / "judge_feedback.json"

# ---------------------------------------------------------------------------
# Stage 3 — EDA
# ---------------------------------------------------------------------------

EDA_TABLES_PATH: Path = ARTIFACTS_DIR / "eda_tables.json"

# ---------------------------------------------------------------------------
# Stage 4 — Iterative Stratification Split
# ---------------------------------------------------------------------------

TRAIN_DATASET_PATH: Path   = DATA_DIR      / "train_dataset.json"
TEST_DATASET_PATH: Path    = DATA_DIR      / "test_dataset.json"
SPLIT_MANIFEST_PATH: Path  = ARTIFACTS_DIR / "split_manifest.json"

# ---------------------------------------------------------------------------
# Stage 5 — Modeling
# ---------------------------------------------------------------------------

# Fraction of total held out for stratified test (stage 4)
STRAT_TEST_SIZE: float = 0.25

# Internal validation carve-out from the training fold (for threshold tuning)
VAL_SIZE: float = 0.15
TEST_SIZE: float = 0.15  # kept for backward-compat reference only

# Batch size placeholder for future neural / LLM inference stages
BATCH_SIZE: int = 32

# TF-IDF settings tuned for Hebrew (right-to-left, no built-in stopwords)
TFIDF_CONFIG: dict[str, Any] = {
    "analyzer": "word",
    "ngram_range": (1, 2),
    "min_df": 2,
    "max_df": 0.95,
    "sublinear_tf": True,       # Apply log(1+tf) scaling
    "encoding": "utf-8",
    "decode_error": "replace",
}

LR_CONFIG: dict[str, Any] = {
    "max_iter": 1000,
    "solver": "lbfgs",
    "C": 1.0,
    "class_weight": "balanced",   # Compensates for label-frequency imbalance
    "random_state": 42,           # RANDOM_SEED value
}

EVAL_RESULTS_PATH: Path = ARTIFACTS_DIR / "eval_results.json"

# Zero-shot LLM baseline settings — mDeBERTa-v3 is multilingual (Hebrew-capable)
ZS_MODEL: str = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"
ZS_THRESHOLD: float = 0.5
ZS_BATCH_SIZE: int = 8

# ---------------------------------------------------------------------------
# Stage 6 — Report
# ---------------------------------------------------------------------------

SLIDE3_PATH: Path     = REPORTS_DIR / "slide3_summary.md"
README_EDA_PATH: Path = REPORTS_DIR / "README_eda.md"

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

RANDOM_SEED: int = 42      # used by modeling internal train/val split and sklearn
STRAT_SEED: int = 1240     # used by iterative stratification (split.py) and EDA
