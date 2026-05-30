"""
modeling.py
===========
STEP 5 — ML Baseline Training & Evaluation.
Ported from src/baseline_pipeline.py.

Differences from source:
  - load_data() reads pre-split train_dataset.json + test_dataset.json produced by
    splitting.py (iterative stratification), instead of performing its own random
    train_test_split on the full dataset.
  - preprocess() carves a validation fold out of the train set only; the test set
    is passed in directly from the splitting stage.
  - All print() calls converted to logger.
  - Path constants sourced from src.config.

Original function names, variable names, and docstrings are preserved verbatim.
All new comments are in English.

Pipeline stages
---------------
1. load_data()        – Read train and test JSON files produced by splitting.py
2. preprocess()       – Binarise labels, carve validation split from train set
3. train_baseline()   – TF-IDF + OneVsRestClassifier(LogisticRegression)
4. tune_thresholds()  – Find per-label decision thresholds on the validation set
5. evaluate()         – Micro / Macro Precision, Recall, F1 + per-label report
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# gpu_check.py lives in the project root; make it importable when run directly
_PROJECT_ROOT = Path(__file__).parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))
from gpu_check import get_device  # noqa: E402  (intentional late import)

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer

from src.config import (
    TRAIN_DATASET_PATH,
    TEST_DATASET_PATH,
    VAL_SIZE,
    BATCH_SIZE,
    TFIDF_CONFIG,
    LR_CONFIG,
    ZS_MODEL,
    ZS_THRESHOLD,
    ZS_BATCH_SIZE,
    RANDOM_SEED,
    EVAL_RESULTS_PATH,
)

# ---------------------------------------------------------------------------
# Configuration — module-level aliases for backward compat / standalone use
# ---------------------------------------------------------------------------

BATCH_SIZE: int = BATCH_SIZE      # type: ignore[assignment]
TFIDF_CONFIG: dict[str, Any] = TFIDF_CONFIG  # type: ignore[assignment]
LR_CONFIG: dict[str, Any] = LR_CONFIG  # type: ignore[assignment]
ZS_MODEL: str = ZS_MODEL          # type: ignore[assignment]
ZS_THRESHOLD: float = ZS_THRESHOLD  # type: ignore[assignment]
ZS_BATCH_SIZE: int = ZS_BATCH_SIZE  # type: ignore[assignment]
VAL_SIZE: float = VAL_SIZE        # type: ignore[assignment]
TEST_SIZE: float = 0.15           # kept for reference; not used in three-way split path


# ---------------------------------------------------------------------------
# Stage 1: Data Loading
# ---------------------------------------------------------------------------

def load_data(
    train_path: Path = TRAIN_DATASET_PATH,
    test_path: Path = TEST_DATASET_PATH,
) -> tuple[list[dict], list[dict]]:
    """
    Load the train and test JSON datasets produced by splitting.py.

    Returns
    -------
    tuple[list[dict], list[dict]]
        ``(train_data, test_data)`` — each element is a sample dict with at
        minimum the keys ``"text"`` (str) and ``"labels"`` (list[str]).

    Raises
    ------
    FileNotFoundError
        If either dataset file does not exist.
    """
    for path in (train_path, test_path):
        if not path.exists():
            raise FileNotFoundError(
                f"Dataset not found at '{path}'. "
                "Run 'python run_pipeline.py' or src/splitting.py first to produce the split."
            )

    with train_path.open(encoding="utf-8") as fh:
        train_data = json.load(fh)
    with test_path.open(encoding="utf-8") as fh:
        test_data = json.load(fh)

    logger.info(
        "Loaded %d train + %d test samples from '%s' / '%s'.",
        len(train_data), len(test_data), train_path.name, test_path.name,
    )
    return train_data, test_data


# ---------------------------------------------------------------------------
# Stage 2: Preprocessing
# ---------------------------------------------------------------------------

def preprocess(
    train_data: list[dict],
    test_data: list[dict],
    val_size: float = VAL_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[
    list[str], list[str], list[str],        # texts_train, texts_val, texts_test
    np.ndarray, np.ndarray, np.ndarray,     # y_train, y_val, y_test
    MultiLabelBinarizer,                    # fitted binariser (needed for label names)
]:
    """
    Extract texts and labels from the raw data, binarise the label lists,
    and perform a two-way train / val split on the training fold.

    The test set is already held out by splitting.py; the validation set is
    carved from the train fold here and used exclusively for threshold tuning.

    Parameters
    ----------
    train_data : list[dict]
        Training samples from splitting stage.
    test_data : list[dict]
        Test samples from splitting stage.
    val_size : float
        Fraction of training samples reserved for validation (default 0.15).
    random_seed : int
        Seed for reproducible splits.

    Returns
    -------
    tuple
        ``(texts_train, texts_val, texts_test, y_train, y_val, y_test, mlb)``
    """
    texts_all_train: list[str] = [s["text"] for s in train_data]
    labels_all_train: list[list[str]] = [s["labels"] for s in train_data]
    texts_test: list[str] = [s["text"] for s in test_data]
    labels_test: list[list[str]] = [s["labels"] for s in test_data]

    # Fit binariser on training labels; transform test labels with same schema
    mlb = MultiLabelBinarizer()
    y_all_train: np.ndarray = mlb.fit_transform(labels_all_train)
    y_test: np.ndarray = mlb.transform(labels_test)

    # Carve validation set from training fold
    texts_train, texts_val, y_train, y_val = train_test_split(
        texts_all_train, y_all_train,
        test_size=val_size,
        random_state=random_seed,
    )

    logger.info(
        "Train: %d | Val: %d | Test: %d | Labels (%d): %s",
        len(texts_train), len(texts_val), len(texts_test),
        len(mlb.classes_), list(mlb.classes_),
    )
    return texts_train, texts_val, texts_test, y_train, y_val, y_test, mlb


# ---------------------------------------------------------------------------
# Stage 3: Baseline Training
# ---------------------------------------------------------------------------

def train_baseline(
    texts_train: list[str],
    y_train: np.ndarray,
) -> tuple[TfidfVectorizer, OneVsRestClassifier]:
    """
    Build and fit the classical ML baseline:
    TF-IDF (uni+bigram) → OneVsRestClassifier(LogisticRegression).

    Parameters
    ----------
    texts_train : list[str]
        Hebrew training texts.
    y_train : np.ndarray
        Binary label matrix for training samples.

    Returns
    -------
    tuple
        ``(vectorizer, classifier)`` — both already fitted.
    """
    vectorizer = TfidfVectorizer(**TFIDF_CONFIG)
    X_train = vectorizer.fit_transform(texts_train)

    classifier = OneVsRestClassifier(
        LogisticRegression(**LR_CONFIG),
        n_jobs=-1,
    )
    classifier.fit(X_train, y_train)

    logger.info(
        "Fitted TF-IDF (%d features) + OneVsRest(LogisticRegression).",
        X_train.shape[1],
    )
    return vectorizer, classifier


# ---------------------------------------------------------------------------
# Stage 4a: Threshold Tuning (on validation set)
# ---------------------------------------------------------------------------

def tune_thresholds(
    texts_val: list[str],
    y_val: np.ndarray,
    vectorizer: TfidfVectorizer,
    classifier: OneVsRestClassifier,
    thresholds: list[float] | None = None,
) -> np.ndarray:
    """
    Find the per-label decision threshold that maximises F1 on the
    **validation set**. The resulting thresholds are then passed to
    :func:`evaluate` so the test set is never used for tuning.

    Parameters
    ----------
    texts_val : list[str]
        Hebrew validation texts.
    y_val : np.ndarray
        Ground-truth binary label matrix for the validation set.
    vectorizer : TfidfVectorizer
        Fitted vectoriser.
    classifier : OneVsRestClassifier
        Fitted classifier (must support ``predict_proba``).
    thresholds : list[float] | None
        Candidate threshold values to search over.
        Defaults to ``[0.1, 0.2, ..., 0.9]``.

    Returns
    -------
    np.ndarray
        1-D array of shape ``(n_labels,)`` with the best threshold per label.
    """
    if thresholds is None:
        thresholds = [round(t * 0.1, 1) for t in range(1, 10)]  # 0.1 … 0.9

    X_val = vectorizer.transform(texts_val)
    # predict_proba on OneVsRestClassifier returns (n_samples, n_labels)
    proba = classifier.predict_proba(X_val)

    n_labels = y_val.shape[1]
    best_thresholds = np.full(n_labels, 0.5)  # sensible default

    for label_idx in range(n_labels):
        best_f1 = -1.0
        for t in thresholds:
            y_pred_col = (proba[:, label_idx] >= t).astype(int)
            score = f1_score(y_val[:, label_idx], y_pred_col, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_thresholds[label_idx] = t

    logger.info(
        "Best per-label thresholds: %s",
        ", ".join(f"{t:.1f}" for t in best_thresholds),
    )
    return best_thresholds


# ---------------------------------------------------------------------------
# Stage 4b: Evaluation (on test set)
# ---------------------------------------------------------------------------

def evaluate(
    texts_test: list[str],
    y_test: np.ndarray,
    vectorizer: TfidfVectorizer,
    classifier: OneVsRestClassifier,
    mlb: MultiLabelBinarizer,
    thresholds: np.ndarray | None = None,
    model_name: str = "TF-IDF + LR Baseline",
) -> dict[str, float]:
    """
    Evaluate the fitted model on the test set and print a summary table.

    Predictions are made by applying per-label *thresholds* (tuned on the
    validation set) to the classifier's probability outputs. If no thresholds
    are supplied, the default 0.5 cut-off is used for every label.

    Metrics reported
    ----------------
    * Micro / Macro Precision
    * Micro / Macro Recall
    * Micro / Macro F1-Score
    * Per-label breakdown via ``classification_report``

    Parameters
    ----------
    texts_test : list[str]
        Hebrew test texts.
    y_test : np.ndarray
        Ground-truth binary label matrix.
    vectorizer : TfidfVectorizer
        Fitted vectoriser.
    classifier : OneVsRestClassifier
        Fitted classifier.
    mlb : MultiLabelBinarizer
        Fitted binariser (provides human-readable label names).
    thresholds : np.ndarray | None
        Per-label thresholds from :func:`tune_thresholds`. Falls back to
        0.5 for every label when ``None``.
    model_name : str
        Display name used in the printed table header.

    Returns
    -------
    dict[str, float]
        Flat dictionary of all computed metric values.
    """
    X_test = vectorizer.transform(texts_test)
    proba = classifier.predict_proba(X_test)

    if thresholds is None:
        thresholds = np.full(y_test.shape[1], 0.5)

    # Apply per-label thresholds instead of the hard 0.5 default
    y_pred = (proba >= thresholds).astype(int)

    def _score(fn, average):
        return fn(y_test, y_pred, average=average, zero_division=0)

    metrics = {
        "precision_micro":  _score(precision_score, "micro"),
        "precision_macro":  _score(precision_score, "macro"),
        "recall_micro":     _score(recall_score,    "micro"),
        "recall_macro":     _score(recall_score,    "macro"),
        "f1_micro":         _score(f1_score,        "micro"),
        "f1_macro":         _score(f1_score,        "macro"),
    }

    sep = "=" * 62
    logger.info(sep)
    logger.info("  EVALUATION RESULTS — %s", model_name)
    logger.info(sep)
    logger.info("  %-28s%10s%10s", "Metric", "Micro", "Macro")
    logger.info("  %s", "-" * 48)
    for base in ("precision", "recall", "f1"):
        lbl = base.capitalize()
        micro = metrics[f"{base}_micro"]
        macro = metrics[f"{base}_macro"]
        logger.info("  %-28s%10.4f%10.4f", lbl, micro, macro)
    logger.info(sep)

    logger.info("  Per-label Classification Report:")
    report = classification_report(
        y_test, y_pred,
        target_names=mlb.classes_,
        zero_division=0,
    )
    for line in report.splitlines():
        logger.info("  %s", line)

    return metrics


# ---------------------------------------------------------------------------
# Placeholder: Zero-shot LLM Baseline
# ---------------------------------------------------------------------------

def zero_shot_llm_baseline(
    texts_test: list[str],
    y_test: np.ndarray,
    mlb: MultiLabelBinarizer,
    model_id: str = ZS_MODEL,
    threshold: float = ZS_THRESHOLD,
    batch_size: int = ZS_BATCH_SIZE,
    device=None,           # torch.device — injected by main(); None → CPU
) -> dict[str, float]:
    """
    Zero-shot multi-label classification using a multilingual NLI model.

    Uses HuggingFace ``zero-shot-classification`` pipeline with
    ``MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`` — a multilingual DeBERTa
    model fine-tuned on MNLI + XNLI that supports Hebrew out of the box.

    Each text is scored against every candidate label via textual entailment.
    Scores >= *threshold* are predicted as present (multi_label=True).

    The result is stored in ``y_pred_zeroshot`` and evaluated with the same
    Micro/Macro metrics used for the TF-IDF baseline.

    Install dependency if missing::

        pip install transformers torch

    Returns
    -------
    dict[str, float]
        Same metric keys as :func:`evaluate`, or empty dict on failure.
    """
    try:
        from transformers import pipeline as hf_pipeline
    except ImportError:
        logger.warning(
            "'transformers' not installed. Run: pip install transformers torch"
        )
        return {}

    labels: list[str] = list(mlb.classes_)
    logger.info("Loading '%s' (first run downloads ~550 MB) …", model_id)

    try:
        # Map torch.device → HuggingFace device index (-1 = CPU, 0 = first GPU)
        _hf_device = 0 if (device is not None and device.type == "cuda") else -1
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            zs_pipe = hf_pipeline(
                "zero-shot-classification",
                model=model_id,
                device=_hf_device,
            )
    except Exception as exc:
        logger.error("Failed to load model: %s", exc)
        return {}

    logger.info("Running inference on %d samples …", len(texts_test))

    # ---- Inference loop ----
    # y_pred_zeroshot shape: (n_test_samples, n_labels) — same as y_test
    y_pred_zeroshot = np.zeros((len(texts_test), len(labels)), dtype=int)

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for i in range(0, len(texts_test), batch_size):
            batch = texts_test[i : i + batch_size]
            results = zs_pipe(batch, candidate_labels=labels, multi_label=True)
            # pipeline returns a dict when batch size is 1, list otherwise
            if isinstance(results, dict):
                results = [results]
            for j, result in enumerate(results):
                # Map label → score, then threshold
                score_map: dict[str, float] = dict(zip(result["labels"], result["scores"]))
                for k, label in enumerate(labels):
                    if score_map.get(label, 0.0) >= threshold:
                        y_pred_zeroshot[i + j, k] = 1
            done = min(i + batch_size, len(texts_test))
            logger.info("  [%4d/%d] samples processed", done, len(texts_test))

    # ---- Evaluation (identical structure to evaluate()) ----
    def _score(fn, average):
        return fn(y_test, y_pred_zeroshot, average=average, zero_division=0)

    metrics = {
        "precision_micro": _score(precision_score, "micro"),
        "precision_macro": _score(precision_score, "macro"),
        "recall_micro":    _score(recall_score,    "micro"),
        "recall_macro":    _score(recall_score,    "macro"),
        "f1_micro":        _score(f1_score,        "micro"),
        "f1_macro":        _score(f1_score,        "macro"),
    }

    model_name = f"Zero-Shot ({model_id.split('/')[-1]})"
    sep = "=" * 62
    logger.info(sep)
    logger.info("  EVALUATION RESULTS — %s", model_name)
    logger.info(sep)
    logger.info("  %-28s%10s%10s", "Metric", "Micro", "Macro")
    logger.info("  %s", "-" * 48)
    for base in ("precision", "recall", "f1"):
        lbl = base.capitalize()
        logger.info(
            "  %-28s%10.4f%10.4f",
            lbl, metrics[f"{base}_micro"], metrics[f"{base}_macro"],
        )
    logger.info(sep)

    logger.info("  Per-label Classification Report:")
    report = classification_report(
        y_test, y_pred_zeroshot,
        target_names=mlb.classes_,
        zero_division=0,
    )
    for line in report.splitlines():
        logger.info("  %s", line)

    return metrics


# ---------------------------------------------------------------------------
# Callable stage function for the pipeline orchestrator
# ---------------------------------------------------------------------------

def run_baseline(device=None) -> None:
    """
    Full modeling pass: load split data, preprocess, train, tune, evaluate.

    Parameters
    ----------
    device : torch.device | None
        GPU/CPU device forwarded to zero_shot_llm_baseline. Resolved by
        the pipeline orchestrator before calling this function.
    """
    train_data, test_data = load_data()

    texts_train, texts_val, texts_test, y_train, y_val, y_test, mlb = preprocess(
        train_data, test_data
    )

    vectorizer, classifier = train_baseline(texts_train, y_train)

    # 4a. Evaluate on train and val sets (threshold = 0.5) to check fit / generalisation
    train_metrics = evaluate(texts_train, y_train, vectorizer, classifier, mlb, model_name="TF-IDF + LR — Train set")
    val_metrics   = evaluate(texts_val,   y_val,   vectorizer, classifier, mlb, model_name="TF-IDF + LR — Validation set")

    # 4b. Tune per-label thresholds on the validation set
    thresholds = tune_thresholds(texts_val, y_val, vectorizer, classifier)

    # 4c. Evaluate on the held-out test set using tuned thresholds
    tfidf_metrics = evaluate(
        texts_test, y_test, vectorizer, classifier, mlb, thresholds,
        model_name="TF-IDF + LR — Test set",
    )

    # 5. Zero-shot LLM baseline on the same test set (uses GPU if available)
    zeroshot_metrics = zero_shot_llm_baseline(texts_test, y_test, mlb, device=device)

    # 6. Side-by-side summary
    sep = "=" * 62
    logger.info(sep)
    logger.info("  FINAL COMPARISON SUMMARY")
    logger.info(sep)
    rows = [
        ("Precision", "precision_micro", "precision_macro"),
        ("Recall",    "recall_micro",    "recall_macro"),
        ("F1-Score",  "f1_micro",        "f1_macro"),
    ]
    logger.info("  %-18s %6s  %10s  %10s", "Metric", "", "TF-IDF", "Zero-Shot")
    logger.info("  %s", "-" * 50)
    for label, micro_key, macro_key in rows:
        tfidf_micro  = tfidf_metrics.get(micro_key, float("nan"))
        tfidf_macro  = tfidf_metrics.get(macro_key, float("nan"))
        zs_micro     = zeroshot_metrics.get(micro_key, float("nan"))
        zs_macro     = zeroshot_metrics.get(macro_key, float("nan"))
        logger.info("  %-18s %6s  %10.4f  %10.4f", label, "Micro", tfidf_micro, zs_micro)
        logger.info("  %-18s %6s  %10.4f  %10.4f", "", "Macro", tfidf_macro, zs_macro)
        logger.info("  %s", "-" * 50)
    logger.info(sep)

    # 7. Save all evaluation results to JSON
    eval_output = {
        "tfidf_train": train_metrics,
        "tfidf_val": val_metrics,
        "tfidf_test": tfidf_metrics,
        "zeroshot_test": zeroshot_metrics,
    }
    with EVAL_RESULTS_PATH.open("w", encoding="utf-8") as fh:
        json.dump(eval_output, fh, indent=2, ensure_ascii=False)
    logger.info("Evaluation results saved to '%s'.", EVAL_RESULTS_PATH)


def main() -> None:
    """Standalone entry point — resolves device then runs the full baseline."""
    device = get_device()
    run_baseline(device=device)


if __name__ == "__main__":
    main()
