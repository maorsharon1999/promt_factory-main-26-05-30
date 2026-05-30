"""
eda.py — Statistical multi-label EDA & visualization for PTSD Hebrew slang dataset.

Ported from parent eda.py.
Produces 8 PNGs under visuals/ and eda_tables.json.
All comments and docstrings are in English.
File I/O uses utf-8-sig encoding.
"""

from __future__ import annotations

import json
import logging
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)

# RTL Hebrew rendering helpers — applied to tick/legend labels only
try:
    import arabic_reshaper
    from bidi.algorithm import get_display as bidi_display

    def rtl(text: str) -> str:
        """Reshape and apply BiDi algorithm for correct Hebrew rendering in matplotlib."""
        return bidi_display(arabic_reshaper.reshape(text))

except ImportError:
    warnings.warn(
        "arabic_reshaper or python-bidi not installed. "
        "Hebrew labels may render incorrectly. "
        "Install: pip install arabic-reshaper python-bidi",
        stacklevel=2,
    )

    def rtl(text: str) -> str:  # type: ignore[misc]
        return text


import matplotlib
matplotlib.use("Agg")  # non-interactive backend for PNG export
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns

# ---------------------------------------------------------------------------
# Shared rcParams — Hebrew-safe font fallback chain
# ---------------------------------------------------------------------------

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Noto Sans Hebrew", "David CLM", "Arial", "DejaVu Sans"],
        "axes.titlesize": 13,
        "axes.labelsize": 11,
        "xtick.labelsize": 9,
        "ytick.labelsize": 9,
        "figure.dpi": 150,
    }
)

SEED = 1240

# ---------------------------------------------------------------------------
# Paths & constants — sourced from config for pipeline consistency
# ---------------------------------------------------------------------------

from src.config import PROJECT_ROOT, VISUALS_DIR, CLEAN_DATASET_PATH, EDA_TABLES_PATH  # noqa: E402

BASE_DIR = PROJECT_ROOT  # preserve original name used throughout this module

# Sanity thresholds for text length (plan §2.5)
_MIN_MEDIAN_WORDS = 8
_MIN_STD_WORDS = 3


# ---------------------------------------------------------------------------
# Save helper (plan §4.1)
# ---------------------------------------------------------------------------


def save_fig(fig: plt.Figure, name: str) -> Path:
    """Save figure at 300 DPI into visuals/ with tight bounding box."""
    VISUALS_DIR.mkdir(exist_ok=True)
    out = VISUALS_DIR / name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved -> %s", out)
    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(path: str | Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    """
    Load JSON dataset and binarize multi-labels.

    Returns:
        df:      raw DataFrame with all original columns
        Y:       binary label matrix (n_samples × n_labels)
        classes: ordered list of label names
    """
    with open(path, encoding="utf-8-sig") as f:
        records = json.load(f)

    df = pd.DataFrame(records)
    df["word_count"] = df["text"].apply(lambda t: len(str(t).split()))
    df["char_count"] = df["text"].apply(lambda t: len(str(t)))
    df["label_count"] = df["labels"].apply(len)

    # Academic-checklist aliases — persona := platform, event_type := example_type
    df["persona"] = df["platform"]
    df["event_type"] = df["example_type"]

    mlb = MultiLabelBinarizer()
    Y = mlb.fit_transform(df["labels"])
    classes: list[str] = list(mlb.classes_)

    return df, Y, classes


# ---------------------------------------------------------------------------
# Individual chart functions
# ---------------------------------------------------------------------------


def plot_label_marginals(Y: np.ndarray, classes: list[str]) -> None:
    """Bar chart of per-label marginal frequencies (plan §2.2)."""
    counts = Y.sum(axis=0)
    order = np.argsort(counts)[::-1]
    sorted_labels = [rtl(classes[i]) for i in order]
    sorted_counts = counts[order]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.barh(sorted_labels, sorted_counts, color=sns.color_palette("muted", len(classes)))
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title("Label Marginal Frequencies")
    ax.bar_label(bars, padding=3, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "01_label_marginals.png")

    # Sanity check: warn about sparse labels
    for i, c in enumerate(sorted_counts):
        if c <= 6:
            logger.warning("Sparse label '%s' — only %d examples.", classes[order[i]], c)


def plot_label_cooccurrence(Y: np.ndarray, classes: list[str]) -> None:
    """Co-occurrence heatmap of label pairs (plan §2.3)."""
    cooc = Y.T @ Y  # shape: (n_labels, n_labels)
    rtl_classes = [rtl(c) for c in classes]
    cooc_df = pd.DataFrame(cooc, index=rtl_classes, columns=rtl_classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        cooc_df,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Co-occurrence count"},
    )
    ax.set_title("Label Co-occurrence Matrix")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, "02_label_cooccurrence.png")


def plot_label_correlation(Y: np.ndarray, classes: list[str]) -> None:
    """Pearson correlation matrix between binary label vectors (phi coefficient for binary data)."""
    # np.corrcoef treats rows as variables; transpose so each label is a variable
    corr = np.corrcoef(Y.T)
    rtl_classes = [rtl(c) for c in classes]
    corr_df = pd.DataFrame(corr, index=rtl_classes, columns=rtl_classes)

    fig, ax = plt.subplots(figsize=(9, 7))
    sns.heatmap(
        corr_df,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        vmin=-1,
        vmax=1,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson correlation (phi)"},
    )
    ax.set_title("Label Pearson Correlation Matrix")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, "02b_label_correlation.png")


def plot_label_cardinality(df: pd.DataFrame) -> None:
    """Histogram of label-set size per record, including 0 for hard-negatives (plan §2.4)."""
    counts = Counter(df["label_count"])
    x = sorted(counts.keys())
    y = [counts[k] for k in x]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([str(v) for v in x], y, color=sns.color_palette("pastel")[0])
    ax.set_xlabel("Number of labels per record")
    ax.set_ylabel("Record count")
    ax.set_title("Label-Set Cardinality Distribution")
    ax.yaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    for xi, yi in zip(x, y):
        ax.text(x.index(xi), yi + 0.3, str(yi), ha="center", fontsize=9)
    fig.tight_layout()
    save_fig(fig, "03_cardinality.png")


def plot_length_by_platform(df: pd.DataFrame) -> None:
    """Violin plot of word count distribution per platform with sanity check (plan §2.5)."""
    platforms = sorted(df["platform"].unique())

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, col, label in zip(
        axes, ["word_count", "char_count"], ["Word count", "Character count"]
    ):
        sns.violinplot(
            data=df,
            x="platform",
            y=col,
            order=platforms,
            hue="platform",
            hue_order=platforms,
            palette="Set2",
            legend=False,
            ax=ax,
            inner="quartile",
        )
        ax.set_title(f"{label} by Platform")
        ax.set_xlabel("Platform")
        ax.set_ylabel(label)

    fig.tight_layout()
    save_fig(fig, "04_length_by_platform.png")

    # Sanity gate: flag uniform-5-word collapse
    logger.info("[TEXT LENGTH SANITY]")
    for platform in platforms:
        sub = df[df["platform"] == platform]["word_count"]
        median_w = sub.median()
        std_w = sub.std()
        flag = ""
        if median_w < _MIN_MEDIAN_WORDS:
            flag += f" LOW MEDIAN ({median_w:.1f} < {_MIN_MEDIAN_WORDS})"
        if std_w < _MIN_STD_WORDS:
            flag += f" LOW STD ({std_w:.1f} < {_MIN_STD_WORDS})"
        if flag:
            logger.warning("%s: median=%.1f words  std=%.1f%s", platform, median_w, std_w, flag)
        else:
            logger.info("%s: median=%.1f words  std=%.1f", platform, median_w, std_w)


def plot_platform_x_type(df: pd.DataFrame) -> None:
    """Stacked bar: platform × example_type (plan §2.6)."""
    ct = pd.crosstab(df["platform"], df["example_type"])
    fig, ax = plt.subplots(figsize=(8, 5))
    ct.plot(kind="bar", stacked=True, ax=ax, colormap="Set3", edgecolor="white")
    ax.set_title("Platform × Example Type")
    ax.set_xlabel("Platform")
    ax.set_ylabel("Record count")
    ax.legend(title="example_type", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    fig.tight_layout()
    save_fig(fig, "05_platform_x_type.png")


def plot_severity_x_label(Y: np.ndarray, df: pd.DataFrame, classes: list[str]) -> None:
    """Heatmap: severity × label frequency (plan §2.6)."""
    severity_order = ["mild", "medium", "strong"]
    matrix = []
    for sev in severity_order:
        mask = (df["severity"] == sev).values
        matrix.append(Y[mask].sum(axis=0))
    mat_df = pd.DataFrame(matrix, index=severity_order, columns=[rtl(c) for c in classes])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mat_df, annot=True, fmt="d", cmap="Blues", linewidths=0.5, ax=ax)
    ax.set_title("Severity × Label Frequency")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, "06_severity_x_label.png")


def plot_explicitness_x_label(Y: np.ndarray, df: pd.DataFrame, classes: list[str]) -> None:
    """Heatmap: explicitness × label frequency (plan §2.6)."""
    exp_values = sorted(df["explicitness"].dropna().unique())
    matrix = []
    for exp in exp_values:
        mask = (df["explicitness"] == exp).values
        matrix.append(Y[mask].sum(axis=0))
    mat_df = pd.DataFrame(matrix, index=exp_values, columns=[rtl(c) for c in classes])

    fig, ax = plt.subplots(figsize=(10, 4))
    sns.heatmap(mat_df, annot=True, fmt="d", cmap="Greens", linewidths=0.5, ax=ax)
    ax.set_title("Explicitness × Label Frequency")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    fig.tight_layout()
    save_fig(fig, "07_explicitness_x_label.png")


def plot_slang_top20(df: pd.DataFrame) -> None:
    """Top-20 slang token frequency bar chart (plan §2.7)."""
    all_slang: list[str] = [s for row in df["slang_used"] for s in row]
    coverage = df["slang_used"].apply(lambda x: len(x) > 0).mean()
    logger.info("Slang coverage: %.1f%% of records have non-empty slang_used.", coverage * 100)

    if not all_slang:
        logger.warning("No slang tokens found — skipping chart 08.")
        return

    top = Counter(all_slang).most_common(20)
    tokens, freqs = zip(*top)
    rtl_tokens = [rtl(t) for t in tokens]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(rtl_tokens, freqs, color=sns.color_palette("rocket_r", len(tokens)))
    ax.invert_yaxis()
    ax.set_xlabel("Frequency")
    ax.set_title(f"Top-20 Military Slang Tokens (coverage={coverage:.1%})")
    ax.bar_label(ax.containers[0], padding=3, fontsize=8)
    fig.tight_layout()
    save_fig(fig, "08_slang_top20.png")


# ---------------------------------------------------------------------------
# EDA tables
# ---------------------------------------------------------------------------


def build_eda_tables(df: pd.DataFrame, Y: np.ndarray, classes: list[str]) -> dict:
    """Compile numeric summary tables for report (plan §2.8)."""
    label_counts = {classes[i]: int(Y[:, i].sum()) for i in range(len(classes))}
    platform_counts = df["platform"].value_counts().to_dict()
    example_type_counts = df["example_type"].value_counts().to_dict()
    severity_counts = df["severity"].value_counts().to_dict()
    length_stats = (
        df[["word_count", "char_count"]]
        .describe()
        .round(2)
        .to_dict()
    )
    cardinality_dist = df["label_count"].value_counts().sort_index().to_dict()
    cardinality_dist = {int(k): int(v) for k, v in cardinality_dist.items()}

    return {
        "n_records": len(df),
        "label_marginals": {k: int(v) for k, v in sorted(label_counts.items(), key=lambda x: -x[1])},
        "platform_distribution": {str(k): int(v) for k, v in platform_counts.items()},
        "example_type_distribution": {str(k): int(v) for k, v in example_type_counts.items()},
        "severity_distribution": {str(k): int(v) for k, v in severity_counts.items()},
        "text_length_stats": {
            metric: {str(k): v for k, v in stat.items()}
            for metric, stat in length_stats.items()
        },
        "label_cardinality_distribution": cardinality_dist,
        "mean_label_cardinality": round(float(df["label_count"].mean()), 3),
        "slang_coverage_rate": round(
            float(df["slang_used"].apply(lambda x: len(x) > 0).mean()), 4
        ),
    }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_eda_pipeline(
    input_path: str | Path = CLEAN_DATASET_PATH,
    tables_path: str | Path = EDA_TABLES_PATH,
) -> dict:
    """
    Full EDA pass: load data, produce all charts, persist numeric tables.

    Falls back to raw dataset if clean file is absent.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        fallback = BASE_DIR / "data" / "dataset.json"
        logger.warning("%s not found, falling back to %s", input_path.name, fallback.name)
        input_path = fallback

    logger.info("Loading %s …", input_path.name)
    df, Y, classes = load_dataset(input_path)
    logger.info("Records: %d | Labels: %d | Columns: %s", len(df), len(classes), list(df.columns))

    logger.info("Generating charts …")
    plot_label_marginals(Y, classes)
    plot_label_cooccurrence(Y, classes)
    plot_label_correlation(Y, classes)
    plot_label_cardinality(df)
    plot_length_by_platform(df)
    plot_platform_x_type(df)
    plot_severity_x_label(Y, df, classes)
    plot_explicitness_x_label(Y, df, classes)
    plot_slang_top20(df)

    tables = build_eda_tables(df, Y, classes)
    tables_path = Path(tables_path)
    with open(tables_path, "w", encoding="utf-8-sig") as f:
        json.dump(tables, f, ensure_ascii=False, indent=2)
    logger.info("Tables -> %s", tables_path)
    logger.info("All charts saved to %s/", VISUALS_DIR)

    return tables


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Multi-label EDA for PTSD Hebrew dataset")
    parser.add_argument("--input", default=str(BASE_DIR / "dataset1240.clean.json"))
    parser.add_argument("--tables", default=str(BASE_DIR / "eda_tables.json"))
    args = parser.parse_args()

    run_eda_pipeline(input_path=args.input, tables_path=args.tables)
