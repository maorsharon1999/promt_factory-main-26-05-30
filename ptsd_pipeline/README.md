# Project Sasha — PTSD Hebrew NLP Pipeline

End-to-end multi-label classification pipeline that detects clinical PTSD symptom indicators in
Hebrew military-slang text. Generates synthetic training data, filters it with an LLM judge,
runs EDA, splits into train/test, trains a TF-IDF + Logistic Regression baseline, and produces
a structured report.

---

## At a Glance

- **Input:** LLM-generated synthetic Hebrew utterances (WhatsApp, tweet, Reddit, diary)
- **Output:** Trained multi-label classifier + quality metrics + EDA visuals + slide summary
- **Labels:** 12 PTSD symptom categories (multi-label, includes hard-negative class)

---

## Pipeline Flow

```
run_pipeline.py
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 1 │ Data Generation    │ src/data_generation.py       │
│          │ Synthetic Hebrew   │ → data/dataset.json          │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 2 │ Quality Judge      │ src/quality_judge.py         │
│          │ G-Eval LLM filter  │ → data/dataset.clean.json    │
│          │                    │   artifacts/quality_metrics  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 3 │ EDA                │ src/eda.py                   │
│          │ Charts + tables    │ → visuals/*.png              │
│          │                    │   artifacts/eda_tables.json  │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 4 │ Stratified Split   │ src/splitting.py             │
│          │ Iterative strat.   │ → data/train_dataset.json    │
│          │ (Sechidis 2011)    │   data/test_dataset.json     │
│          │                    │   artifacts/split_manifest   │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 5 │ ML Baseline        │ src/modeling.py              │
│          │ TF-IDF + LR +      │ → console / logs/ metrics    │
│          │ Zero-shot mDeBERTa │                              │
└──────────────────────────────────────────────────────────────┘
       │
       ▼
┌──────────────────────────────────────────────────────────────┐
│  Stage 6 │ Report             │ src/report.py                │
│          │ Slide 3 summary    │ → reports/slide3_summary.md  │
│          │ + README           │   reports/README_eda.md      │
└──────────────────────────────────────────────────────────────┘
```

---

## The 6 Stages

| # | Stage | Module | Entry function | Reads | Writes |
|---|-------|--------|----------------|-------|--------|
| 1 | Data Generation | `src/data_generation.py` | `generate_dataset()` | — (LLM) | `data/dataset.json` |
| 2 | Quality Judge | `src/quality_judge.py` | `run_judge_pipeline()` | `data/dataset.json` | `data/dataset.clean.json`, `artifacts/quality_metrics.json`, `artifacts/judge_cache.json` |
| 3 | EDA | `src/eda.py` | `run_eda_pipeline()` | `data/dataset.clean.json` | `visuals/01–08_*.png`, `artifacts/eda_tables.json` |
| 4 | Split | `src/splitting.py` | `run_split_pipeline()` | `data/dataset.clean.json` | `data/train_dataset.json`, `data/test_dataset.json`, `artifacts/split_manifest.json` |
| 5 | Baseline | `src/modeling.py` | `run_baseline()` | `data/train_dataset.json`, `data/test_dataset.json` | logged macro-F1 (train/val/test) |
| 6 | Report | `src/report.py` | `run_report_pipeline()` | `artifacts/quality_metrics.json`, `artifacts/split_manifest.json`, `artifacts/eda_tables.json` | `reports/slide3_summary.md`, `reports/README_eda.md` |

Each stage reads its input from disk and writes its output to disk, so any single stage can be
re-run independently without repeating earlier stages.

---

## Folder Map

```
ptsd_pipeline/
├── run_pipeline.py        # Orchestrator — runs all 6 stages in sequence
├── gpu_check.py           # Detects CUDA / MPS / CPU at startup
├── src/                   # All pipeline logic lives here
│   ├── config.py          # Central path + hyperparameter constants
│   ├── logging_setup.py   # Console INFO + rotating file DEBUG handler
│   ├── data_generation.py # Stage 1 — LLM prompt factory + dataset builder
│   ├── quality_judge.py   # Stage 2 — G-Eval judge, SHA1 verdict cache
│   ├── eda.py             # Stage 3 — 8 matplotlib/seaborn charts
│   ├── splitting.py       # Stage 4 — iterative stratification split
│   ├── modeling.py        # Stage 5 — TF-IDF + LR + zero-shot baseline
│   └── report.py          # Stage 6 — Slide 3 + README_eda renderer
├── data/                  # Input and split datasets (JSON)
├── artifacts/             # Machine-readable outputs (JSON metrics/manifests)
├── reports/               # Human-readable outputs (Markdown)
├── visuals/               # EDA chart PNGs (300 DPI)
└── logs/                  # pipeline_run.log (DEBUG level)
```

---

## How to Run

```powershell
# Full pipeline end-to-end
python run_pipeline.py

# Run specific stages only (e.g. re-run EDA + split + baseline)
python run_pipeline.py --stages 3 4 5

# Skip data generation (dataset.json already exists)
python run_pipeline.py --skip-generation

# Skip LLM entirely — uses mock data (no Ollama / API key needed)
python run_pipeline.py --mock
```

---

## Key Configuration Knobs

All paths and hyperparameters live in **`src/config.py`**. Key values:

| Constant | Value | Used by |
|----------|-------|---------|
| `STRAT_SEED` | `1240` | Iterative split + EDA reproducibility |
| `RANDOM_SEED` | `42` | Modeling internal train/val split |
| `STRAT_TEST_SIZE` | `0.25` | 75/25 train/test split ratio |
| `VAL_SIZE` | `0.15` | Validation carve-out from training fold |
| `TFIDF_CONFIG` | unigrams+bigrams, `sublinear_tf=True` | Stage 5 feature extraction |
| `LR_CONFIG` | `C=1.0`, `class_weight="balanced"` | Stage 5 classifier |
| `ZS_MODEL` | `MoritzLaurer/mDeBERTa-v3-base-mnli-xnli` | Zero-shot baseline |

To change a path or parameter, edit `src/config.py` — no other file needs to change.

---

## Logging

Every stage writes to console (`INFO`) and to `logs/pipeline_run.log` (`DEBUG`).
Configured once in `run_pipeline.py` via `src/logging_setup.configure_logging()`.
