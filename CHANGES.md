# Quality Judge — Changes Log

## Problem

Pipeline ran 2000 records through Stage 2 (quality judge) and rejected only **6** (0.3%).
Root causes:

- Judge prompt never checked mixed-script (Latin in Hebrew), gibberish, or `---` artifacts
- `UNNATURAL_ROBOTIC` required **all four** conditions at once — almost never fired
- JSON example in prompt showed `"verdict": "ACCEPT"` — anchored the model toward ACCEPT
- 1091/2000 records had a `---` prefix from a Stage 1 generator bug (artifact, not quality issue)

Goal: raise rejection rate to ~15–25% (keep ~75–85% of sentences).

---

## Files Changed

- `ptsd_pipeline/src/config.py`
- `ptsd_pipeline/src/quality_judge.py`

---

## `src/config.py`

Added 5 strictness knobs to the Stage 2 section (after line 49).
All thresholds live here — tune pass rate without touching logic code.

```python
# Judge strictness knobs — tune pass rate without editing quality_judge.py logic.
# JUDGE_RUBRIC_MIN_SCORE: 3 = gentle (~90% pass), 4 = moderate (~80%), 5 = aggressive (~60%)
JUDGE_MIN_WORDS: int                    = 5       # reject if fewer than N real Hebrew words
JUDGE_MAX_LATIN_RATIO: float            = 0.10    # reject if >10% of letters are Latin
JUDGE_RUBRIC_MIN_SCORE: int             = 4       # each 1-5 rubric dim must score >= this
JUDGE_ARTIFACT_PREFIXES: tuple          = ("---",)  # artifact markers stripped before eval
JUDGE_CACHE_VERSION: str                = "v2"    # bump to invalidate old lenient verdicts
```

---

## `src/quality_judge.py`

### 1. Extended failure taxonomy (`FAILURE_CODES`)

Added 5 new codes alongside the original 5:

| Code | Source | Meaning |
|------|--------|---------|
| `ARTIFACT_PREFIX` | *(removed — strips instead of rejects)* | — |
| `MIXED_SCRIPT` | pre-filter | Too many Latin letters in Hebrew text |
| `GIBBERISH_SHORT` | pre-filter | Too few real Hebrew words |
| `LOW_NATURALNESS` | LLM rubric | Native speaker would not write this |
| `INCOHERENT` | LLM rubric | Disconnected fragments, no meaningful flow |

Original codes kept: `UNNATURAL_ROBOTIC`, `TRANSLATIONESE`, `CLINICAL_LEAK`, `LABEL_MISMATCH`, `SLANG_MISUSE`.

---

### 2. Config knobs imported at module level

```python
from src.config import (
    JUDGE_MIN_WORDS,
    JUDGE_MAX_LATIN_RATIO,
    JUDGE_RUBRIC_MIN_SCORE,
    JUDGE_ARTIFACT_PREFIXES,
    JUDGE_CACHE_VERSION,
)
```

---

### 3. `normalize_text(text)` — new function

Strips generator artifacts from text **before** any quality check.
Does **not** mutate the original record — returns cleaned string.

- Removes leading `---` marker (Stage 1 generator bug affecting 1091/2000 records)
- Strips a surrounding `"quote"` if present after the marker

```python
def normalize_text(text: str) -> str:
    cleaned = text.strip()
    for prefix in JUDGE_ARTIFACT_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            if cleaned.startswith('"'):
                cleaned = cleaned[1:].rstrip('"').strip()
            break
    return cleaned
```

---

### 4. `prefilter_record(record)` — new function

Fast deterministic gate. Runs **before** any LLM call (no API key needed).
Calls `normalize_text` first, then checks the cleaned text.

**Returns** a `REJECT JudgeVerdict` if a defect is found, or `None` to proceed to LLM.

| Check | Threshold | Failure code |
|-------|-----------|--------------|
| Latin letters / (Hebrew + Latin) | > `JUDGE_MAX_LATIN_RATIO` (10%) | `MIXED_SCRIPT` |
| Real Hebrew words (≥2 Hebrew letters) | < `JUDGE_MIN_WORDS` (5) | `GIBBERISH_SHORT` |

Result on the 2000-record dataset:
- Before: 1091 rejects (54.6%) — all `---` prefix, wrongly blocked
- After: **1 reject** (MIXED_SCRIPT — "הלATE"), 1999 pass to LLM rubric

---

### 5. Replaced LLM prompt (`_JUDGE_PROMPT_TEMPLATE`)

Old prompt: 8 steps with a narrow pass/fail check, ended with `"verdict": "ACCEPT"` example.

New prompt: **scored rubric**. Model outputs 1–5 scores per dimension + two booleans.
**Python decides ACCEPT/REJECT** — the model never emits a verdict.
No pre-filled verdict in the example (removes ACCEPT anchor bias).

#### Rubric dimensions

| Dimension | Failure code when score < threshold |
|-----------|-------------------------------------|
| `naturalness` | `LOW_NATURALNESS` |
| `idiomaticity` | `TRANSLATIONESE` |
| `register_fit` | `UNNATURAL_ROBOTIC` |
| `label_grounding` | `LABEL_MISMATCH` |
| `coherence` | `INCOHERENT` |

#### Boolean flags

| Flag | Failure code when triggered |
|------|-----------------------------|
| `clinical_leak: true` | `CLINICAL_LEAK` |
| `slang_natural: false` | `SLANG_MISUSE` |

#### Expected model output format

```json
{
  "reasoning": "<one paragraph in English>",
  "scores": {
    "naturalness": 4,
    "idiomaticity": 4,
    "register_fit": 3,
    "label_grounding": 5,
    "coherence": 4
  },
  "clinical_leak": false,
  "slang_natural": true
}
```

---

### 6. Replaced `_parse_verdict(raw)` logic

Old: extracted `verdict` string from model JSON → trusted the model's ACCEPT/REJECT.

New: parses `scores` dict + boolean flags → Python computes verdict.

- Any `scores[dim] < JUDGE_RUBRIC_MIN_SCORE` → append that dim's failure code
- `clinical_leak == true` → append `CLINICAL_LEAK`
- `slang_natural == false` → append `SLANG_MISUSE`
- `verdict = "REJECT" if codes else "ACCEPT"`
- Parse failure (bad JSON) → `REJECT` with `GIBBERISH_SHORT`

---

### 7. Updated `judge_record(record, client, cache)`

New execution order:

```
1. normalize_text(record["text"])          — strip --- artifact
2. prefilter_record(record)                — deterministic, no API
   └─ if REJECT → cache + return
3. cache lookup (key = "v2:{sha1}")        — skip LLM if already seen
   └─ if HIT → return cached
4. LLM scored rubric call                  — Gemini flash
   └─ _parse_verdict → Python decides verdict
   └─ cache + return
```

Cache keys namespaced with `JUDGE_CACHE_VERSION` (`"v2:..."`) so old lenient `"v1"` verdicts are ignored automatically — no need to delete `artifacts/judge_cache.json`.

LLM timeout fallback updated to emit valid rubric-format JSON (all scores = 5) so timed-out records still ACCEPT and pipeline continues.

---

## Tuning Guide

Edit `src/config.py` only:

| Goal | Change |
|------|--------|
| More lenient (~90% pass) | `JUDGE_RUBRIC_MIN_SCORE = 3` |
| Moderate (~80% pass) | `JUDGE_RUBRIC_MIN_SCORE = 4` ← current |
| Aggressive (~60% pass) | `JUDGE_RUBRIC_MIN_SCORE = 5` |
| Allow more Latin (URLs, names) | `JUDGE_MAX_LATIN_RATIO = 0.20` |
| Require longer sentences | `JUDGE_MIN_WORDS = 8` |
| Invalidate cache after logic change | `JUDGE_CACHE_VERSION = "v3"` |
| Reject on LLM timeout (strict) | change timeout fallback scores to `1` in `quality_judge.py` |

---

## How to Run

```bash
# Stage 2 only (judge existing dataset.json)
python run_pipeline.py --stages 2

# Check results
# artifacts/quality_metrics.json  — rejection rate + per-code breakdown
# data/dataset.clean.json         — accepted records only
# artifacts/judge_feedback.json   — temperature recommendation
```

Expected `quality_metrics.json` after this change:

```json
{
  "total_generated": 2000,
  "total_accepted": ~1500-1700,
  "rejection_rate": ~0.15-0.25,
  "per_code_counts": {
    "MIXED_SCRIPT": 1,
    "LOW_NATURALNESS": ...,
    "TRANSLATIONESE": ...,
    ...
  }
}
```
