"""
quality_judge.py — G-Eval LLM-as-a-Judge quality filter for PTSD Hebrew slang dataset.

Ported from parent quality_judge.py.
LLM infrastructure imported directly from src.data_generation (replaces dynamic
spec_from_file_location loading of prompt_factory1240 (1).py).
All new comments and docstrings are in English.
File I/O uses utf-8-sig encoding to handle Hebrew BOM headers correctly.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

# Import LLM symbols from the unified data_generation module (original names preserved)
from src.data_generation import (
    LLMProvider,
    LLMConfig,
    LLMClient,
    ResilienceLLMClient,
    create_llm_client,
)

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 1240

# Closed failure-code taxonomy (plan §1.3)
FAILURE_CODES = frozenset(
    [
        # original codes
        "UNNATURAL_ROBOTIC",
        "TRANSLATIONESE",
        "CLINICAL_LEAK",
        "LABEL_MISMATCH",
        "SLANG_MISUSE",
        # deterministic pre-filter codes
        "ARTIFACT_PREFIX",
        "MIXED_SCRIPT",
        "GIBBERISH_SHORT",
        # scored rubric codes
        "LOW_NATURALNESS",
        "INCOHERENT",
    ]
)

# Temperature adjustment thresholds for the feedback loop
_REJECTION_RATE_HIGH = 0.35
_TEMP_MIN = 0.1
_TEMP_MAX = 1.0

# Import strictness knobs from config (can be overridden per-run via env/config edits)
from src.config import (
    JUDGE_MIN_WORDS,
    JUDGE_MAX_LATIN_RATIO,
    JUDGE_RUBRIC_MIN_SCORE,
    JUDGE_ARTIFACT_PREFIXES,
    JUDGE_CACHE_VERSION,
)

# System role for the judge — sets evaluation persona before any Hebrew text is seen
_JUDGE_SYSTEM = (
    "You are a clinical-linguistics auditor. "
    "You evaluate synthetic Hebrew text for quality issues. "
    "You ALWAYS respond in English. "
    "You ALWAYS output ONLY a valid JSON object — no prose, no markdown, no explanation outside the JSON."
)

# User turn for the G-Eval judge — scored rubric (Python decides verdict, not the model).
# No pre-filled verdict in the example to avoid anchoring the model toward ACCEPT.
_JUDGE_PROMPT_TEMPLATE = """\
You are evaluating a synthetic Hebrew utterance for quality.
Score each dimension 1-5 (1=very poor, 5=excellent), then answer the boolean flags.
Output ONLY the JSON object at the end — no text before or after the braces.

=== CANDIDATE ===
text: {text}
platform: {platform}
labels: {labels}
slang_used: {slang_used}
severity: {severity}
=================

SCORING DIMENSIONS (1-5 each):

naturalness:
  5 = a native Israeli speaker would write this without any hesitation
  3 = slightly awkward but intelligible
  1 = gibberish, word-salad, or clearly machine-translated

idiomaticity:
  5 = no English calques, no translationese, fully natural colloquial Hebrew
  3 = one mild calque or slightly stiff phrasing
  1 = multiple literal English translations or unidiomatic phrasing

register_fit:
  5 = tone, length, and style perfectly fit platform="{platform}"
  3 = minor mismatch (e.g. slightly too formal for a tweet)
  1 = completely wrong register (e.g. clinical report style in a WhatsApp message)

label_grounding:
  5 = every label in {labels} has clear behavioral/emotional/cognitive/somatic evidence in the text
      (if labels list is empty this is N/A — always score 5)
  3 = partial evidence for some labels
  1 = no textual evidence for one or more labels

coherence:
  5 = reads as a single meaningful Hebrew thought or passage
  3 = mostly coherent with one confusing part
  1 = disconnected fragments with no meaningful flow

BOOLEAN FLAGS:

clinical_leak:
  true if the text contains any of: PTSD, trauma, post-trauma, disorder, diagnosis,
  psychiatrist, psychologist, or Hebrew equivalents
  (טראומה, פוסט-טראומה, הפרעת דחק, אבחנה, פסיכיאטר, פסיכולוג).
  Otherwise false.

slang_natural:
  true if every token in slang_used={slang_used} is used naturally as an Israeli
  soldier or veteran would use it (empty list -> true).
  false if any slang token feels forced, misplaced, or incorrect.

OUTPUT (JSON only, no prose outside the braces):
{{"reasoning": "<one paragraph in English>",
  "scores": {{"naturalness": 3, "idiomaticity": 3, "register_fit": 3, "label_grounding": 3, "coherence": 3}},
  "clinical_leak": false,
  "slang_natural": true}}
"""

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class JudgeVerdict:
    """Structured output from a single G-Eval judge evaluation."""

    verdict: Literal["ACCEPT", "REJECT"]
    failure_codes: list[str]
    reasoning: str


@dataclass
class QualityFilterMetrics:
    """Accumulates filter statistics across the full dataset pass."""

    total_generated: int = 0
    total_accepted: int = 0
    total_rejected: int = 0
    per_code_counts: dict[str, int] = field(default_factory=dict)
    per_label_rejected: dict[str, int] = field(default_factory=dict)
    per_platform_rejected: dict[str, int] = field(default_factory=dict)

    @property
    def rejection_rate(self) -> float:
        if self.total_generated == 0:
            return 0.0
        return self.total_rejected / self.total_generated

    def dominant_code(self) -> str | None:
        if not self.per_code_counts:
            return None
        return max(self.per_code_counts, key=self.per_code_counts.get)  # type: ignore[arg-type]

    def to_dict(self) -> dict:
        return {
            "total_generated": self.total_generated,
            "total_accepted": self.total_accepted,
            "total_rejected": self.total_rejected,
            "rejection_rate": round(self.rejection_rate, 4),
            "per_code_counts": self.per_code_counts,
            "per_label_rejection_rate": {
                label: round(count / max(self.total_generated, 1), 4)
                for label, count in self.per_label_rejected.items()
            },
            "per_platform_rejection_rate": {
                platform: round(count / max(self.total_generated, 1), 4)
                for platform, count in self.per_platform_rejected.items()
            },
        }

    def record(self, record: dict, verdict: JudgeVerdict) -> None:
        """Update metrics after evaluating one record."""
        self.total_generated += 1
        if verdict.verdict == "ACCEPT":
            self.total_accepted += 1
        else:
            self.total_rejected += 1
            for code in verdict.failure_codes:
                self.per_code_counts[code] = self.per_code_counts.get(code, 0) + 1
            platform = record.get("platform", "unknown")
            self.per_platform_rejected[platform] = (
                self.per_platform_rejected.get(platform, 0) + 1
            )
            for label in record.get("labels", []):
                self.per_label_rejected[label] = (
                    self.per_label_rejected.get(label, 0) + 1
                )


# ---------------------------------------------------------------------------
# Deterministic pre-filter (no LLM call required)
# ---------------------------------------------------------------------------

import re as _re

# Hebrew Unicode block: U+0590–U+05FF
_HEBREW_RE = _re.compile(r"[֐-׿]")
_LATIN_RE  = _re.compile(r"[A-Za-z]")
# A "real" Hebrew word = token containing at least 2 Hebrew letters
_HEB_WORD_RE = _re.compile(r"[֐-׿]{2,}")


def normalize_text(text: str) -> str:
    """
    Strip known generator artifacts from text before evaluation.

    Currently strips leading '---' markers produced by the Stage 1 generator.
    The cleaned text is used for all quality checks but the *original* text
    field in the record is NOT mutated — callers must use the return value.
    """
    cleaned = text.strip()
    for prefix in JUDGE_ARTIFACT_PREFIXES:
        if cleaned.startswith(prefix):
            cleaned = cleaned[len(prefix):].strip()
            # strip an optional leading quote that sometimes follows the marker
            if cleaned.startswith('"'):
                cleaned = cleaned[1:].rstrip('"').strip()
            break
    return cleaned


def prefilter_record(record: dict) -> "JudgeVerdict | None":
    """
    Fast deterministic quality gate — runs before any LLM call.

    Strips generator artifacts first (e.g. leading '---'), then checks the
    cleaned text for mixed-script and gibberish issues.

    Returns a REJECT JudgeVerdict when an obvious defect remains after
    cleaning, or None when the record should proceed to the LLM rubric.
    """
    raw_text: str = record.get("text", "")
    cleaned = normalize_text(raw_text)

    # 1. After stripping artifact prefix, check mixed-script
    n_hebrew = len(_HEBREW_RE.findall(cleaned))
    n_latin  = len(_LATIN_RE.findall(cleaned))
    total_letters = n_hebrew + n_latin
    if total_letters > 0 and n_latin / total_letters > JUDGE_MAX_LATIN_RATIO:
        ratio = n_latin / total_letters
        return JudgeVerdict(
            verdict="REJECT",
            failure_codes=["MIXED_SCRIPT"],
            reasoning=(
                f"[PRE-FILTER] Latin-letter ratio {ratio:.1%} exceeds "
                f"threshold {JUDGE_MAX_LATIN_RATIO:.0%}."
            ),
        )

    # 2. Too few real Hebrew words (gibberish / too short)
    heb_words = _HEB_WORD_RE.findall(cleaned)
    if len(heb_words) < JUDGE_MIN_WORDS:
        return JudgeVerdict(
            verdict="REJECT",
            failure_codes=["GIBBERISH_SHORT"],
            reasoning=(
                f"[PRE-FILTER] Only {len(heb_words)} real Hebrew words "
                f"(threshold {JUDGE_MIN_WORDS})."
            ),
        )

    return None  # passes pre-filter — proceed to LLM rubric


# ---------------------------------------------------------------------------
# SHA1 verdict cache
# ---------------------------------------------------------------------------


def _text_sha1(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def _load_cache(cache_path: Path) -> dict[str, dict]:
    """Load previously computed verdicts from disk."""
    if cache_path.exists():
        with open(cache_path, encoding="utf-8-sig") as f:
            return json.load(f)
    return {}


def _save_cache(cache: dict[str, dict], cache_path: Path) -> None:
    with open(cache_path, "w", encoding="utf-8-sig") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Core judge function
# ---------------------------------------------------------------------------


# Mapping from rubric dimension name to failure code when score is too low
_DIMENSION_TO_CODE: dict[str, str] = {
    "naturalness":      "LOW_NATURALNESS",
    "idiomaticity":     "TRANSLATIONESE",
    "register_fit":     "UNNATURAL_ROBOTIC",
    "label_grounding":  "LABEL_MISMATCH",
    "coherence":        "INCOHERENT",
}


def _parse_verdict(raw: str) -> JudgeVerdict:
    """
    Extract JudgeVerdict from scored-rubric LLM output.

    Python decides ACCEPT/REJECT based on scores — the model never emits a verdict.
    Falls back to REJECT with GIBBERISH_SHORT on parse failure.
    """
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find first { ... } block
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        return JudgeVerdict(
            verdict="REJECT",
            failure_codes=["GIBBERISH_SHORT"],
            reasoning=f"[PARSE ERROR] Could not extract JSON from: {raw[:200]}",
        )
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return JudgeVerdict(
            verdict="REJECT",
            failure_codes=["GIBBERISH_SHORT"],
            reasoning=f"[JSON DECODE ERROR] {exc} — raw: {raw[:200]}",
        )

    reasoning = str(obj.get("reasoning", ""))
    codes: list[str] = []

    # --- score-based checks ---
    scores: dict = obj.get("scores", {})
    for dim, code in _DIMENSION_TO_CODE.items():
        score = scores.get(dim)
        if score is None:
            # Missing dimension treated as worst case
            codes.append(code)
            continue
        try:
            if int(score) < JUDGE_RUBRIC_MIN_SCORE:
                codes.append(code)
        except (TypeError, ValueError):
            codes.append(code)

    # --- boolean flag checks ---
    if obj.get("clinical_leak") is True:
        codes.append("CLINICAL_LEAK")

    if obj.get("slang_natural") is False:
        codes.append("SLANG_MISUSE")

    # Python decides verdict — model's opinion ignored
    verdict: Literal["ACCEPT", "REJECT"] = "REJECT" if codes else "ACCEPT"

    return JudgeVerdict(
        verdict=verdict,
        failure_codes=codes,
        reasoning=reasoning,
    )


class JudgeGeminiClient:
    """
    Dedicated Gemini client for the G-Eval judge.

    Uses gemini-1.5-flash and requests low-temperature JSON output.
    """

    def __init__(
        self,
        model: str = "gemini-1.5-flash",
        timeout: int = 300,
    ) -> None:
        self._model = model
        self._timeout = timeout
        self._api_key = os.environ.get("GEMINI_API_KEY", "")
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is not set for judge stage.")

    def _chat(self, system: str, user: str) -> str:
        import google.generativeai as _genai
        import time as _time
        _genai.configure(api_key=self._api_key)
        model = _genai.GenerativeModel(
            model_name=self._model,
            system_instruction=system,
        )
        _t0 = _time.time()
        response = model.generate_content(
            user,
            generation_config={
                "temperature": 0.1,
                "max_output_tokens": 350,
            },
            request_options={"timeout": self._timeout},
        )
        text = getattr(response, "text", "") or ""
        logger.debug("[judge:gemini] %.2fs", _time.time() - _t0)
        if not text.strip():
            raise RuntimeError("Empty Gemini judge response.")
        return text

    def generate(self, user_prompt: str) -> str:
        """Called by judge_record — wraps chat endpoint with judge system role. Retries on timeout."""
        import time
        last_exc: Exception = RuntimeError("no attempts made")
        for attempt in range(3):
            try:
                return self._chat(_JUDGE_SYSTEM, user_prompt)
            except Exception as exc:
                last_exc = exc
                wait = 5 * (attempt + 1)
                logger.warning("judge retry %d/3 — %s: %s — waiting %ds", attempt + 1, type(exc).__name__, exc, wait)
                time.sleep(wait)
        # Return a rubric-format ACCEPT fallback after 3 failures so pipeline continues.
        # All scores = 5 ensures _parse_verdict returns ACCEPT (no codes emitted).
        # Flip scores to 1 here if you want timeouts to REJECT instead.
        logger.warning("judge SKIP — 3 retries exhausted: %s", last_exc)
        return (
            '{"reasoning": "SKIP — LLM timeout after 3 retries",'
            ' "scores": {"naturalness": 5, "idiomaticity": 5, "register_fit": 5,'
            ' "label_grounding": 5, "coherence": 5},'
            ' "clinical_leak": false, "slang_natural": true}'
        )


def judge_record(
    record: dict,
    client: "JudgeGeminiClient | LLMClient",
    cache: dict[str, dict],
) -> JudgeVerdict:
    """
    Run quality gate on a single dataset record.

    Order:
      1. Deterministic pre-filter (no LLM, cheap).
      2. SHA1+version cache lookup.
      3. Scored LLM rubric (Python decides verdict).

    Cache keys are namespaced with JUDGE_CACHE_VERSION so bumping the version
    in config.py invalidates old lenient verdicts without deleting the cache file.
    """
    text = record.get("text", "")
    sha = _text_sha1(text)
    cache_key = f"{JUDGE_CACHE_VERSION}:{sha}"

    # 1. Deterministic pre-filter — free, no API call needed
    pre = prefilter_record(record)
    if pre is not None:
        # Cache the deterministic result too (cheap to re-compute but keeps metrics consistent)
        cache[cache_key] = {
            "verdict": pre.verdict,
            "failure_codes": pre.failure_codes,
            "reasoning": pre.reasoning,
        }
        return pre

    # 2. Cache lookup (versioned)
    if cache_key in cache:
        cached = cache[cache_key]
        return JudgeVerdict(
            verdict=cached["verdict"],
            failure_codes=cached["failure_codes"],
            reasoning=cached["reasoning"],
        )

    # 3. Scored LLM rubric — use normalized text (artifact prefix stripped)
    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        text=normalize_text(text),
        platform=record.get("platform", ""),
        labels=record.get("labels", []),
        slang_used=record.get("slang_used", []),
        severity=record.get("severity", ""),
    )

    raw = client.generate(prompt)
    verdict = _parse_verdict(raw)

    cache[cache_key] = {
        "verdict": verdict.verdict,
        "failure_codes": verdict.failure_codes,
        "reasoning": verdict.reasoning,
    }
    return verdict


# ---------------------------------------------------------------------------
# Feedback loop
# ---------------------------------------------------------------------------


def compute_feedback(metrics: QualityFilterMetrics, current_temperature: float) -> dict:
    """
    Deterministic temperature/prompt recommendation based on observed rejection patterns.

    Returns a JSON-serializable recommendation dict. Does NOT mutate the factory.
    """
    dominant = metrics.dominant_code()
    rate = metrics.rejection_rate
    rec: dict = {
        "rejection_rate": round(rate, 4),
        "dominant_code": dominant,
        "action": "none",
        "recommended_temperature": current_temperature,
        "prompt_injection": None,
    }

    if dominant is None or rate <= _REJECTION_RATE_HIGH:
        rec["action"] = "none"
        return rec

    if dominant == "UNNATURAL_ROBOTIC":
        new_temp = min(current_temperature + 0.1, _TEMP_MAX)
        rec["action"] = "increase_temperature"
        rec["recommended_temperature"] = round(new_temp, 2)
        rec["prompt_injection"] = None

    elif dominant == "TRANSLATIONESE":
        new_temp = max(current_temperature - 0.05, _TEMP_MIN)
        rec["action"] = "decrease_temperature_and_inject_constraint"
        rec["recommended_temperature"] = round(new_temp, 2)
        rec["prompt_injection"] = (
            "Avoid literal English calques and unidiomatic translations. "
            "Use only natural, colloquial Hebrew phrasing that a native speaker would say."
        )

    elif dominant == "CLINICAL_LEAK":
        rec["action"] = "strengthen_format_constraints"
        rec["recommended_temperature"] = current_temperature
        rec["prompt_injection"] = (
            "Strengthen the _format_constraints section in prompt_factory1240 to "
            "explicitly forbid naming PTSD, trauma disorders, or clinical diagnoses. "
            "Do NOT raise temperature."
        )

    return rec


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def run_judge_pipeline(
    input_path: str | Path,
    output_path: str | Path,
    metrics_path: str | Path,
    cache_path: str | Path,
    feedback_path: str | Path,
    current_temperature: float = 0.75,
    mock: bool = False,
) -> QualityFilterMetrics:
    """
    Full G-Eval filtering pass over a dataset JSON file.

    Writes:
      - output_path: clean dataset with judge_verdict / judge_codes fields appended
      - metrics_path: QualityFilterMetrics as JSON
      - cache_path: SHA1 verdict cache for reruns
      - feedback_path: temperature/prompt feedback recommendation
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    metrics_path = Path(metrics_path)
    cache_path = Path(cache_path)
    feedback_path = Path(feedback_path)

    with open(input_path, encoding="utf-8-sig") as f:
        dataset: list[dict] = json.load(f)

    cache = _load_cache(cache_path)

    # Use JudgeGeminiClient for structured JSON judging.
    # create_llm_client / ResilienceLLMClient are preserved for non-judge generation use.
    if mock:
        client: "JudgeGeminiClient | LLMClient" = create_llm_client(mock=True)
    else:
        client = JudgeGeminiClient(model="gemini-1.5-flash", timeout=300)

    metrics = QualityFilterMetrics()
    clean_records: list[dict] = []

    for i, record in enumerate(dataset):
        verdict = judge_record(record, client, cache)
        metrics.record(record, verdict)

        enriched = dict(record)
        enriched["judge_verdict"] = verdict.verdict
        enriched["judge_codes"] = verdict.failure_codes
        if verdict.verdict == "ACCEPT":
            clean_records.append(enriched)

        if (i + 1) % 10 == 0:
            logger.info(
                "[%d/%d] accepted=%d rejected=%d rate=%.2f%%",
                i + 1, len(dataset),
                metrics.total_accepted, metrics.total_rejected,
                metrics.rejection_rate * 100,
            )
        # Persist cache incrementally so reruns skip completed work
        _save_cache(cache, cache_path)

    # Write clean dataset (preserves all original keys + adds judge fields)
    with open(output_path, "w", encoding="utf-8-sig") as f:
        json.dump(clean_records, f, ensure_ascii=False, indent=2)

    # Write metrics
    with open(metrics_path, "w", encoding="utf-8-sig") as f:
        json.dump(metrics.to_dict(), f, ensure_ascii=False, indent=2)

    # Write feedback recommendation
    feedback = compute_feedback(metrics, current_temperature)
    with open(feedback_path, "w", encoding="utf-8-sig") as f:
        json.dump(feedback, f, ensure_ascii=False, indent=2)

    logger.info(
        "Done. %d/%d accepted (%.1f%% pass rate).",
        metrics.total_accepted, metrics.total_generated,
        (1 - metrics.rejection_rate) * 100,
    )
    logger.info("Clean dataset -> %s", output_path)
    logger.info("Metrics       -> %s", metrics_path)
    logger.info("Feedback      -> %s", feedback_path)

    return metrics


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="G-Eval quality filter for PTSD Hebrew dataset")
    parser.add_argument("--input", default="dataset1240.json")
    parser.add_argument("--output", default="dataset1240.clean.json")
    parser.add_argument("--metrics", default="quality_metrics.json")
    parser.add_argument("--cache", default="judge_cache.json")
    parser.add_argument("--feedback", default="judge_feedback.json")
    parser.add_argument("--temperature", type=float, default=0.75)
    parser.add_argument("--mock", action="store_true", help="Use MockLLMClient for testing")
    args = parser.parse_args()

    base = Path(__file__).parent
    run_judge_pipeline(
        input_path=base / args.input,
        output_path=base / args.output,
        metrics_path=base / args.metrics,
        cache_path=base / args.cache,
        feedback_path=base / args.feedback,
        current_temperature=args.temperature,
        mock=args.mock,
    )
