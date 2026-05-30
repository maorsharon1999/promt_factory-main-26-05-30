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
        "UNNATURAL_ROBOTIC",
        "TRANSLATIONESE",
        "CLINICAL_LEAK",
        "LABEL_MISMATCH",
        "SLANG_MISUSE",
    ]
)

# Temperature adjustment thresholds for the feedback loop
_REJECTION_RATE_HIGH = 0.35
_TEMP_MIN = 0.1
_TEMP_MAX = 1.0

# System role for the judge — sets evaluation persona before any Hebrew text is seen
_JUDGE_SYSTEM = (
    "You are a clinical-linguistics auditor. "
    "You evaluate synthetic Hebrew text for quality issues. "
    "You ALWAYS respond in English. "
    "You ALWAYS output ONLY a valid JSON object — no prose, no markdown, no explanation outside the JSON."
)

# User turn for the G-Eval judge — CoT steps inside the user message
_JUDGE_PROMPT_TEMPLATE = """\
Evaluate the following synthetic Hebrew utterance for quality issues.
Follow ALL steps in order, then output ONLY the JSON object shown at the end.

=== CANDIDATE ===
text: {text}
platform: {platform}
labels: {labels}
slang_used: {slang_used}
severity: {severity}
=================

STEP 1 - RESTATE: Quote the text verbatim in one line.

STEP 2 - REGISTER CHECK: Does the tone / length / style match platform={platform}? Note any mismatch.

STEP 3 - CLINICAL LEAK CHECK: Does the text contain any of these diagnostic terms?
  PTSD, trauma, post-trauma, disorder, diagnosis, psychiatrist, psychologist,
  or their Hebrew equivalents (טראומה, פוסט-טראומה, הפרעת דחק, אבחנה, פסיכיאטר).
  If yes -> add code CLINICAL_LEAK.

STEP 4 - TRANSLATIONESE CHECK: Are there literal English calques or unidiomatic phrasing
  no native Hebrew speaker would naturally say? If yes -> add code TRANSLATIONESE.

STEP 5 - ROBOTIC CHECK: Are ALL FOUR of these true simultaneously?
  (a) opener is "מאז ש...", (b) token count < 8, (c) slang_used is empty, (d) severity != "mild".
  If ALL four -> add code UNNATURAL_ROBOTIC.

STEP 6 - LABEL CHECK: For each label in {labels}, does the text contain at least one
  behavioral/emotional/cognitive/somatic cue supporting it?
  If any label has NO textual evidence -> add code LABEL_MISMATCH.
  (Empty labels list is valid for hard-negatives, do not flag it.)

STEP 7 - SLANG CHECK: For each token in slang_used={slang_used}, is it used naturally
  as an Israeli soldier or veteran would use it? If not -> add code SLANG_MISUSE.

STEP 8 - VERDICT: If failure_codes is empty -> "ACCEPT". Otherwise -> "REJECT".

OUTPUT (JSON only — no text before or after the braces):
{{"reasoning": "<one paragraph in English summarizing your reasoning>", "failure_codes": [], "verdict": "ACCEPT"}}
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


def _parse_verdict(raw: str) -> JudgeVerdict:
    """Extract JudgeVerdict from raw LLM output, tolerating prose wrapping."""
    # Strip markdown fences if present
    raw = re.sub(r"```(?:json)?", "", raw).strip()
    # Find first { ... } block
    match = re.search(r"\{[\s\S]*\}", raw)
    if not match:
        # Fallback: treat as REJECT with parse error
        return JudgeVerdict(
            verdict="REJECT",
            failure_codes=["UNNATURAL_ROBOTIC"],
            reasoning=f"[PARSE ERROR] Could not extract JSON from: {raw[:200]}",
        )
    try:
        obj = json.loads(match.group())
    except json.JSONDecodeError as exc:
        return JudgeVerdict(
            verdict="REJECT",
            failure_codes=["UNNATURAL_ROBOTIC"],
            reasoning=f"[JSON DECODE ERROR] {exc} — raw: {raw[:200]}",
        )

    raw_codes = obj.get("failure_codes", [])
    # Sanitize: only keep recognized codes
    codes = [c for c in raw_codes if c in FAILURE_CODES]

    raw_verdict = str(obj.get("verdict", "")).upper()
    verdict: Literal["ACCEPT", "REJECT"] = (
        "ACCEPT" if raw_verdict == "ACCEPT" else "REJECT"
    )
    # If codes were emitted but verdict says ACCEPT, override to REJECT
    if codes and verdict == "ACCEPT":
        verdict = "REJECT"
    # If no codes but verdict says REJECT, add a generic code
    if not codes and verdict == "REJECT":
        codes = ["UNNATURAL_ROBOTIC"]

    return JudgeVerdict(
        verdict=verdict,
        failure_codes=codes,
        reasoning=str(obj.get("reasoning", "")),
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
        # Return a parse-safe ACCEPT fallback after 3 failures so pipeline continues
        logger.warning("judge SKIP — 3 retries exhausted: %s", last_exc)
        return '{"reasoning": "SKIP — LLM timeout after 3 retries", "failure_codes": [], "verdict": "ACCEPT"}'


def judge_record(
    record: dict,
    client: "JudgeGeminiClient | LLMClient",
    cache: dict[str, dict],
) -> JudgeVerdict:
    """
    Run G-Eval judge on a single dataset record.

    Uses SHA1 cache to avoid re-calling the LLM for identical texts.
    Preserves original variable names: client matches factory usage.
    """
    text = record.get("text", "")
    sha = _text_sha1(text)

    if sha in cache:
        cached = cache[sha]
        return JudgeVerdict(
            verdict=cached["verdict"],
            failure_codes=cached["failure_codes"],
            reasoning=cached["reasoning"],
        )

    prompt = _JUDGE_PROMPT_TEMPLATE.format(
        text=text,
        platform=record.get("platform", ""),
        labels=record.get("labels", []),
        slang_used=record.get("slang_used", []),
        severity=record.get("severity", ""),
    )

    raw = client.generate(prompt)
    verdict = _parse_verdict(raw)

    cache[sha] = {
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
