"""
data_generation.py
==================
Ported from src/prompt_factory.py.
Generates synthetic Hebrew PTSD-indicator dataset via an LLM pipeline.

Public symbols re-exported for downstream consumers (quality_judge, run_pipeline):
    LLMProvider, LLMConfig, LLMClient, OllamaClient, OpenRouterClient,
    OpenAIClient, MockLLMClient, ResilienceLLMClient, create_llm_client,
    PromptFactory, Scenario, DatasetScenario, DatasetExample,
    DatasetPromptBuilder, DatasetGenerator, generate_dataset

All new comments are in English. Original variable names and docstrings preserved verbatim.
"""

from __future__ import annotations

import abc
import enum
import json
import logging
import os
import random
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

@dataclass
class Scenario:
    label: str
    language: str = "hebrew"
    platform: Literal["whatsapp", "reddit", "tweet", "diary"] = "whatsapp"
    speaker_role: str = "israeli reservist"
    age_range: str = "22-35"
    explicitness: Literal["explicit", "implicit", "behavioral"] = "implicit"
    severity: Literal["mild", "medium", "strong"] = "medium"
    slang_level: Literal["low", "medium", "high"] = "medium"
    include_military_context: bool = True
    forbidden_terms: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Label behavior map
# ---------------------------------------------------------------------------

LABEL_BEHAVIOR_MAP: dict[str, list[str]] = {
    "sleep_disturbance": [
        "trouble falling or staying asleep",
        "waking up suddenly in the middle of the night",
        "persistent insomnia or restless nights",
        "mind racing and unable to shut off thoughts at night",
    ],
    "hypervigilance": [
        "constantly scanning the surrounding environment",
        "heightened sensitivity to unexpected sounds or movements",
        "choosing seats that face exits or have a clear view of the room",
        "exaggerated startle response to sudden stimuli",
    ],
    "avoidance": [
        "deliberately avoiding specific locations or routes",
        "staying away from situations or objects that act as triggers",
        "refusing to watch news or consume media about certain events",
        "unwillingness to return to places connected to difficult experiences",
    ],
    "intrusive_memories": [
        "sudden uninvited flashbacks disrupting the present moment",
        "intrusive unwanted memories surfacing without warning",
        "sensory triggers (smells, sounds) bringing back vivid recollections",
        "random detailed recollections of past events breaking concentration",
    ],
    "anger_irritability": [
        "very low tolerance for minor frustrations",
        "sudden outbursts of anger disproportionate to the situation",
        "verbally aggressive reactions toward people nearby",
        "persistent underlying tension that builds and erupts unpredictably",
    ],
    "emotional_numbing": [
        "apparent absence of emotional reactions to normally moving events",
        "sense of emotional detachment from loved ones and surroundings",
        "feeling internally empty or hollow",
        "disconnection from one's own feelings and the people around them",
    ],
    "guilt_shame": [
        "intense self-blame over past decisions or actions",
        "survivor guilt and questioning why others suffered more",
        "moral injury from actions witnessed or participated in",
        "recurring expressions of regret about choices made under pressure",
    ],
    "functional_impairment": [
        "difficulty concentrating or performing normally at work",
        "withdrawing from social engagements and relationships",
        "reduced overall functioning compared to before",
        "disruption to everyday routines and basic daily tasks",
    ],
}

# ---------------------------------------------------------------------------
# Phrasing variation pools
# ---------------------------------------------------------------------------

_ROLE_INTROS: list[str] = [
    "You are generating synthetic text written by {speaker_role}, aged {age_range}.",
    "Simulate the voice of a {speaker_role} in the {age_range} age group.",
    "Write as if you are a {speaker_role} (age {age_range}) expressing themselves naturally.",
    "Produce text that authentically represents a {speaker_role}, {age_range} years old.",
]

_PLATFORM_FRAMES: dict[str, list[str]] = {
    "whatsapp": [
        "The text is a WhatsApp message sent to a close friend or family group chat.",
        "This is a casual WhatsApp message — informal, conversational, possibly using abbreviations.",
        "Format the output as a WhatsApp chat message: short, direct, informal.",
    ],
    "reddit": [
        "The text is a Reddit post or comment on a relevant Israeli forum.",
        "Format as a Reddit post: slightly longer, semi-anonymous, reflective tone.",
        "Write this as a Reddit comment — candid, somewhat introspective, internet-informal.",
    ],
    "tweet": [
        "The text is a tweet — concise, punchy, written under character pressure.",
        "Format as a tweet or X post: brief, emotionally raw, possibly using hashtags.",
        "Write a tweet-length post: terse, unfiltered, social-media casual.",
    ],
    "diary": [
        "The text is a personal diary entry — introspective, unguarded, private.",
        "Format as a diary entry: first-person, emotionally open, not meant for an audience.",
        "Write a private journal passage: honest, stream-of-consciousness, unedited feel.",
    ],
}

_LABEL_INTROS: dict[str, list[str]] = {
    "sleep_disturbance": [
        "The person is experiencing significant sleep difficulties.",
        "The writer is struggling with their sleep in a way that affects daily life.",
        "Sleep problems are at the center of what this person is describing.",
    ],
    "hypervigilance": [
        "The person is in a heightened state of alertness to their environment.",
        "An exaggerated awareness of surroundings and threats is the focus.",
        "The writer is showing signs of being constantly on guard.",
    ],
    "avoidance": [
        "The person is actively avoiding certain situations, places, or topics.",
        "Avoidance behavior is the dominant theme in this message.",
        "The writer refuses to engage with or return to certain triggering contexts.",
    ],
    "intrusive_memories": [
        "The person is experiencing unwanted memories breaking into the present.",
        "Intrusive recollections are disrupting the writer's current experience.",
        "Past events are surfacing unexpectedly and disturbing the person's focus.",
    ],
    "anger_irritability": [
        "The person is expressing or describing episodes of anger and irritability.",
        "Frustration and volatile reactions are the main theme.",
        "The writer is describing a short fuse and tension that is hard to contain.",
    ],
    "emotional_numbing": [
        "The person describes feeling emotionally flat or disconnected.",
        "An inability to feel or connect emotionally is the subject.",
        "The writer conveys a sense of internal emptiness and detachment.",
    ],
    "guilt_shame": [
        "The person is processing deep feelings of guilt or shame.",
        "Self-blame and moral weight are at the center of this text.",
        "The writer is grappling with responsibility and regret.",
    ],
    "functional_impairment": [
        "The person is describing an inability to function normally.",
        "Day-to-day functioning has broken down for the writer.",
        "The text reflects difficulty managing basic responsibilities and social life.",
    ],
}

_MILITARY_CONTEXT_PHRASES: list[str] = [
    "Include authentic Israeli military vocabulary where natural — such as references to miluim (reserve duty), "
    "a base (basa), a checkpoint (mahsom), unit (plugah), commander (mefaked), operational period, or returning home from duty.",
    "Weave in organic Israeli military language: miluim, basa, mahsom, operational zones, reserve call-up, "
    "squad mates (anashim min hayechida), or post-duty reintegration.",
    "Use genuine IDF-related terms naturally within the text — reserve duty (miluim), checkpoints, base life, "
    "unit dynamics, or the experience of returning from operational service.",
]

_SLANG_INSTRUCTIONS: dict[str, list[str]] = {
    "low": [
        "Use standard modern Hebrew with minimal slang. The tone should be clear and accessible.",
        "Keep language relatively neutral — everyday Hebrew without heavy colloquialisms.",
    ],
    "medium": [
        "Use a moderate level of Israeli slang and informal expressions typical of everyday speech.",
        "Mix standard Hebrew with common Israeli colloquialisms and spoken-language patterns.",
        "Include typical informal Hebrew phrasing: contractions, filler words, colloquial vocabulary.",
    ],
    "high": [
        "Use heavy Israeli slang, internet speak, and highly informal spoken-language register.",
        "Write in the most natural, unfiltered Israeli informal Hebrew — slang-heavy, fast, raw.",
        "Max out on colloquial Hebrew: heavy slang, abbreviations, expressive particles, youth language.",
    ],
}

_SEVERITY_INSTRUCTIONS: dict[str, list[str]] = {
    "mild": [
        "The intensity should be mild — the person hints at difficulty without full disclosure.",
        "Keep the emotional weight light — something is off but not overwhelming.",
        "The issue is present but understated; the person minimizes or glosses over it.",
    ],
    "medium": [
        "The emotional weight is moderate — the person is clearly affected but still functional.",
        "Write at a medium intensity — real struggle, described with some openness.",
        "The difficulty is evident and acknowledged, without being catastrophic.",
    ],
    "strong": [
        "The emotional intensity is high — the person is significantly distressed.",
        "Write at strong emotional weight — the burden is heavy and clearly communicated.",
        "The person is deeply affected and the text should reflect that without exaggeration.",
    ],
}

_EXPLICITNESS_INSTRUCTIONS: dict[str, list[str]] = {
    "explicit": [
        "The person directly names or describes what they are going through.",
        "Write explicitly — the subject states the problem clearly.",
    ],
    "implicit": [
        "The struggle is communicated indirectly — the reader infers the difficulty from context.",
        "Write implicitly — hints, omissions, and subtext carry the emotional content.",
        "The difficulty is present but not named; the reader must read between the lines.",
    ],
    "behavioral": [
        "Express the issue purely through described behaviors or actions — no emotional language.",
        "The text is behavioral: what the person is doing, not what they are feeling.",
        "Focus on actions and observable behaviors as the sole vehicle of expression.",
    ],
}

_VARIABILITY_REMINDERS: list[str] = [
    "Ensure the output does not sound formulaic or template-like.",
    "Prioritize naturalness — the text should feel spontaneous, not constructed.",
    "Vary sentence length and rhythm. Avoid sounding like a textbook example.",
    "Make the text feel like a real person wrote it, not a language model.",
    "Inject realistic imperfections: incomplete thoughts, run-on sentences, abrupt endings.",
]

_OUTPUT_INSTRUCTIONS: list[str] = [
    "Output ONLY 2–4 sentences of natural Hebrew text. No translation. No explanation. No meta-commentary.",
    "Return 2–4 sentences in Hebrew only. Do not include English. Do not explain the output.",
    "Write between 2 and 4 sentences in authentic Hebrew. Output nothing else.",
]


# ---------------------------------------------------------------------------
# PromptFactory
# ---------------------------------------------------------------------------

class PromptFactory:
    """Builds research-grade LLM prompts for synthetic Hebrew text generation."""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_prompt(self, scenario: Scenario) -> str:
        """Construct a single prompt string from a Scenario."""
        parts: list[str] = []

        parts.append(self._build_role_section(scenario))
        parts.append(self._build_platform_section(scenario))
        parts.append(self._build_label_section(scenario))
        parts.append(self._build_cues_section(scenario))
        parts.append(self._build_style_section(scenario))
        parts.append(self._format_constraints(scenario))
        parts.append(self._build_output_section())

        return "\n\n".join(p for p in parts if p.strip())

    def generate_batch_prompts(self, scenarios: list[Scenario]) -> list[str]:
        """Generate prompts for a list of Scenario objects (bulk pipeline use)."""
        return [self.build_prompt(s) for s in scenarios]

    # ------------------------------------------------------------------
    # Section builders
    # ------------------------------------------------------------------

    def _build_role_section(self, scenario: Scenario) -> str:
        template = random.choice(_ROLE_INTROS)
        role_line = template.format(
            speaker_role=scenario.speaker_role,
            age_range=scenario.age_range,
        )
        return f"[ROLE]\n{role_line}"

    def _build_platform_section(self, scenario: Scenario) -> str:
        options = _PLATFORM_FRAMES.get(scenario.platform, _PLATFORM_FRAMES["whatsapp"])
        line = random.choice(options)
        return f"[PLATFORM]\n{line}"

    def _build_label_section(self, scenario: Scenario) -> str:
        intro_options = _LABEL_INTROS.get(scenario.label)
        if intro_options:
            intro = random.choice(intro_options)
        else:
            intro = f"The person is experiencing something related to: {scenario.label.replace('_', ' ')}."

        explicitness_line = random.choice(
            _EXPLICITNESS_INSTRUCTIONS.get(scenario.explicitness, _EXPLICITNESS_INSTRUCTIONS["implicit"])
        )
        severity_line = random.choice(
            _SEVERITY_INSTRUCTIONS.get(scenario.severity, _SEVERITY_INSTRUCTIONS["medium"])
        )

        return (
            f"[LABEL INSTRUCTION]\n"
            f"{intro}\n"
            f"{explicitness_line}\n"
            f"{severity_line}"
        )

    def _build_cues_section(self, scenario: Scenario) -> str:
        cues = self._get_label_cues(scenario.label)
        random.shuffle(cues)
        formatted = "\n".join(f"  - {c}" for c in cues)
        header = random.choice([
            "Behavioral indicators to reflect (do NOT use clinical names):",
            "The text should organically reflect some of the following behaviors:",
            "Draw from these behavioral cues — weave them naturally into the text:",
        ])
        return f"[BEHAVIORAL CUES]\n{header}\n{formatted}"

    def _build_style_section(self, scenario: Scenario) -> str:
        slang_line = random.choice(
            _SLANG_INSTRUCTIONS.get(scenario.slang_level, _SLANG_INSTRUCTIONS["medium"])
        )
        variability_line = random.choice(_VARIABILITY_REMINDERS)

        military_line = ""
        if scenario.include_military_context:
            military_line = "\n" + random.choice(_MILITARY_CONTEXT_PHRASES)

        return (
            f"[STYLE CONSTRAINTS]\n"
            f"{slang_line}\n"
            f"{variability_line}"
            f"{military_line}"
        )

    def _format_constraints(self, scenario: Scenario) -> str:
        """Build the hard constraints block from forbidden terms and style rules."""
        base_constraints = [
            "Do NOT use clinical or diagnostic terminology (e.g., PTSD, trauma disorder, psychiatric).",
            "Do NOT frame the text as a diagnosis, self-diagnosis, or medical description.",
            "Do NOT include English in the generated Hebrew text.",
        ]

        if scenario.forbidden_terms:
            terms_str = ", ".join(f'"{t}"' for t in scenario.forbidden_terms)
            base_constraints.append(
                f"The following terms must NEVER appear in the output: {terms_str}."
            )

        formatted = "\n".join(f"  • {c}" for c in base_constraints)
        return f"[HARD CONSTRAINTS]\n{formatted}"

    def _build_output_section(self) -> str:
        instruction = random.choice(_OUTPUT_INSTRUCTIONS)
        return f"[OUTPUT]\n{instruction}"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_label_cues(self, label: str) -> list[str]:
        """Return behavioral cues for the given label. Returns empty list for unknown labels."""
        return list(LABEL_BEHAVIOR_MAP.get(label, []))


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def make_scenario(**kwargs) -> Scenario:
    """Construct a Scenario with keyword overrides for defaults."""
    return Scenario(**kwargs)


def quick_prompt(label: str, **kwargs) -> str:
    """One-liner helper: build a single prompt for a label with optional overrides."""
    scenario = Scenario(label=label, **kwargs)
    return PromptFactory().build_prompt(scenario)


# ---------------------------------------------------------------------------
# LLM Abstraction Layer
# ---------------------------------------------------------------------------

class LLMProvider(enum.Enum):
    GEMINI = "gemini"
    OLLAMA = "ollama"
    OPENROUTER = "openrouter"
    OPENAI = "openai"


@dataclass
class LLMConfig:
    provider: LLMProvider = LLMProvider.GEMINI
    model_name: str = "gemini-1.5-flash"
    api_key: str | None = None
    base_url: str | None = None
    fallback_enabled: bool = True
    allow_paid_apis: bool = False
    timeout: int = 60


class LLMClient(abc.ABC):
    @abc.abstractmethod
    def generate(self, prompt: str) -> str: ...

    @abc.abstractmethod
    def health_check(self) -> bool: ...

    def switch_provider(self, provider: LLMProvider) -> "LLMClient":
        return _build_client(LLMConfig(provider=provider))


class GeminiClient(LLMClient):
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._model_name = config.model_name or "gemini-1.5-flash"
        self._api_key = config.api_key or os.environ.get("GEMINI_API_KEY", "")

    def _extract_text(self, response: object) -> str:
        # Try the canonical response.text first, then fallback to candidate parts.
        text = getattr(response, "text", None)
        if isinstance(text, str) and text.strip():
            return text.strip()
        candidates = getattr(response, "candidates", None)
        if not candidates:
            return ""
        parts: list[str] = []
        for cand in candidates:
            content = getattr(cand, "content", None)
            if content is None:
                continue
            for part in getattr(content, "parts", []) or []:
                ptxt = getattr(part, "text", None)
                if isinstance(ptxt, str) and ptxt.strip():
                    parts.append(ptxt.strip())
        return "\n".join(parts).strip()

    def generate(self, prompt: str) -> str:
        import time as _time
        if not self._api_key:
            raise RuntimeError("GEMINI_API_KEY is not set.")
        try:
            import google.generativeai as genai
        except Exception as exc:
            raise RuntimeError(
                "google-generativeai is not installed. Install it with: pip install google-generativeai"
            ) from exc

        genai.configure(api_key=self._api_key)
        model = genai.GenerativeModel(self._model_name)
        _t0 = _time.time()
        try:
            response = model.generate_content(
                prompt,
                generation_config={
                    "temperature": 0.75,
                    "max_output_tokens": 120,
                },
                request_options={"timeout": self._config.timeout},
            )
            text = self._extract_text(response)
            if not text:
                raise RuntimeError("Gemini returned an empty response.")
            logger.debug("[gemini:generate] %.2fs", _time.time() - _t0)
            return text
        except Exception as exc:
            msg = str(exc)
            if "429" in msg or "quota" in msg.lower() or "rate" in msg.lower():
                raise RuntimeError(f"Gemini quota/rate error: {msg}") from exc
            if "timeout" in msg.lower() or "deadline" in msg.lower():
                raise RuntimeError(f"Gemini timeout error: {msg}") from exc
            raise RuntimeError(f"Gemini API error: {msg}") from exc

    def health_check(self) -> bool:
        return bool(self._api_key)


class OllamaClient(LLMClient):
    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._base_url = (config.base_url or "http://localhost:11434").rstrip("/")
        self._model = config.model_name or "llama3"

    def generate(self, prompt: str) -> str:
        import time as _time
        url = f"{self._base_url}/api/generate"
        payload = json.dumps(
            {"model": self._model, "prompt": prompt, "stream": False,
             "options": {
                 "temperature": 0.75,
                 # Safety cap — stop sequences below handle early exit in practice.
                 "num_predict": 120,
                 "num_ctx": 2048,
                 "num_gpu": 999,
             },
             # llama3 end-of-turn / end-of-text tokens.
             # Ollama stops as soon as any of these appear instead of grinding
             # to num_predict. Output is ~40-70 tokens; this saves 50-80 tokens
             # of wasted generation per call (~2-3 seconds each).
             "stop": ["<|eot_id|>", "<|end_of_text|>", "\n\n\n"],
             }
        ).encode()
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json"}
        )
        _t0 = _time.time()
        with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        _elapsed = _time.time() - _t0
        _eval_count = data.get("eval_count", 0)
        _tps = _eval_count / max(data.get("eval_duration", 1) / 1e9, 0.001)
        logger.debug("[ollama:generate] %.2fs | %d tokens | %.1f tok/s",
                     _elapsed, _eval_count, _tps)
        return data.get("response", "")

    def health_check(self) -> bool:
        try:
            urllib.request.urlopen(
                f"{self._base_url}/api/tags", timeout=5
            )
            return True
        except Exception:
            return False


class OpenRouterClient(LLMClient):
    _DEFAULT_BASE = "https://openrouter.ai/api/v1"
    _DEFAULT_MODEL = "mistralai/mistral-7b-instruct:free"

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._api_key = config.api_key or os.environ.get("OPENROUTER_API_KEY", "")
        self._base_url = (config.base_url or self._DEFAULT_BASE).rstrip("/")
        self._model = config.model_name or self._DEFAULT_MODEL

    def generate(self, prompt: str) -> str:
        if not self._api_key:
            raise RuntimeError("OPENROUTER_API_KEY is not set.")
        payload = json.dumps(
            {"model": self._model, "messages": [{"role": "user", "content": prompt}]}
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        return bool(self._api_key)


class OpenAIClient(LLMClient):
    _DEFAULT_BASE = "https://api.openai.com/v1"
    _DEFAULT_MODEL = "gpt-3.5-turbo"

    def __init__(self, config: LLMConfig) -> None:
        if not config.allow_paid_apis:
            raise RuntimeError(
                "OpenAI client is disabled by default. "
                "Set allow_paid_apis=True in LLMConfig to enable."
            )
        self._config = config
        self._api_key = config.api_key or os.environ.get("OPENAI_API_KEY", "")
        self._base_url = (config.base_url or self._DEFAULT_BASE).rstrip("/")
        self._model = config.model_name or self._DEFAULT_MODEL

    def generate(self, prompt: str) -> str:
        if not self._api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        payload = json.dumps(
            {"model": self._model, "messages": [{"role": "user", "content": prompt}]}
        ).encode()
        req = urllib.request.Request(
            f"{self._base_url}/chat/completions",
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=self._config.timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["choices"][0]["message"]["content"]

    def health_check(self) -> bool:
        return bool(self._api_key)


class MockLLMClient(LLMClient):
    _HEBREW_RESPONSES: list[str] = [
        "לא ישנתי כמעט בכלל הלילה. שוב. שוב קמתי באמצע הלילה ופשוט לא הצלחתי לחזור לישון.",
        "אני לא יודע איך להסביר את זה, פשוט לא מצליח להירגע. כל רעש קטן מזנק אותי.",
        "מאז שחזרתי מהמילואים אני מרגיש כאילו אני לא כאן. כאילו הכל קורה מסביבי ואני לא חלק מזה.",
        "פשוט לא יכול להגיע לשם. יודע שאני צריך אבל הגוף שלי מסרב. מעדיף לעשות עיקוף של עשרים דקות.",
        "הכל בסדר. עייף קצת. לא יודע, המון דברים בראש, אין כוח לדבר על זה עכשיו.",
        "פתאום באמצע הארוחה נזכרתי, ולא הצלחתי להמשיך. פשוט יצאתי החוצה לנשום.",
        "כבר לא זוכר מתי לא הייתי ערני ככה. כל כניסה לחדר, אני סורק הכל קודם.",
        "חזרתי מהבסיס לפני שבועיים. עוד לא ממש חזרתי.",
    ]

    def __init__(self, config: LLMConfig | None = None) -> None:
        self._config = config

    def generate(self, prompt: str) -> str:
        idx = abs(hash(prompt)) % len(self._HEBREW_RESPONSES)
        return self._HEBREW_RESPONSES[idx]

    def health_check(self) -> bool:
        return True


def _build_client(config: LLMConfig) -> LLMClient:
    if config.provider == LLMProvider.GEMINI:
        return GeminiClient(config)
    if config.provider == LLMProvider.OLLAMA:
        return OllamaClient(config)
    if config.provider == LLMProvider.OPENROUTER:
        return OpenRouterClient(config)
    if config.provider == LLMProvider.OPENAI:
        return OpenAIClient(config)
    raise ValueError(f"Unknown provider: {config.provider}")


class ResilienceLLMClient(LLMClient):
    """Wraps a primary LLMClient with free-first automatic fallback."""

    def __init__(self, config: LLMConfig) -> None:
        self._config = config
        self._primary = _build_client(config)
        self._fallback_chain: list[LLMClient] = self._build_fallback_chain(config)

    def _build_fallback_chain(self, config: LLMConfig) -> list[LLMClient]:
        chain: list[LLMClient] = []
        if os.environ.get("GEMINI_API_KEY") and config.provider != LLMProvider.GEMINI:
            chain.append(GeminiClient(LLMConfig(provider=LLMProvider.GEMINI)))
        if (
            os.environ.get("OPENROUTER_API_KEY")
            and config.provider != LLMProvider.OPENROUTER
        ):
            chain.append(
                OpenRouterClient(LLMConfig(provider=LLMProvider.OPENROUTER))
            )
        return chain

    def generate(self, prompt: str) -> str:
        if not self._config.fallback_enabled:
            return self._primary.generate(prompt)
        candidates = [self._primary] + self._fallback_chain
        last_error: Exception = RuntimeError("No LLM providers are available.")
        for client in candidates:
            try:
                return client.generate(prompt)
            except Exception as exc:
                last_error = exc
        return f"[LLM ERROR] All providers failed. Last error: {last_error}"

    def health_check(self) -> bool:
        return self._primary.health_check()

    def switch_provider(self, provider: LLMProvider) -> "ResilienceLLMClient":
        new_config = LLMConfig(
            provider=provider,
            model_name=self._config.model_name,
            api_key=self._config.api_key,
            base_url=self._config.base_url,
            fallback_enabled=self._config.fallback_enabled,
            allow_paid_apis=self._config.allow_paid_apis,
            timeout=self._config.timeout,
        )
        return ResilienceLLMClient(new_config)


def create_llm_client(
    provider: LLMProvider = LLMProvider.GEMINI,
    model_name: str = "gemini-1.5-flash",
    api_key: str | None = None,
    base_url: str | None = None,
    fallback_enabled: bool = True,
    allow_paid_apis: bool = False,
    mock: bool = False,
) -> LLMClient:
    if mock:
        return MockLLMClient()
    config = LLMConfig(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        base_url=base_url,
        fallback_enabled=fallback_enabled,
        allow_paid_apis=allow_paid_apis,
    )
    return ResilienceLLMClient(config)


# ---------------------------------------------------------------------------
# Dataset Generation Pipeline
# ---------------------------------------------------------------------------

CONTEXT_BANK: list[str] = [
    "מאז שחזרתי מהמילואים", "בחודשים האחרונים", "מאז כל הסיפור", "מאז התקופה בעזה", "לאחרונה",
    "אני שם לב ש", "מאז שחזרתי הביתה", "מאז שהשתחררתי מהצו", "בימים האחרונים", "מאז שחזרנו מהקו",
    "מהרגע שהורדתי את המדים", "בזמן האחרון", "מאז שכל הבלגן התחיל", "מאז ה-7 באוקטובר", "מאז התקופה בצפון",
    "כל פעם שאני נזכר במילואים", "מאז שחזרתי לשגרה", "מאז שיצאנו משם", "מאז שנגמר הסבב", "בתקופה האחרונה",
    "מאז שחזרתי לעבודה", "מאז הצו שמונה האחרון", "כל פעם שאני לבד", "מאז שהיינו שם בפנים",
    "מאז שחזרתי למציאות", "בשבועות האחרונים", "מאז שחזרתי ללימודים", "מאז שכל זה נגמר",
    "מאז שחזרתי למשפחה", "כל פעם שאני יוצא מהבית", "מאז שחזרתי לישון במיטה שלי", "מאז שחזרתי מהסבב האחרון",
]

BEHAVIOR_BANKS: dict[str, list[str]] = {
    "sleep_disturbance": [
        "לא מצליח לישון רצוף", "מתעורר מכל רעש קטן", "נרדם רק לפנות בוקר", "מתעורר עם דפיקות לב בלילה",
        "בוהה בתקרה שעות לפני שנרדם", "מתעורר באמצע הלילה ספוג בזיעה", "לא מצליח להירדם בלי טלוויזיה דולקת",
        "מפחד ללכת לישון לפעמים", "נרדם בקושי ומתעורר אחרי שעה", "מרגיש שגם כשאני ישן אני לא באמת נח",
        "קם כל כמה שעות לבדוק שהכל בסדר", "ישן ממש מעט שעות בלילה", "מתהפך במיטה עד הבוקר",
        "לא מצליח לעצום עיניים", "השינה שלי נהייתה ממש גרועה", "נרדם רק מול המחשב",
        "מתעורר בבהלה מכל שטות", "חלומות מוזרים מפריעים לי לישון", "הלילות הפכו להיות סיוט",
        "הגוף שלי עייף אבל המוח לא נרדם",
    ],
    "hypervigilance": [
        "קופץ מכל טריקת דלת", "תמיד מחפש איפה היציאה", "לא מצליח להירגע במקומות סגורים",
        "כל רעש חזק מקפיץ אותי", "סורק את כל האנשים מסביב ברחוב", "יושב תמיד עם הגב לקיר",
        "מרגיש דרוך כל הזמן בלי סיבה", "כל אופנוע שעובר נשמע לי כמו התרעה", "בודק את הדלת עשר פעמים לפני שאני נכנס",
        "כל רעש של זיקוקים מלחיץ אותי", "לא יכול לשבת במקומות הומים", "מרגיש שאני חייב להיות מוכן לכל תרחיש",
        "קופץ כשאנשים נוגעים בי מאחורה", "מסתכל לצדדים כל הזמן כשאני הולך ברחוב", "מרגיש שהגוף שלי במתח מתמיד",
        "כל טריקה של חלון נשמעת לי כמו בום", "לא מסוגל להוריד את המגננות", "דרוך כאילו אני עדיין בעמדה",
        "מחפש מחסה אוטומטית כשיש רעש פתאומי", "מרגיש שהאדרנלין לא יורד",
    ],
    "avoidance": [
        "לא פותח חדשות יותר", "נמנע מלנסוע לשם", "מחליף נושא ישר כשמדברים על זה", "הפסקתי ללכת למפגשי צוות",
        "לא יכול לראות סרטי מלחמה", "מתרחק מכל מה שמזכיר לי את התקופה ההיא", "פשוט לא רוצה לדבר על המילואים",
        "נמנע מלפגוש חברים מהצבא", "הפסקתי לעבור ברחוב שבו זה קרה", "מתעלם מהודעות בקבוצה של הפלוגה",
        "בורח מכל סיטואציה שיכולה להזכיר לי", "מעדיף לא לצאת מהבית כדי לא לשמוע סיפורים",
        "סוגר את הרדיו כשיש שירים של המלחמה", "לא מסוגל להסתכל על תמונות מהתקופה הזאת",
        "מתחמק משאלות של אנשים על מה שהיה", "שיניתי את הדרך הביתה כדי לא לעבור ליד הבסיס",
        "מנסה להדחיק הכל ולא לחשוב על זה", "נמנע מכל קשר עם הצבא עכשיו", "לא הולך לאזכרות או טקסים",
        "פשוט חוסם את כל מה שקשור לזה",
    ],
    "intrusive_memories": [
        "פתאום חוזרים לי קטעים לראש", "ריח מסוים ישר מחזיר אותי לשם", "רואה תמונות משם כשאני עוצם עיניים",
        "המראות משם לא עוזבים אותי", "פתאום אני מוצא את עצמי שוב בתוך האירוע", "הקולות משם מהדהדים לי בראש",
        "זיכרונות פשוט צפים בלי שליטה", "מרגיש כאילו אני שוב שם לרגע", "ריח של שריפה מקפיץ לי זיכרונות",
        "תמונות מהלחימה רצות לי בראש", "פתאום משום מקום הכל חוזר אליי", "המחשבות עוקבות אחריי לכל מקום",
        "רואה את הפנים שלהם מול העיניים שלי", "זה מרגיש חי מדי בראש שלי", "קשה לי להוציא את התמונות האלה מהראש",
        "הזיכרונות משם ממש מוחשיים לי", "כל דבר קטן מזכיר לי את מה שראיתי", "מרגיש שהמלחמה רצה לי בראש בלופ",
        "המחשבות על מה שהיה שם רודפות אותי", "פלאשבקים של דברים שקרו שם",
    ],
    "anger_irritability": [
        "מתעצבן מכל שטות", "אין לי סבלנות לאנשים", "מתפוצץ על הילדים בלי סיבה", "הכל מעלה לי את הסעיף",
        "מרגיש עצבני כל הזמן", "צועק על אנשים בכביש בלי הפסקה", "כל דבר קטן שמישהו אומר מרתיח אותי",
        "איבדתי את הסבלנות לכל העולם", "נהייתי תוקפני כלפי הסביבה שלי", "מתעצבן כששואלים אותי מה נשמע",
        "רב עם חברים על דברים מטופשים", "מרגיש שהדם שלי רותח בקלות", "כל ויכוח קטן הופך לפיצוץ",
        "פשוט אין לי כוח לשמוע אף אחד", "נהייתי ממש קצר עם בת הזוג שלי", "התגובות שלי נהיו ממש קיצוניות",
        "כל שטות בעבודה מוציאה אותי מדעתי", "מרגיש כעס פנימי שלא משתחרר", "מתעצבן על דברים שפעם לא הפריעו לי",
        "קשה לי לשלוט בעצבים שלי לאחרונה",
    ],
    "emotional_numbing": [
        "מרגיש מנותק מהכל", "לא באמת מתרגש מכלום", "מרגיש קצת ריק בפנים", "הכל נראה לי חסר משמעות עכשיו",
        "לא מצליח להרגיש שמחה", "מרגיש כמו רובוט שפשוט פועל", "התחושות שלי פשוט נעלמו",
        "לא מרגיש קרוב לאף אחד יותר", "קשה לי להרגיש אהבה או חום", "הכל נראה לי אפור ושטוח",
        "כבר לא נהנה מדברים שאהבתי", "מרגיש כאילו אני צופה בחיים מהצד", "הרגשות שלי פשוט קפאו",
        "לא מצליח להתחבר לאף אחד", "מרגיש אדישות להכל", "כאילו יש לי חומת אבן סביב הלב",
        "שום דבר לא באמת נוגע בי יותר", "מרגיש ריחוק מהמשפחה והחברים", "הכל מרגיש לי רחוק וזר",
        "איבדתי את היכולת להתרגש",
    ],
    "guilt_shame": [
        "לא מפסיק לחשוב מה הייתי יכול לעשות אחרת", "מרגיש לא בסדר עם עצמי", "אוכל את עצמי על דברים שקרו שם",
        "מרגיש רע שאני פה והם עדיין שם", "למה אני חזרתי והוא לא", "מרגיש אשם על זה שאני מנסה לחזור לשגרה",
        "משהו שם מרגיש לי לא פתור", "מרגיש שלא עשיתי מספיק בשביל הצוות", "מסתכל במראה ולא מזהה את עצמי",
        "מפחד שאנשים ידעו מה באמת עשינו שם", "מרגיש שהייתי צריך להיות יותר חזק", "אשם על זה שאני חי ונהנה",
        "מרגיש בושה על דברים שחשבתי עליהם שם", "אוכל את עצמי על החלטות שקיבלתי", "מרגיש רע עם זה שעזבתי אותם",
        "למה יצאתי לפני כולם", "מרגיש שמשהו בי נהרס שם", "אשם על זה שקשה לי עכשיו",
        "מרגיש שאני לא ראוי לכל הטוב הזה", "האשמה פשוט לא מרפה ממני",
    ],
    "functional_impairment": [
        "לא מצליח להתרכז בעבודה", "כבר אין לי כוח לאנשים", "שוכח דברים בסיסיים כל הזמן",
        "לא מצליח לעשות כלום בבית", "בוהה במסך שעות בלי לעשות כלום", "קשה לי לקבל החלטות פשוטות",
        "מרגיש שהראש שלי לא עובד כמו פעם", "לא מצליח לנהל שיחה רגילה", "כל משימה קטנה נראית לי הר",
        "הפסקתי לתפקד כמו שצריך", "מוצא את עצמי פשוט בוהה בקיר", "לא מצליח לחזור לקצב של העבודה",
        "שוכח פגישות ודברים חשובים", "אין לי מוטיבציה לכלום", "מרגיש שכל פעולה דורשת ממני המון כוח",
        "קשה לי לשבת על הלימודים", "הזנחתי את כל התחביבים שלי", "לא מצליח להחזיק סדר יום נורמלי",
        "כל הזמן מאחר לכל מקום", "מרגיש שהחיים שלי נעצרו",
    ],
}

NEGATIVE_BEHAVIOR_BANK: list[str] = [
    "המילואים האלה גמרו אותי מעייפות", "הרסר שוב דפק אותנו עם התורנויות", "אין לי כוח לעוד צו 8 בזמן הקרוב",
    "הציוד שקיבלנו היה פשוט פח", "נמאס לי לישון בשקי שינה", "האוכל בבסיס היה פשוט נוראי",
    "שוב אין תקציב לנסיעות של המילואים", "נשרף לי כל הקיץ על המילואים האלה",
    "המענק של המילואים עדיין לא נכנס לחשבון", "הבוס שלי עושה לי פרצופים על המילואים",
    "הפסדתי המון חומר בלימודים בגלל הצו", "התנאים בשטח היו פשוט זוועה", "נמאס לי מהבירוקרטיה של המערכת",
    "הקצינה שוב שכחה לחתום לי על הטפסים", "שוב פעם מקפיצים אותנו לאימון מיותר",
    "אין לי כוח לשמירות האלה יותר", "התבאסתי רצח שלא הייתי ביומולדת של הבת שלי",
    'הפסדתי טיסה לחו"ל בגלל הצו המטומטם הזה', "שוב פעם המדים האלה מלאים בחול",
    "היה חם רצח ואין מזגן באוהל", "נמאס לי מהפקקים בדרך לבסיס", 'החמ"ל הזה זה המקום הכי משעמם בעולם',
    "שוב פעם אכלנו לוף כל השבוע", "הנעליים הצבאיות גמרו לי את הרגליים", "היה קו קשה אבל לפחות נגמר",
    "התגעגעתי בטירוף לאוכל של הבית", "השירותים בבסיס זה הדבר הכי מגעיל שיש",
    "נמאס לי מהצעקות של המפקד", "כל היום רק חיכינו וחיכינו לשוברים",
    "למה תמיד המילואימניקים מקבלים את הציוד הישן", "הפסדתי מלא כסף מהעסק בגלל הסבב הזה",
]

TONE_BANK: dict[str, list[str]] = {
    "restrained": ["בשקט", "בלי להגזים", "לא יודע להסביר", "בקטע מוזר", "פשוט ככה"],
    "tired": ["אין לי כוח לזה", "אני גמור", "תשוש מהכל", "פשוט עייף", "בלי אנרגיות"],
    "cynical": ["כרגיל", "מה חדש", "אותו סיפור", "איזה יופי", "ממש חגיגה"],
    "frustrated": ["זה פשוט לא הגיוני", "כבר נמאס לי", "זה מעצבן ברמות", "חלאס", "אי אפשר ככה"],
    "detached": ["לא יודע", "וואלה", "משהו כזה", "בערך", "לא ממש משנה"],
}

SENTENCE_STARTERS: list[str] = [
    "אני שם לב ש", "לא יודע למה אבל", "בזמן האחרון", "מאז שחזרתי", "פתאום קלטתי ש",
    "וואלה", "תכלס", "בכנות", "שמעו", "משהו עובר עליי", "אני מרגיש ש", "יש לי קטע כזה ש",
    "בקיצור", "האמת ש", "קשה לי עם זה ש", "אני מוצא את עצמי", "לא מזמן קרה ש",
    "הקטע הוא ש", "בכלל לא חשבתי ש", "בלי קשר לכלום", "פשוט", "מסתבר ש",
    "איכשהו", "נראה לי ש", "זה התחיל כש",
]

SENTENCE_ENDINGS: list[str] = [
    "וזה מתחיל להפריע לי", "ולא ממש שמתי לב לזה עד עכשיו", "ואין לי מושג למה",
    "וזה פשוט ככה", "משהו הזוי", "וזהו בגדול", "וזה פשוט לא עובר",
    "אני כבר לא יודע מה לעשות", "וזה קורה כל הזמן", "כבר התרגלתי לזה לצערם של כולם",
    "וזה די מפחיד אותי", "משהו שאי אפשר להסביר", "וזה פשוט תקוע לי בראש",
    "וזה הכי מעצבן שיש", "ככה זה נראה כרגע", "וזה לא משתנה", "וזה ממש מוזר לי",
    "פשוט ככה", "וזהו", "משהו שם נשבר", "וזה פשוט מלווה אותי", "ואני לא רואה לזה סוף",
    "וזה קצת מלחיץ אותי", "וזה לא עוזב אותי", "כאילו זה חלק ממני עכשיו",
]

MILITARY_SLANG_BANK: list[str] = [
    'צו 8', 'חמ"ל', "שמירה", "קו", "עמדה", "צוות", "פלוגה", "מילואים", "סדיר", "ווסט",
    'נגמ"ש', "מדים", "בסיס", "מפקד", "קשר", "אימון", "שטח", "סבב", "כוננות", "משימה",
    "חוליה", "גדוד", "מחלקה", "נקודת מפגש",
]

AMBIGUOUS_BANK: list[str] = [
    "אני קצת אחר מאז", "קשה לי להסביר", "משהו בי השתנה", "הכל נראה אחרת פתאום",
    "אני לא אותו אדם", "דברים פשוט מרגישים אחרת", "קשה לי לחזור לעצמי",
    "אני מנסה להבין מה קורה לי", "משהו פשוט לא אותו דבר", "אני בתקופה קצת מוזרה",
    "קשה לשים על זה את האצבע", "התחושה הזאת לא עוזבת", "אני מרגיש שמשהו השתבש",
    "הכל נהיה קצת כבד", "אני לא מצליח להשתחרר מזה", "זה פשוט יושב עליי",
    "משהו שם בתקופה ההיא עשה לי משהו", "אני פשוט לא מוצא את המקום שלי",
    "דברים קטנים נהיו מסובכים", "משהו בתחושה הכללית לא בסדר",
]

_DATASET_FORBIDDEN_TERMS: list[str] = [
    "PTSD", "פוסט טראומה", "אבחון", "הפרעה", "diagnosis", "disorder",
    "post trauma", "psychiatric",
]

_HARD_NEGATIVE_THEMES: list[str] = [
    "exhaustion from reserve duty schedule, no PTSD indicators",
    "frustration with military bureaucracy and logistics only",
    "physical tiredness from long guard shifts, not psychological",
    "annoyance at poor planning by commanders",
    "boredom from base routine, missing family",
    "anger about poor equipment or living conditions at the base",
    "missing family while on miluim, otherwise fine",
    "complaints about food and organization at the base",
    "fatigue from lack of sleep due only to guard duty schedule",
]

_AMBIGUOUS_THEMES: list[str] = [
    "vague emotional discomfort, unclear if normal stress or more",
    "borderline restlessness after reserve duty, ambiguous interpretation",
    "difficulty concentrating — could be work pressure or something more",
    "vague unease about returning to certain places, reason unclear",
    "mild irritability that could be tiredness or something deeper",
]

_POSITIVE_COMBOS: list[list[str]] = [
    ["sleep_disturbance"],
    ["hypervigilance"],
    ["avoidance"],
    ["anger_irritability"],
    ["intrusive_memories"],
    ["emotional_numbing"],
    ["functional_impairment"],
    ["guilt_shame"],
    ["sleep_disturbance", "hypervigilance"],
    ["sleep_disturbance", "intrusive_memories"],
    ["anger_irritability", "guilt_shame"],
    ["avoidance", "emotional_numbing"],
    ["hypervigilance", "avoidance"],
    ["functional_impairment", "emotional_numbing"],
    ["sleep_disturbance", "anger_irritability"],
    ["guilt_shame", "intrusive_memories"],
    ["sleep_disturbance", "hypervigilance", "avoidance"],
    ["anger_irritability", "functional_impairment", "guilt_shame"],
]

_POSITIVE_WEIGHTS: list[int] = [
    5, 5, 4, 4, 2, 2, 2, 1,
    4, 2, 2, 2, 2, 1, 1, 1,
    1, 1,
]


@dataclass
class DatasetExample:
    id: str
    text: str
    labels: list[str]
    example_type: str
    platform: str
    explicitness: str
    severity: str
    slang_used: list[str]
    synthetic: bool = True


@dataclass
class DatasetScenario:
    labels: list[str]
    example_type: str
    platform: str
    explicitness: str
    severity: str
    slang_level: str
    include_military_context: bool
    theme_hint: str = ""


def _detect_slang(text: str) -> list[str]:
    return [term for term in MILITARY_SLANG_BANK if term in text]


def _plan_scenarios() -> list[DatasetScenario]:
    platforms = ["whatsapp", "reddit", "tweet", "diary"]

    def pick_combo() -> list[str]:
        return list(random.choices(_POSITIVE_COMBOS, weights=_POSITIVE_WEIGHTS, k=1)[0])

    scenarios: list[DatasetScenario] = []

    for _ in range(35):
        scenarios.append(DatasetScenario(
            labels=pick_combo(),
            example_type="positive_clear",
            platform=random.choice(platforms),
            explicitness=random.choice(["explicit", "behavioral", "behavioral"]),
            severity=random.choice(["medium", "strong", "strong"]),
            slang_level=random.choice(["low", "medium", "high"]),
            include_military_context=random.random() > 0.25,
        ))

    for _ in range(25):
        scenarios.append(DatasetScenario(
            labels=pick_combo(),
            example_type="implicit",
            platform=random.choice(platforms),
            explicitness="implicit",
            severity=random.choice(["mild", "medium"]),
            slang_level=random.choice(["medium", "high"]),
            include_military_context=random.random() > 0.3,
        ))

    for _ in range(25):
        scenarios.append(DatasetScenario(
            labels=[],
            example_type="hard_negative",
            platform=random.choice(platforms),
            explicitness=random.choice(["explicit", "behavioral"]),
            severity=random.choice(["mild", "medium"]),
            slang_level=random.choice(["low", "medium", "high"]),
            include_military_context=True,
            theme_hint=random.choice(_HARD_NEGATIVE_THEMES),
        ))

    for _ in range(15):
        labels = pick_combo() if random.random() > 0.4 else []
        scenarios.append(DatasetScenario(
            labels=labels,
            example_type="ambiguous",
            platform=random.choice(platforms),
            explicitness="implicit",
            severity="mild",
            slang_level=random.choice(["medium", "high"]),
            include_military_context=random.random() > 0.4,
            theme_hint=random.choice(_AMBIGUOUS_THEMES),
        ))

    random.shuffle(scenarios)
    return scenarios


_FEW_SHOT_EXEMPLARS: dict[str, list[str]] = {
    "sleep_disturbance": [
        "מאז שחזרתי אני לא באמת ישן רצוף. כל רעש קטן בבית מעיר אותי.",
        "שלוש בלילה, שוב. לא יודע למה אני מסוגל להירדם בכל מקום חוץ מהמיטה שלי.",
        "אמרתי לאישה שלי שאני בסדר אבל כבר שבוע שאני מתעורר בארבע ולא חוזר לישון.",
        "הראש לא נכבה בלילה. שוכב ומחכה לבוקר.",
    ],
    "hypervigilance": [
        "אני כבר לא שם לב שאני תמיד מחפש איפה היציאה כשאני נכנס למקום.",
        "ישבנו במסעדה ולא הצלחתי להפסיק לבדוק מי נכנס ויוצא. אישה שלי שמה לב לפני שאני.",
        "כל דלת שנטרקת מקפיצה אותי. אני שומע את זה ויודע שזה טיפשי אבל הגוף לא מקשיב.",
        "אני יושב תמיד עם הגב לקיר. לא מתכוון, פשוט ככה זה יוצא.",
    ],
    "avoidance": [
        "אין לי כוח לראות חדשות יותר. ישר נהיה לי כבד בראש.",
        "עברתי דרך ארוכה יותר היום כדי לא לעבור ליד הצומת הזה.",
        "הם הזמינו אותי למפגש של הפלוגה. אמרתי שיש לי משהו.",
        "יש מקומות שאני פשוט לא הולך אליהם יותר. לא מסביר למה.",
    ],
    "intrusive_memories": [
        "פתאום באמצע הארוחה נזכרתי, ולא הצלחתי להמשיך. פשוט יצאתי החוצה לנשום.",
        "הריח של הדלק הזכיר לי משהו ולא הצלחתי להמשיך לנהוג כמה דקות.",
        "לפעמים אני באמצע שיחה ואז הראש פשוט הולך לשם. אנשים לא מבינים.",
        "זיכרון ספציפי חוזר לי בלי שאני בוחר בו. ברגעים הכי לא מתאימים.",
    ],
    "anger_irritability": [
        "אני מתפוצץ על שטויות בזמן האחרון. הבן שלי שפך מיץ ונעשיתי משוגע. זה לא אני.",
        "הסבלנות שלי מאפס. לא יודע מה קרה לי.",
        "כולם מסביבי הולכים על קצות אצבעות. אני שומע את זה ומרגיש גרוע.",
        "פורץ על אנשים ואחר כך מצטער. לא מצליח לשלוט בזה.",
    ],
    "emotional_numbing": [
        "מאז שחזרתי מהמילואים אני מרגיש כאילו אני לא כאן. כאילו הכל קורה מסביבי ואני לא חלק מזה.",
        "הייתה חתונה של חבר. שמחתי בשבילו אבל לא הרגשתי כלום בפועל.",
        "הם מספרים בדיחות ואני צוחק כי ככה עושים. בפנים, אין כלום.",
        "אני נמצא אבל לא נוכח. לא יודע איך להסביר את זה אחרת.",
    ],
    "guilt_shame": [
        "יש דברים שלא מספרים. אתה מבין שצריך להמשיך הלאה אבל משהו נשאר.",
        "כולם אמרו שעשינו מה שהיינו צריכים. אבל אני עדיין חוזר לזה.",
        "חשבתי שזה יעבור עם הזמן. זה לא עובר.",
        "אני לא יכול להסביר לאנשים שלא היו שם. ולא בטוח שאני רוצה.",
    ],
    "functional_impairment": [
        "חזרתי לעבודה אבל אני לא ממש שם. יושב מול המסך ואין לי מושג מה עשיתי ביום.",
        "אמא שלי שאלה אם הכל בסדר. אמרתי כן. היא לא האמינה לי.",
        "אני לא מסוגל לסיים דברים פשוטים. מתחיל, מפסיק, שוכח.",
        "אפילו לסדר את הדירה נהיה מאמץ. פשוט לא מתחיל.",
    ],
    "hard_negative": [
        "עייף ממות. שמירה כל לילה זה פשוט גרוע.",
        "המפקד שלנו לא יודע לארגן כלום. שלוש שעות חיכינו לאוטובוס שלא הגיע.",
        "אין אינטרנט, האוכל גרוע, קר בלילות. מילואים קלאסי.",
        "גוועתי מקור כל הלילה על עמדה. מישהו יסביר לי למה זה קורה בחודש מאי?",
        "מתגעגע הביתה. שבוע ועוד ואני שם.",
    ],
    "ambiguous": [
        "הכל בסדר. עייף קצת. לא יודע, המון דברים בראש, אין כוח לדבר על זה עכשיו.",
        "חזרתי מהבסיס לפני שבועיים. עוד לא ממש חזרתי.",
        "קצת קשה, אבל עובר.",
        "לא יודע, בזמן האחרון אני פחות אני. אולי עייפות.",
    ],
}

_SITUATION_MAP: dict[str, list[str]] = {
    "sleep_disturbance": [
        "The person can't sleep well. They wake up at night and can't fall back asleep. Their mind won't shut off.",
        "Since returning from reserve duty, sleep is broken. Night after night, awake at odd hours for no clear reason.",
        "The person lies awake for hours. Tired all day but can't sleep when it matters.",
    ],
    "hypervigilance": [
        "The person finds themselves constantly scanning their surroundings without meaning to. Every sudden sound startles them.",
        "The person can't fully relax in public. They automatically position themselves to see the room and exits.",
        "Every small noise makes the person tense or jump. They're constantly alert in a way that's exhausting.",
    ],
    "avoidance": [
        "The person is avoiding certain places, routes, or topics without fully explaining why to anyone.",
        "The person can't bring themselves to watch the news or go back to certain locations.",
        "There are places the person just doesn't go anymore. They take longer routes. They change the subject.",
    ],
    "intrusive_memories": [
        "In the middle of ordinary moments, something triggers a memory and the person is suddenly somewhere else mentally.",
        "The person keeps getting pulled back to moments they didn't choose. Smells, sounds, random things trigger it.",
        "Specific memories come back without warning, breaking concentration at the worst times.",
    ],
    "anger_irritability": [
        "The person has been losing their temper over small things lately — things that never used to bother them.",
        "The person snaps at people around them and regrets it. They can't explain why everything irritates them.",
        "Patience is gone. The person reacts too strongly to minor frustrations and knows it but can't stop.",
    ],
    "emotional_numbing": [
        "The person feels emotionally flat. Present physically but disconnected from what's happening around them.",
        "The person notices they don't react the way they used to. Things that would have moved them don't anymore.",
        "There's a hollow quality to how the person experiences things. Like watching life from a distance.",
    ],
    "guilt_shame": [
        "The person carries something they can't easily explain to others. A weight that doesn't go away.",
        "The person keeps returning to decisions made under pressure. Wondering if they could have done differently.",
        "There's a sense of responsibility the person can't shake — being okay when others weren't.",
    ],
    "functional_impairment": [
        "The person is struggling to function at their usual level. Work, daily tasks, social life all feel harder.",
        "The person is withdrawing. Can't concentrate. Basic things are piling up and they can't get started.",
        "Functioning has dropped noticeably. Going through the motions but not really there.",
    ],
}

_NEGATIVE_SITUATION_POOL: list[str] = [
    "The person is physically exhausted from guard duty shifts and the reserve duty schedule — normal military fatigue, nothing more.",
    "The person is frustrated by poor logistics, bad planning, and disorganized commanders — annoyed but otherwise fine.",
    "The person is complaining about the food, equipment, boredom, and missing home — typical reservist venting.",
    "The person is tired, missing family, counting down the days. Completely normal reservist feelings, no deeper issues.",
    "The person is angry at bureaucracy and wasted time on base — a typical complaint about reserve duty inefficiency.",
]

_AMBIGUOUS_SITUATION_POOL: list[str] = [
    "The person mentions something vague — a feeling of being off, not quite themselves. Cause is unclear.",
    "The person is restless and a bit disconnected, but doesn't name why. Could be fatigue, could be something else.",
    "There's mild discomfort the person doesn't fully articulate. The reader can't be sure of the source.",
    "Something feels slightly wrong, but the person downplays it. The emotional signal is present but deniable.",
]


class DatasetPromptBuilder:
    _PLATFORM_BRIEF: dict[str, str] = {
        "whatsapp": "WhatsApp message to a close friend or family. Short, casual, informal — like real texting.",
        "reddit": "Reddit post or comment on an Israeli forum. Semi-anonymous, slightly longer, candid.",
        "tweet": "Tweet or X post. Very short, punchy, unfiltered.",
        "diary": "Personal diary entry. First-person, private, unguarded thoughts.",
    }

    def build(self, scenario: DatasetScenario) -> str:
        situation = self._get_situation(scenario)
        examples = self._get_examples(scenario)
        platform = self._PLATFORM_BRIEF.get(scenario.platform, self._PLATFORM_BRIEF["whatsapp"])
        slang_note = (
            "You may use 1-2 natural Hebrew military terms if they fit organically (e.g. מילואים, בסיס, עמדה)."
            if scenario.include_military_context and random.random() > 0.5
            else "Do NOT force military vocabulary — only include it if it sounds completely natural."
        )
        return (
            "You are generating realistic synthetic Hebrew text for NLP research.\n\n"
            "TASK: Write 2-4 short sentences in natural Israeli Hebrew.\n\n"
            f"SITUATION: {situation}\n\n"
            f"FORMAT: {platform}\n\n"
            "STYLE RULES:\n"
            "  - Write like a real Israeli person — not a translator, not a writer\n"
            "  - Use simple, everyday spoken Hebrew (not formal, not literary)\n"
            "  - Keep sentences short and natural, like real speech or texting\n"
            "  - No dramatic language, no poetic phrasing, no exaggeration\n"
            "  - No clinical vocabulary whatsoever\n"
            f"  - {slang_note}\n"
            "  - Output Hebrew ONLY — no English, no translations, no explanations\n"
            "  - Never use: PTSD, פוסט טראומה, אבחון, הפרעה, disorder\n\n"
            "EXAMPLES OF THE CORRECT STYLE:\n"
            + "\n".join(f'"{ex}"' for ex in examples)
            + "\n\nNOW WRITE THE TEXT (Hebrew only, 2-4 sentences, nothing else):"
        )

    def _get_situation(self, scenario: DatasetScenario) -> str:
        if scenario.example_type == "hard_negative":
            return random.choice(_NEGATIVE_SITUATION_POOL)
        if scenario.example_type == "ambiguous":
            base = random.choice(_AMBIGUOUS_SITUATION_POOL)
            if scenario.labels:
                label = random.choice(scenario.labels)
                hints = _SITUATION_MAP.get(label, [])
                if hints:
                    hint = random.choice(hints)
                    base = f"{base} Very faint hint toward: {hint[:70]}. Keep it deniable and indirect."
            return base
        parts: list[str] = []
        for label in scenario.labels:
            opts = _SITUATION_MAP.get(label, [])
            if opts:
                parts.append(random.choice(opts))
        if not parts:
            return "The person is having a hard time lately without naming the reason clearly."
        combined = " ".join(parts[:2])
        if scenario.explicitness == "implicit":
            combined += " Express this indirectly — the reader should infer the difficulty, not be told it."
        return combined

    def _get_examples(self, scenario: DatasetScenario) -> list[str]:
        pool: list[str] = []
        if scenario.example_type == "hard_negative":
            pool = list(_FEW_SHOT_EXEMPLARS.get("hard_negative", []))
        elif scenario.example_type == "ambiguous":
            pool = list(_FEW_SHOT_EXEMPLARS.get("ambiguous", []))
        else:
            for label in scenario.labels:
                pool.extend(_FEW_SHOT_EXEMPLARS.get(label, []))
        if not pool:
            pool = [
                "מאז שחזרתי אני לא באמת ישן רצוף. כל רעש קטן בבית מעיר אותי.",
                "הכל בסדר. עייף קצת. לא יודע, המון דברים בראש.",
                "אני מתפוצץ על שטויות בזמן האחרון. זה לא אני.",
            ]
        random.shuffle(pool)
        return pool[:3]
    def build_sectioned(self, scenario: DatasetScenario) -> str:
        # Section-based builder kept for reference; few-shot `build` is the active path.
        parts: list[str] = []
        parts.append(self._role())
        parts.append(self._platform(scenario.platform))
        if scenario.example_type == "hard_negative":
            parts.append(self._hard_negative_section(scenario))
        elif scenario.example_type == "ambiguous":
            parts.append(self._ambiguous_section(scenario))
        else:
            parts.append(self._label_section(scenario))
            cues = self._cues_section(scenario)
            if cues:
                parts.append(cues)
        parts.append(self._style_section(scenario))
        parts.append(self._constraints_section())
        parts.append(f"[OUTPUT]\n{random.choice(_OUTPUT_INSTRUCTIONS)}")
        return "\n\n".join(p for p in parts if p.strip())

    def _role(self) -> str:
        age = random.choice(["22-28", "28-35", "35-45"])
        role = random.choice([
            "israeli reservist",
            "combat soldier recently returned from miluim",
            "IDF veteran",
            "young soldier on reserve duty",
        ])
        return f"[ROLE]\n{random.choice(_ROLE_INTROS).format(speaker_role=role, age_range=age)}"

    def _platform(self, platform: str) -> str:
        opts = _PLATFORM_FRAMES.get(platform, _PLATFORM_FRAMES["whatsapp"])
        return f"[PLATFORM]\n{random.choice(opts)}"

    def _label_section(self, scenario: DatasetScenario) -> str:
        intros: list[str] = []
        for label in scenario.labels:
            opts = _LABEL_INTROS.get(label)
            if opts:
                intros.append(random.choice(opts))
        sev = random.choice(_SEVERITY_INSTRUCTIONS.get(scenario.severity, _SEVERITY_INSTRUCTIONS["medium"]))
        exp = random.choice(_EXPLICITNESS_INSTRUCTIONS.get(scenario.explicitness, _EXPLICITNESS_INSTRUCTIONS["implicit"]))
        return "[LABEL INSTRUCTION]\n" + "\n".join(intros) + f"\n{exp}\n{sev}"

    def _cues_section(self, scenario: DatasetScenario) -> str:
        all_cues: list[str] = []
        for label in scenario.labels:
            all_cues.extend(LABEL_BEHAVIOR_MAP.get(label, []))
        if not all_cues:
            return ""
        random.shuffle(all_cues)
        chosen = all_cues[:random.randint(2, min(4, len(all_cues)))]
        formatted = "\n".join(f"  - {c}" for c in chosen)
        header = random.choice([
            "Behavioral indicators to reflect (do NOT use clinical names):",
            "The text should organically reflect some of the following behaviors:",
            "Draw from these behavioral cues — weave them naturally into the text:",
        ])
        return f"[BEHAVIORAL CUES]\n{header}\n{formatted}"

    def _hard_negative_section(self, scenario: DatasetScenario) -> str:
        theme = scenario.theme_hint or random.choice(_HARD_NEGATIVE_THEMES)
        return (
            f"[CONTEXT]\n"
            f"The person is a soldier or reservist experiencing: {theme}.\n"
            f"This must NOT include any PTSD-like indicators — only normal military stress, fatigue, or frustration.\n"
            f"The text should reflect everyday complaints, tiredness, or minor frustrations only."
        )

    def _ambiguous_section(self, scenario: DatasetScenario) -> str:
        theme = scenario.theme_hint or random.choice(_AMBIGUOUS_THEMES)
        hint = ""
        if scenario.labels:
            cues: list[str] = []
            for label in scenario.labels:
                cues.extend(LABEL_BEHAVIOR_MAP.get(label, []))
            if cues:
                hint = f"\nA vague hint toward: {random.choice(cues)} — but kept ambiguous and deniable."
        return (
            f"[CONTEXT]\n"
            f"Theme: {theme}.{hint}\n"
            f"The emotional content must be unclear — a reader cannot be certain of the interpretation.\n"
            f"Keep the emotional signal very mild and indirect."
        )

    def _style_section(self, scenario: DatasetScenario) -> str:
        slang = random.choice(_SLANG_INSTRUCTIONS.get(scenario.slang_level, _SLANG_INSTRUCTIONS["medium"]))
        variability = random.choice(_VARIABILITY_REMINDERS)
        military = ""
        if scenario.include_military_context:
            military = "\n" + random.choice(_MILITARY_CONTEXT_PHRASES)
        return f"[STYLE CONSTRAINTS]\n{slang}\n{variability}{military}"

    def _constraints_section(self) -> str:
        terms = ", ".join(f'"{t}"' for t in _DATASET_FORBIDDEN_TERMS)
        return (
            f"[HARD CONSTRAINTS]\n"
            f"  • Do NOT use clinical or diagnostic terminology.\n"
            f"  • Do NOT include English in the output.\n"
            f"  • These terms must NEVER appear: {terms}."
        )


def _pick_slang_injection() -> str | None:
    roll = random.random()
    if roll < 0.80:
        return None
    pool = MILITARY_SLANG_BANK[:]
    if roll < 0.95:
        return random.choice(pool)
    return f"{random.choice(pool)} ו{random.choice(pool)}"


_SLANG_PERIOD_TERMS: set[str] = {"מילואים", "סדיר", "סבב", "קו", "אימון", "המבצע", "התקופה"}

_SUBJECT_NOUNS: set[str] = {
    "השינה", "הלילות", "החלומות", "הזיכרונות", "המחשבות", "הקולות",
    "האשמה", "הגוף", "התחושות", "התגובות", "הרגשות", "הזיכרונות",
}

_INTERROGATIVES: set[str] = {"למה", "איך", "מה", "איפה", "מתי"}


def _behavior_shape(phrase: str) -> str:
    # Return "clause" if phrase has its own subject or is a question; else "verb".
    s = phrase.strip()
    first = s.split()[0] if s else ""
    if first in _SUBJECT_NOUNS:
        return "clause"
    # Definite-noun subjects: starts with ה + ends with common plural/fem suffix
    if (first.startswith("ה") and len(first) > 3
            and any(first.endswith(suf) for suf in ("ים", "ות", "ה", "ן"))):
        return "clause"
    if first in _INTERROGATIVES:
        return "clause"
    return "verb"


def _compose_positive_clause(context: str, behavior: str) -> str:
    # Choose the right glue between context opener and behavior phrase.
    if _behavior_shape(behavior) == "clause":
        return f"{context} {behavior}."
    return f"{context} אני {behavior}."


def _apply_slang_to_sentences(sentences: list[str], slang: str, context: str) -> None:
    # Avoid "מאז {slang} — מאז …" collision; avoid non-period slang in temporal prefix.
    if context.startswith("מאז") or slang not in _SLANG_PERIOD_TERMS:
        sentences.append(f"({slang} — בלי קשר.)")
    else:
        sentences.insert(0, f"מאז {slang} —")


def _safe_tone_tail(tone_word: str) -> str:
    # Short tone words need a framing wrapper to stand alone as a sentence.
    if len(tone_word) < 6:
        return f"זה {tone_word}."
    return f"{tone_word}."


def _assemble_positive(labels: list[str], explicitness: str) -> str:
    context = random.choice(CONTEXT_BANK)
    sentences: list[str] = []
    chosen_labels = labels[:2] if len(labels) > 1 else labels
    behaviors: list[str] = []
    for label in chosen_labels:
        bank = BEHAVIOR_BANKS.get(label, [])
        if bank:
            behaviors.append(random.choice(bank))
    if not behaviors:
        behaviors = ["מרגיש לא בסדר"]
    tone_key = random.choice(list(TONE_BANK.keys()))
    tone_word = random.choice(TONE_BANK[tone_key])
    if explicitness == "implicit":
        starter = random.choice(SENTENCE_STARTERS)
        ending = random.choice(SENTENCE_ENDINGS)
        first_clause = _compose_positive_clause(context, behaviors[0])
        sentences.append(f"{starter} {first_clause}")
        if len(behaviors) > 1:
            sentences.append(f"{behaviors[1]}. {ending}.")
        else:
            sentences.append(f"{ending}.")
    else:
        sentences.append(_compose_positive_clause(context, behaviors[0]))
        if len(behaviors) > 1:
            b1 = behaviors[1]
            if _behavior_shape(b1) == "clause":
                sentences.append(f"{b1}.")
            else:
                sentences.append(f"בנוסף, {b1}.")
        sentences.append(_safe_tone_tail(tone_word))
    slang = _pick_slang_injection()
    if slang:
        _apply_slang_to_sentences(sentences, slang, context)
    return " ".join(sentences)


def _assemble_hard_negative() -> str:
    behaviors = random.sample(NEGATIVE_BEHAVIOR_BANK, k=random.randint(1, 3))
    tone_key = random.choice(["tired", "cynical", "frustrated"])
    tone_word = random.choice(TONE_BANK[tone_key])
    sentences = [f"{b}." for b in behaviors]
    sentences.append(_safe_tone_tail(tone_word))
    slang = _pick_slang_injection()
    if slang:
        sentences.append(f"({slang} — זה החיים.)")
    return " ".join(sentences)


def _assemble_ambiguous(labels: list[str]) -> str:
    context = random.choice(CONTEXT_BANK)
    ambiguous_phrase = random.choice(AMBIGUOUS_BANK)
    tone_word = random.choice(TONE_BANK["detached"] + TONE_BANK["restrained"])
    sentences = [f"{context}, {ambiguous_phrase}."]
    if labels:
        bank = BEHAVIOR_BANKS.get(random.choice(labels), [])
        if bank:
            subtle = random.choice(bank)
            sentences.append(f"אולי זה בגלל ש{subtle}, לא בטוח.")
    sentences.append(_safe_tone_tail(tone_word))
    return " ".join(sentences)


_POLISH_REJECT_PATTERNS: list[str] = [
    "(תרגום", "(translation", "(כתובה", "ניסיוני לשמור", "שכתוב:", "פלט עברית",
    "\u05be",  # maqaf used heavily in niqqud contexts
]

_NIQQUD_RANGE = (0x05B0, 0x05C7)  # Hebrew vowel points


def _has_niqqud(text: str) -> bool:
    return any(_NIQQUD_RANGE[0] <= ord(c) <= _NIQQUD_RANGE[1] for c in text)


import re as _re

# Structural defects the LLM should fix; reject polished output if still present.
_BROKEN_SYNTAX_PATTERNS: list[_re.Pattern[str]] = [
    _re.compile("אני ה[א-ת]{2,}"),            # "אני" + definite-noun subject
    _re.compile("אני (למה|איך|מה|איפה|מתי)"),  # "אני" + interrogative
    _re.compile("מאז.{0,40}—\\s*מאז"),         # duplicate "מאז" within ~40 chars
    _re.compile("(בערך|וואלה|כרגיל)[.\\s]*$"), # dangling short-tone tail
]


def _has_broken_syntax(text: str) -> bool:
    # Return True if any known structural defect is present in text.
    return any(p.search(text) is not None for p in _BROKEN_SYNTAX_PATTERNS)


def _polish_quality_ok(raw: str, polished: str) -> bool:
    """Return True only if polished is a safe improvement over raw."""
    if len(polished) < 15:
        return False
    # Must contain Hebrew letters
    he_count = sum(1 for c in polished if "\u05D0" <= c <= "\u05EA")
    if he_count < 10:
        return False
    # Reject niqqud (biblical/formal register)
    if _has_niqqud(polished):
        return False
    # Reject if LLM meta-text leaked in
    lower = polished.lower()
    for pat in _POLISH_REJECT_PATTERNS:
        if pat.lower() in lower:
            return False
    # Reject if polished lost too much content (< 30% of raw word count, loosened to allow restructuring)
    raw_words = len(raw.split())
    pol_words = len(polished.split())
    if raw_words > 3 and pol_words < raw_words * 0.3:
        return False
    # Reject if it got dramatically longer (LLM added content)
    if pol_words > raw_words * 2.5:
        return False
    # Reject archaic forms
    archaic = ["אינני", "אינו", "הנני", "הנה כי כן", "אמנם כי"]
    if any(a in polished for a in archaic):
        return False
    # Reject if structural defects survived the polish pass
    if _has_broken_syntax(polished):
        return False
    return True


def _polish_with_few_shot(
    llm: LLMClient,
    raw_text: str,
    scenario: "DatasetScenario",
    builder: "DatasetPromptBuilder",
) -> str:
    # Build a rich few-shot prompt so the LLM rewrites raw_text in authentic Hebrew style.
    base_prompt = builder.build(scenario)
    prompt = (
        base_prompt
        + "\n\n---\nNOW: Rewrite the DRAFT below as ONE coherent Israeli Hebrew message in the style shown above.\n"
        "- Keep the overall meaning and emotional content.\n"
        "- You ARE allowed to restructure sentences to fix broken grammar.\n"
        "- Fix these specific defects if present: 'אני ה...' before a noun, 'אני למה/איך/מה', duplicate 'מאז … — מאז', "
        "or a two-word standalone tail like 'בערך.' or 'וואלה.'.\n"
        "- Output ONLY the rewritten Hebrew text — no explanations, no English, no quotes.\n\n"
        f"DRAFT:\n{raw_text}"
    )
    try:
        result = llm.generate(prompt).strip()
        if len(result) >= 2 and result[0] in ('"', "'", "“") and result[-1] in ('"', "'", "”"):
            result = result[1:-1].strip()
        lines = [ln for ln in result.splitlines() if not ln.strip().startswith("(")]
        result = " ".join(" ".join(ln.split()) for ln in lines if ln.strip()).strip()
        if _polish_quality_ok(raw_text, result):
            return result
    except Exception:
        pass
    return raw_text


def _polish_with_llm(llm: LLMClient, raw_text: str) -> str:
    prompt = (
        "תקן את הטקסט הבא לעברית ישראלית מדוברת ויומיומית. "
        "אל תשנה את המשמעות. אל תוסיף מידע חדש. "
        "השב בטקסט בלבד, ללא הסברים, ללא סוגריים, ללא תרגום.\n\n"
        f"{raw_text}"
    )
    try:
        result = llm.generate(prompt).strip()
        # Strip surrounding quotes if present
        if len(result) >= 2 and result[0] in ('"', "'", "\u201c") and result[-1] in ('"', "'", "\u201d"):
            result = result[1:-1].strip()
        # Strip any line that looks like meta-commentary (starts with "(")
        lines = [ln for ln in result.splitlines() if not ln.strip().startswith("(")]
        result = " ".join(" ".join(ln.split()) for ln in lines if ln.strip()).strip()
        if _polish_quality_ok(raw_text, result):
            return result
    except Exception:
        pass
    return raw_text


def generate_dataset(llm: LLMClient, output_path: str = "dataset.json") -> list[DatasetExample]:
    TARGET = 3000
    PLAN = [
        ("positive_clear", 1050),
        ("implicit", 750),
        ("hard_negative", 750),
        ("ambiguous", 450),
    ]

    def pick_combo() -> list[str]:
        return list(random.choices(_POSITIVE_COMBOS, weights=_POSITIVE_WEIGHTS, k=1)[0])

    platforms = ["whatsapp", "reddit", "tweet", "diary"]
    # Builder provides few-shot exemplar anchoring for the polish step.
    _builder = DatasetPromptBuilder()
    examples: list[DatasetExample] = []
    seen: set[str] = set()
    context_prefix_count: dict[str, int] = {}
    MAX_SAME_PREFIX = max(1, TARGET // 10)  # no single context opener > 10% of total

    def accept(text: str) -> bool:
        if len(text) < 25:
            return False
        hebrew = sum(1 for c in text if "\u05D0" <= c <= "\u05EA")
        if hebrew < 12:
            return False
        tl = text.lower()
        for term in _DATASET_FORBIDDEN_TERMS:
            if term.lower() in tl:
                return False
        # Reject leaked meta-text
        for pat in _POLISH_REJECT_PATTERNS:
            if pat.lower() in tl:
                return False
        # Reject niqqud (biblical register)
        if _has_niqqud(text):
            return False
        # Reject archaic forms
        archaic = ["אינני", "אינו", "הנני"]
        if any(a in text for a in archaic):
            return False
        # Reject sentences cut mid-thought (ends with "בתוך." "מתוך." etc.)
        stripped = text.rstrip(" .")
        dangling = ["בתוך", "מתוך", "בתוך-", "לתוך"]
        if any(stripped.endswith(d) for d in dangling):
            return False
        # Reject structural grammar defects
        if _has_broken_syntax(text):
            return False
        norm = " ".join(text.split()).lower()
        if norm in seen:
            return False
        return True

    for example_type, count in PLAN:
        generated = 0
        attempts = 0
        while generated < count and attempts < count * 8:
            attempts += 1
            labels: list[str] = []
            explicitness = "explicit"
            severity = "medium"

            slang_level = "medium"
            include_military_context = random.random() > 0.3

            if example_type == "positive_clear":
                labels = pick_combo()
                explicitness = random.choice(["explicit", "behavioral"])
                severity = random.choice(["medium", "strong"])
                slang_level = random.choice(["low", "medium", "high"])
                raw = _assemble_positive(labels, explicitness)
            elif example_type == "implicit":
                labels = pick_combo()
                explicitness = "implicit"
                severity = random.choice(["mild", "medium"])
                slang_level = random.choice(["medium", "high"])
                raw = _assemble_positive(labels, explicitness)
            elif example_type == "hard_negative":
                labels = []
                explicitness = "explicit"
                severity = "mild"
                slang_level = random.choice(["low", "medium"])
                include_military_context = True
                raw = _assemble_hard_negative()
            else:
                labels = pick_combo() if random.random() > 0.4 else []
                explicitness = "implicit"
                severity = "mild"
                slang_level = random.choice(["medium", "high"])
                raw = _assemble_ambiguous(labels)

            # Build a scenario so the polish step can inject few-shot exemplars.
            _scenario = DatasetScenario(
                labels=labels,
                example_type=example_type,
                platform=random.choice(platforms),
                explicitness=explicitness,
                severity=severity,
                slang_level=slang_level,
                include_military_context=include_military_context,
            )
            # Give the LLM up to 3 passes to produce an acceptable rewrite.
            text = raw
            for _ in range(3):
                candidate = _polish_with_few_shot(llm, raw, _scenario, _builder)
                if accept(candidate):
                    text = candidate
                    break
            if not accept(text):
                continue

            # Enforce context-opener diversity
            first_word = text.split()[0] if text.split() else ""
            prefix_key = " ".join(text.split()[:3])
            if context_prefix_count.get(prefix_key, 0) >= MAX_SAME_PREFIX:
                continue
            context_prefix_count[prefix_key] = context_prefix_count.get(prefix_key, 0) + 1

            norm = " ".join(text.split()).lower()
            seen.add(norm)
            examples.append(DatasetExample(
                id=f"syn_{len(examples):04d}",
                text=text,
                labels=labels,
                example_type=example_type,
                platform=random.choice(platforms),
                explicitness=explicitness,
                severity=severity,
                slang_used=_detect_slang(text),
                synthetic=True,
            ))
            generated += 1
            logger.info(
                "[%3d/%d] %-14s | labels=%s",
                len(examples), TARGET, example_type, labels,
            )

            # Fault-tolerance: save a checkpoint every 200 samples,
            # independently of the final output file.
            CHECKPOINT_INTERVAL = 200
            if len(examples) % CHECKPOINT_INTERVAL == 0:
                _checkpoint_path = Path(output_path).with_name("dataset.checkpoint.json")
                _checkpoint_records = [
                    {
                        "id": ex.id, "text": ex.text, "labels": ex.labels,
                        "example_type": ex.example_type, "platform": ex.platform,
                        "explicitness": ex.explicitness, "severity": ex.severity,
                        "slang_used": ex.slang_used, "synthetic": ex.synthetic,
                    }
                    for ex in examples
                ]
                with _checkpoint_path.open("w", encoding="utf-8") as _cp_fh:
                    json.dump(_checkpoint_records, _cp_fh, ensure_ascii=False, indent=2)
                logger.info(
                    "Checkpoint saved: %d samples → %s",
                    len(examples), _checkpoint_path,
                )

    records = [
        {
            "id": ex.id, "text": ex.text, "labels": ex.labels,
            "example_type": ex.example_type, "platform": ex.platform,
            "explicitness": ex.explicitness, "severity": ex.severity,
            "slang_used": ex.slang_used, "synthetic": ex.synthetic,
        }
        for ex in examples
    ]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d examples → %s", len(records), output_path)
    return examples


class DatasetGenerator:
    MIN_CHARS: int = 30
    QUALITY_THRESHOLD: float = 0.5
    TARGET: int = 2000
    MAX_RETRIES: int = 3

    def __init__(self, llm: LLMClient) -> None:
        self._llm = llm
        self._builder = DatasetPromptBuilder()
        self._seen: set[str] = set()
        self._examples: list[DatasetExample] = []

    def run(self) -> list[DatasetExample]:
        pool = _plan_scenarios()
        idx = 0
        while len(self._examples) < self.TARGET:
            if idx >= len(pool):
                pool.extend(_plan_scenarios())
            scenario = pool[idx]
            idx += 1
            example = self._attempt(scenario)
            if example:
                self._examples.append(example)
                logger.info(
                    "[%3d/%d] %-14s | %-8s | labels=%s",
                    len(self._examples), self.TARGET,
                    scenario.example_type, scenario.platform, scenario.labels,
                )
        return self._examples

    def _attempt(self, scenario: DatasetScenario) -> DatasetExample | None:
        for _ in range(self.MAX_RETRIES):
            try:
                prompt = self._builder.build(scenario)
                text = self._llm.generate(prompt).strip()
                # Strip surrounding quotes the model sometimes wraps output in
                if len(text) > 2 and text[0] in ('"', "'") and text[-1] in ('"', "'"):
                    text = text[1:-1].strip()
                if not self._is_valid(text):
                    continue
                if self._quality_score(text) < self.QUALITY_THRESHOLD:
                    continue
                norm = self._normalize(text)
                if norm in self._seen:
                    continue
                self._seen.add(norm)
                return DatasetExample(
                    id=f"syn_{len(self._examples):04d}",
                    text=text,
                    labels=list(scenario.labels),
                    example_type=scenario.example_type,
                    platform=scenario.platform,
                    explicitness=scenario.explicitness,
                    severity=scenario.severity,
                    slang_used=_detect_slang(text),
                    synthetic=True,
                )
            except Exception as exc:
                logger.warning("Generation failed: %s", exc)
        return None

    def _is_valid(self, text: str) -> bool:
        if len(text) < self.MIN_CHARS:
            return False
        # Must contain enough Hebrew characters
        hebrew_chars = sum(1 for c in text if "\u05D0" <= c <= "\u05EA")
        if hebrew_chars < 15:
            return False
        # Forbidden terms
        tl = text.lower()
        for term in _DATASET_FORBIDDEN_TERMS:
            if term.lower() in tl:
                return False
        # Translation artifacts and meta-commentary
        reject_phrases = [
            "translation:", "תרגום:", "here is", "here's the", "english:",
            "output:", "note:", "explanation:", "i have written", "```",
            "sure!", "certainly!", "of course!", "here you go",
        ]
        for phrase in reject_phrases:
            if phrase in tl:
                return False
        # Too many English words (more than 3 multi-char ASCII words)
        ascii_words = [
            w for w in text.split()
            if len(w) > 2 and all(ord(c) < 128 and c.isalpha() for c in w)
        ]
        if len(ascii_words) > 3:
            return False
        # Repeated consecutive words
        words = text.split()
        for i in range(len(words) - 1):
            if words[i] == words[i + 1] and len(words[i]) > 1:
                return False
        return True

    def _quality_score(self, text: str) -> float:
        score = 1.0
        # Penalize very short or very long texts
        if len(text) < 40:
            score -= 0.3
        if len(text) > 450:
            score -= 0.3
        # Penalize dense slang (unnatural)
        slang_count = len(_detect_slang(text))
        if slang_count > 3:
            score -= 0.3
        if slang_count > 5:
            score -= 0.3
        # Penalize multiple colons (sign of meta-commentary or lists)
        if text.count(":") > 1:
            score -= 0.2
        # Penalize texts starting with quote characters
        if text and text[0] in ('"', "'", "\u201c", "\u2018"):
            score -= 0.2
        # Penalize texts with very few words
        if len(text.split()) < 5:
            score -= 0.3
        return max(score, 0.0)

    def _normalize(self, text: str) -> str:
        return " ".join(text.split()).lower()

    def export_json(self, path: str = "dataset.json") -> None:
        records = [
            {
                "id": ex.id,
                "text": ex.text,
                "labels": ex.labels,
                "example_type": ex.example_type,
                "platform": ex.platform,
                "explicitness": ex.explicitness,
                "severity": ex.severity,
                "slang_used": ex.slang_used,
                "synthetic": ex.synthetic,
            }
            for ex in self._examples
        ]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(records, f, ensure_ascii=False, indent=2)
        logger.info("Saved %d examples → %s", len(records), path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

# Import canonical path from config so run_pipeline.py and standalone use agree
from src.config import DATASET_OUTPUT_PATH  # noqa: E402


def main() -> None:
    if os.environ.get("GEMINI_API_KEY"):
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
            "No LLM provider available (missing GEMINI_API_KEY and OPENROUTER_API_KEY). "
            "Falling back to MockLLMClient — outputs are deterministic fake Hebrew. "
            "Set GEMINI_API_KEY to generate real synthetic data."
        )
        llm = MockLLMClient()

    DATASET_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    examples = generate_dataset(llm, output_path=str(DATASET_OUTPUT_PATH))
    logger.info("Generated %d synthetic examples.", len(examples))


if __name__ == "__main__":
    main()
