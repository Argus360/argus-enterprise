"""The verifier providers the desktop app can be pointed at.

The engine already speaks to four kinds of verifier (gate.py: ollama/local,
openrouter, anthropic, openai_compatible). What it never had was a way for
the OPERATOR to choose one: the app hardcoded Ollama on the local box, which
on a five-vCPU server with no GPU meant 30 to 90 s per verdict and a breaker
that opened after two of them (CHI pilot, 8 Oct 2026). This catalogue is the
one list the app, the API and the settings UI share — add a row here and
every surface knows about it.

A row maps an operator-facing choice onto what the engine needs:

  engine_provider  the --gate-provider value
  base_url         the --gate-base-url value ("" = the provider's own default)
  key_env          the environment variable the engine reads the key from
                   ("" = no key: the local runtime)
  mapper           (provider, base_url) scene mapping runs on; the mapper
                   does not know "openrouter", so that row routes it through
                   the OpenAI-compatible path at OpenRouter's base URL
  local            True when the provider is the bundled Ollama runtime and
                   the app must start it before the engine
"""
from __future__ import annotations

from dataclasses import dataclass, field

from cvti.contracts import LOCAL_VLM_MODEL

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GEMINI_BASE_URL = "https://generativelanguage.googleapis.com/v1beta/openai"


@dataclass(frozen=True)
class Provider:
    id: str
    label: str
    engine_provider: str
    default_model: str
    base_url: str = ""
    key_env: str = ""
    mapper: tuple[str, str] = ("", "")
    local: bool = False
    # Every environment variable the engine process needs set to the key.
    # The gate and the scene mapper read different names for the same cloud
    # (OPENROUTER_API_KEY vs OPENAI_API_KEY), so one key may fan out to two.
    key_envs: tuple[str, ...] = field(default=())
    note: str = ""
    # Most images one request may carry, when the provider documents a limit.
    # None = send every evidence frame (the gate's default).
    max_frames: int | None = None

    @property
    def needs_key(self) -> bool:
        return bool(self.key_env)

    @property
    def cloud(self) -> bool:
        return not self.local

    def public(self) -> dict:
        """What the UI is shown. Never a key."""
        return {"id": self.id, "label": self.label, "default_model": self.default_model,
                "needs_key": self.needs_key, "local": self.local,
                "custom_base_url": self.id == "custom", "note": self.note}


PROVIDERS: dict[str, Provider] = {
    "ollama": Provider(
        id="ollama", label="On this computer (Ollama)", engine_provider="ollama",
        default_model=LOCAL_VLM_MODEL, local=True,
        mapper=("ollama", ""),
        note="Runs the vision model on this computer. Needs a GPU or a large CPU; "
             "on a small server verdicts take a minute or more."),
    "openrouter": Provider(
        id="openrouter", label="OpenRouter", engine_provider="openrouter",
        default_model="google/gemini-2.5-flash-lite", base_url="",
        key_env="OPENROUTER_API_KEY",
        key_envs=("OPENROUTER_API_KEY", "OPENAI_API_KEY"),
        mapper=("openai_compatible", OPENROUTER_BASE_URL),
        note="One key for many models. Turn off training-capable providers in the "
             "OpenRouter privacy settings before sending customer frames."),
    "groq": Provider(
        id="groq", label="Groq", engine_provider="openai_compatible",
        default_model="qwen/qwen3.8-27b", base_url=GROQ_BASE_URL,
        key_env="OPENAI_API_KEY", key_envs=("OPENAI_API_KEY",),
        mapper=("openai_compatible", GROQ_BASE_URL),
        # Groq caps a request at 3 images and bills 2,048 tokens each; the free
        # tier is 8,000 tokens a minute and 200,000 a day, about one verdict a
        # minute and under thirty a day with three frames.
        max_frames=3,
        note="Fast open models; the free tier does not train on inputs but allows "
             "about 30 verdicts a day with images. Good for a test, thin for a site."),
    "gemini": Provider(
        id="gemini", label="Google Gemini", engine_provider="openai_compatible",
        default_model="gemini-2.5-flash-lite", base_url=GEMINI_BASE_URL,
        key_env="OPENAI_API_KEY", key_envs=("OPENAI_API_KEY",),
        mapper=("openai_compatible", GEMINI_BASE_URL),
        note="Use a paid-tier key: Google's free tier may train on what it is sent."),
    "anthropic": Provider(
        id="anthropic", label="Anthropic Claude", engine_provider="anthropic",
        default_model="claude-haiku-4-5", key_env="ANTHROPIC_API_KEY",
        key_envs=("ANTHROPIC_API_KEY",),
        mapper=("anthropic", ""),
        note="Strongest read of a scene; about ten times the cost of Flash-Lite."),
    "custom": Provider(
        id="custom", label="Custom OpenAI-compatible endpoint", engine_provider="openai_compatible",
        default_model="", base_url="", key_env="OPENAI_API_KEY", key_envs=("OPENAI_API_KEY",),
        mapper=("openai_compatible", ""),
        note="Any server that speaks the OpenAI chat API with images, including "
             "Argus's own verification service."),
}

DEFAULT_PROVIDER_ID = "ollama"


def get_provider(provider_id: str | None) -> Provider:
    """The catalogue row for an id; unknown or empty ids mean the local default."""
    return PROVIDERS.get((provider_id or "").strip().lower(), PROVIDERS[DEFAULT_PROVIDER_ID])


def normalize_settings(raw: dict | None) -> dict:
    """The site's `gate` block, made whole: provider id, model, base_url.

    Missing model falls back to the provider's default; base_url is only
    honoured for the custom row (every other row knows its own)."""
    raw = raw if isinstance(raw, dict) else {}
    spec = get_provider(raw.get("provider"))
    model = str(raw.get("model") or "").strip() or spec.default_model
    base_url = str(raw.get("base_url") or "").strip() if spec.id == "custom" else spec.base_url
    return {"provider": spec.id, "model": model, "base_url": base_url}


def engine_args(settings: dict) -> list[str]:
    """The --gate-* and --mapper-* flags for a normalised settings dict."""
    spec = get_provider(settings.get("provider"))
    args = ["--gate-provider", spec.engine_provider, "--gate-model", settings.get("model") or spec.default_model]
    base_url = settings.get("base_url") or spec.base_url
    if base_url and spec.engine_provider == "openai_compatible":
        args += ["--gate-base-url", base_url]
    if spec.max_frames:
        args += ["--gate-max-frames", str(spec.max_frames)]
    mapper_provider, mapper_base = spec.mapper
    if spec.id == "custom":
        mapper_base = base_url
    if mapper_provider and mapper_provider != spec.engine_provider:
        args += ["--mapper-provider", mapper_provider]
    if mapper_base and mapper_base != base_url:
        args += ["--mapper-base-url", mapper_base]
    return args


def key_environment(settings: dict, api_key: str | None) -> dict[str, str]:
    """Environment variables carrying the key to the engine. Empty for local."""
    spec = get_provider(settings.get("provider"))
    if not spec.needs_key or not api_key:
        return {}
    return {name: api_key for name in (spec.key_envs or (spec.key_env,))}
