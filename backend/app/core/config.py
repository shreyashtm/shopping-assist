"""Application settings, loaded from environment or backend/.env."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

APP_DIR = Path(__file__).resolve().parent.parent
# The catalogue lives at backend/data/, alongside scripts/ rather than inside
# the importable package: it is build output, not source.
BACKEND_DIR = APP_DIR.parent
DATA_DIR = BACKEND_DIR / "data"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Personal Shopping Assistant"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # Comma-separated in the environment, e.g. "http://localhost:3000,https://foo.vercel.app".
    cors_origins: str = "http://localhost:3000"

    # LLM. Absent key is not an error: the app falls back to keyword
    # interpretation so it stays demoable offline. See adapters/llm/.
    #
    # `llm_provider` picks the adapter; the service layer (interpreter.py,
    # recommend.py) is unaware which one is behind `LLMProvider` and does not
    # change either way. Anthropic remains the default because
    # `interpret_model`'s default ("claude-haiku-4-5") is an Anthropic model
    # id -- switching to "openrouter" should come with an OpenRouter-style
    # model id in `interpret_model` too (e.g. "anthropic/claude-haiku-4.5" or
    # any other OpenRouter-routed model).
    llm_provider: Literal["anthropic", "openrouter"] = "anthropic"
    anthropic_api_key: str | None = None
    openrouter_api_key: str | None = None

    # One model call per completed search, and this is it. The user-visible
    # explanations are composed from retrieval evidence (services/explain.py)
    # rather than by a second call, which is why no ranking model appears here.
    #
    # Interpretation is structured extraction -- fill slots, decide what is
    # missing, generate multiple-choice questions -- which Haiku handles well.
    interpret_model: str = "claude-haiku-4-5"

    # Generous: a slow structured extraction that times out degrades the entire
    # search to keyword matching, which is a far worse outcome than waiting.
    interpret_timeout_s: float = 30.0

    # Default transport timeout applied to the provider client itself.
    llm_timeout_s: float = 90.0

    # Reasoning depth. Interpretation is not a hard reasoning problem, so this
    # stays low wherever it is supported at all.
    #
    # `effort` is not universally supported: Haiku 4.5 rejects it outright with
    # a 400, so it stays None there. None means "do not send the parameter".
    interpret_effort: str | None = None

    # Retrieval. More candidates are pulled per bucket than will be shown; the
    # surplus gives scoring room to reorder within a type-correct set before the
    # group is trimmed to max_items.
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    candidates_per_bucket: int = 12

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
