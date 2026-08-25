"""Local-model-backed structured completions, via Ollama.

Third implementation of the `LLMProvider` protocol -- same OpenAI-compatible
chat-completions shape as `OpenRouterProvider`, pointed at a model running on
this machine instead of a paid API. No API key: Ollama's local server does
not require one.

This exists for two reasons: it costs nothing per call, so it is the right
tool for iterating on a bug rather than spending API budget on every attempt
(that is how `retrieval.sanitize_categories()` got root-caused -- fuzzing
this provider against the real pipeline surfaced the same "unrecognized
filters.categories value" failure a live Anthropic call had shown
intermittently, for free); and it means the app runs end to end with zero
external dependency when a local model is good enough for the task.

Structured-output reliability is materially weaker than Anthropic's here,
proportional to the local model's size -- expect a small model (3B class) to
occasionally miss the taxonomy entirely on judgement-heavy fields. That is a
feature for bug-hunting (it fuzzes harder than a well-behaved model would)
and a real trade-off for production use.
"""

import json
import logging
from typing import Any

import httpx

from app.adapters.llm.base import LLMUnavailable

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/v1/chat/completions"


class LocalProvider:
    name = "local"
    is_real = True

    def __init__(self, model: str | None = None, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 120.0):
        # `model` is accepted for symmetry with the other providers'
        # constructors but unused: the model actually served comes from the
        # `model` argument to `structured()` on every call (INTERPRET_MODEL),
        # same as Anthropic and OpenRouter -- Ollama has no separate
        # per-client model selection to configure ahead of time.
        self._base_url = base_url
        self._client = httpx.Client(timeout=timeout_s)

    def structured(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        max_tokens: int = 4000,
        timeout_s: float | None = None,
        effort: str | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_query",
                    "schema": schema,
                    "strict": True,
                },
            },
        }
        # `effort` has no Ollama/local-model equivalent -- there is no
        # request field for it, so it is silently ignored rather than sent
        # as a parameter no local model understands.

        kwargs: dict[str, Any] = {}
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s

        try:
            response = self._client.post(self._base_url, json=payload, **kwargs)
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(
                f"{model} call failed (is `ollama serve` running?): {exc}"
            ) from exc

        body = response.json()
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMUnavailable(f"{model} returned an unexpected response shape") from exc
        if not text:
            raise LLMUnavailable(f"{model} returned no text content")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"{model} returned unparseable JSON") from exc

    def close(self) -> None:
        self._client.close()
