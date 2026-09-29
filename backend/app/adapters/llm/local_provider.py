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

from app.adapters.llm.base import LLMUnavailable, post_json_with_deadline

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "http://localhost:11434/api/chat"

# Ollama loads a model with a 4,096-token window unless asked for more. The
# interpretation prompt (instructions, live taxonomy, schema) is about 3,000
# tokens before the answer, so the default silently cut the prompt: qwen3:8b
# planned thermals for a Goa beach trip and invented shelf names.
NUM_CTX = 12288


def _native_url(base_url: str) -> str:
    """Ollama's native chat endpoint for a configured base URL.

    Earlier configs pointed at the OpenAI-compatible /v1/chat/completions,
    which cannot set the context window or turn thinking off.
    """
    if "/v1/" in base_url:
        return base_url.split("/v1/")[0] + "/api/chat"
    return base_url


class LocalProvider:
    name = "local"
    is_real = True

    def __init__(self, model: str | None = None, base_url: str = DEFAULT_BASE_URL, timeout_s: float = 120.0):
        # `model` is accepted for symmetry with the other providers'
        # constructors but unused: the model actually served comes from the
        # `model` argument to `structured()` on every call (INTERPRET_MODEL),
        # same as Anthropic and OpenRouter -- Ollama has no separate
        # per-client model selection to configure ahead of time.
        self._base_url = _native_url(base_url)
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
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            # Ollama constrains decoding to the schema, so the reply always
            # parses; the shape is still validated by the interpreter.
            "format": schema,
            "stream": False,
            # Reasoning models (qwen3) otherwise think first: 40-55s a search
            # instead of well under half that, for a slot-filling task.
            "think": False,
            "options": {"num_ctx": NUM_CTX, "num_predict": max_tokens},
        }
        # `effort` has no Ollama equivalent beyond `think`, which is always
        # off here, so it is ignored rather than sent.

        try:
            body = post_json_with_deadline(
                self._client, self._base_url, timeout_s, json=payload
            )
        except httpx.HTTPError as exc:
            raise LLMUnavailable(
                f"{model} call failed (is `ollama serve` running?): {exc}"
            ) from exc

        if isinstance(body, dict) and body.get("error"):
            raise LLMUnavailable(f"{model} error: {body['error']}")
        try:
            text = body["message"]["content"]
        except (KeyError, TypeError) as exc:
            raise LLMUnavailable(f"{model} returned an unexpected response shape") from exc
        if not text:
            raise LLMUnavailable(f"{model} returned no text content")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise LLMUnavailable(f"{model} returned unparseable JSON") from exc

    def close(self) -> None:
        self._client.close()
