"""OpenRouter-backed structured completions.

A second implementation of the `LLMProvider` protocol, proving the
abstraction in `adapters/llm/base.py` is a real seam and not just aspiration:
the service layer (`interpreter.py`, `recommend.py`) does not change at all
to use this instead of Anthropic -- only `core/deps.py::load_provider()`
picks which adapter to construct.

OpenRouter exposes an OpenAI-compatible chat-completions endpoint, so
structured output goes through `response_format: json_schema` rather than
Anthropic's `output_config`. Not every model routed through OpenRouter
supports strict JSON-schema output; that surfaces as an API error, which
becomes `LLMUnavailable` like any other transport failure -- the same
contract `AnthropicProvider` gives the rest of the pipeline.
"""

import json
import logging
from typing import Any

import httpx

from app.adapters.llm.base import LLMUnavailable

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"


class OpenRouterProvider:
    name = "openrouter"
    is_real = True

    def __init__(self, api_key: str, timeout_s: float = 60.0):
        self._api_key = api_key
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
        # Omitted entirely when unset, matching AnthropicProvider: some
        # models reject an unsupported reasoning-effort parameter outright
        # rather than ignoring it.
        if effort:
            payload["reasoning_effort"] = effort

        kwargs: dict[str, Any] = {}
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s

        try:
            response = self._client.post(
                CHAT_COMPLETIONS_URL,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                **kwargs,
            )
            response.raise_for_status()
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"{model} call failed: {exc}") from exc

        body = response.json()
        try:
            text = body["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:
            raise LLMUnavailable(f"{model} returned an unexpected response shape") from exc
        if not text:
            raise LLMUnavailable(f"{model} returned no text content")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - schema prevents this
            raise LLMUnavailable(f"{model} returned unparseable JSON") from exc

    def close(self) -> None:
        self._client.close()
