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
import time
from typing import Any

import httpx

from app.adapters.llm.base import LLMUnavailable, post_json_with_deadline

logger = logging.getLogger(__name__)

CHAT_COMPLETIONS_URL = "https://openrouter.ai/api/v1/chat/completions"

# A hop with less time than this left is not started: no model answers a
# full interpretation that fast, so it would only delay the fallback.
MIN_HOP_S = 5.0


class OpenRouterProvider:
    name = "openrouter"
    is_real = True

    def __init__(
        self, api_key: str, timeout_s: float = 60.0, fallback_models: list[str] | None = None
    ):
        self._api_key = api_key
        self._client = httpx.Client(timeout=timeout_s)
        self._fallback_models = fallback_models or []

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
        """Try `model`, then each fallback model, within one shared deadline.

        Free models fail often and independently of each other: a 429 on one,
        an overloaded upstream on another, an empty reply on a third. Each of
        those is an `LLMUnavailable`, and the next model in the chain gets the
        time that is left. Only when every model has failed, or the deadline
        is spent, does the caller degrade to keyword matching.
        """
        started = time.monotonic()
        errors: list[str] = []
        chain = list(dict.fromkeys([model, *self._fallback_models]))
        for hop in chain:
            remaining = None if timeout_s is None else timeout_s - (time.monotonic() - started)
            if errors and remaining is not None and remaining < MIN_HOP_S:
                errors.append(f"{hop}: no time left")
                break
            try:
                return self._call_once(
                    system=system,
                    user=user,
                    schema=schema,
                    model=hop,
                    max_tokens=max_tokens,
                    timeout_s=remaining,
                    effort=effort,
                )
            except LLMUnavailable as exc:
                errors.append(str(exc))
                if hop != chain[-1]:
                    logger.warning("%s; trying the next model", exc)
        if len(errors) == 1:
            raise LLMUnavailable(errors[0])
        raise LLMUnavailable("every model failed: " + "; ".join(errors))

    def _call_once(
        self,
        *,
        system: str,
        user: str,
        schema: dict[str, Any],
        model: str,
        max_tokens: int,
        timeout_s: float | None,
        effort: str | None,
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
            # Route only to endpoints that honour every parameter sent, above
            # all `response_format`. Without this OpenRouter may pick an
            # endpoint that ignores the schema and returns free-form text.
            "provider": {"require_parameters": True},
        }
        # Omitted entirely when unset, matching AnthropicProvider: some
        # models reject an unsupported reasoning-effort parameter outright
        # rather than ignoring it.
        if effort:
            payload["reasoning_effort"] = effort


        try:
            body = post_json_with_deadline(
                self._client,
                CHAT_COMPLETIONS_URL,
                timeout_s,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
        except httpx.HTTPError as exc:
            raise LLMUnavailable(f"{model} call failed: {exc}") from exc

        # OpenRouter can answer HTTP 200 with the upstream failure in the body
        # ("Service temporarily overloaded", code 503). Say so, rather than
        # "unexpected response shape", which hid the cause in the logs.
        if isinstance(body, dict) and body.get("error"):
            error = body["error"]
            message = error.get("message", error) if isinstance(error, dict) else error
            raise LLMUnavailable(f"{model} provider error: {message}")
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
