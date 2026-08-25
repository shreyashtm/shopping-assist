"""Anthropic-backed structured completions."""

import json
import logging
from typing import Any

from app.adapters.llm.base import LLMUnavailable

logger = logging.getLogger(__name__)


class AnthropicProvider:
    name = "anthropic"
    is_real = True

    def __init__(self, api_key: str, timeout_s: float = 60.0):
        import anthropic

        self._client = anthropic.Anthropic(api_key=api_key, timeout=timeout_s)
        self._errors = anthropic

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
        client = self._client
        if timeout_s is not None:
            client = client.with_options(timeout=timeout_s)
        try:
            response = client.messages.create(
                model=model,
                max_tokens=max_tokens,
                system=system,
                messages=[{"role": "user", "content": user}],
                # Constrains the response to valid JSON matching the schema, so
                # there is no brittle parsing of prose on the other side.
                # `effort` is omitted entirely when unset -- some models reject
                # the parameter rather than ignoring it.
                output_config=(
                    {"format": {"type": "json_schema", "schema": schema}, "effort": effort}
                    if effort
                    else {"format": {"type": "json_schema", "schema": schema}}
                ),
            )
        except self._errors.APIError as exc:
            raise LLMUnavailable(f"{model} call failed: {exc}") from exc

        text = next((b.text for b in response.content if b.type == "text"), "")
        if not text:
            raise LLMUnavailable(f"{model} returned no text content")
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:  # pragma: no cover - schema prevents this
            raise LLMUnavailable(f"{model} returned unparseable JSON") from exc
