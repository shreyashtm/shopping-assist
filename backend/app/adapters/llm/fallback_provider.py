"""Tries a chain of providers in order, first success wins.

Exists because "unavailable" has two causes that both need the same
response: no key configured at all, or a live call failing (rate limit,
transport error). Either way the right move is the same -- try the next
configured provider before giving up to keyword interpretation.

Each hop carries its own model id rather than sharing one: a model valid on
Anthropic is meaningless on OpenRouter and vice versa, so reusing a single
`interpret_model` across providers would just move the mismatch bug from one
provider to the fallback path instead of fixing it.
"""

import logging
from typing import Any

from app.adapters.llm.base import LLMProvider, LLMUnavailable

logger = logging.getLogger(__name__)


class FallbackProvider:
    name = "fallback"
    is_real = True

    def __init__(self, chain: list[tuple[LLMProvider, str]]):
        """`chain` is ordered (provider, model) pairs, first tried first."""
        if not chain:
            raise ValueError("FallbackProvider needs at least one (provider, model) pair")
        self._chain = chain

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
        # `model` is accepted only to satisfy the LLMProvider protocol -- each
        # hop uses its own model from `self._chain`, never the caller's value.
        del model
        errors: list[str] = []
        for provider, provider_model in self._chain:
            try:
                return provider.structured(
                    system=system,
                    user=user,
                    schema=schema,
                    model=provider_model,
                    max_tokens=max_tokens,
                    timeout_s=timeout_s,
                    effort=effort,
                )
            except LLMUnavailable as exc:
                errors.append(f"{provider.name}: {exc}")
                logger.warning(
                    "%s unavailable (%s); trying next provider in the fallback chain",
                    provider.name,
                    exc,
                )
        raise LLMUnavailable(f"All providers in fallback chain failed: {'; '.join(errors)}")
