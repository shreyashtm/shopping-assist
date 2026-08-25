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
        """`chain` is ordered (provider, model) pairs, first tried first.

        Falling through is driven by *failure*, never by elapsed time. Every
        hop gets the caller's full timeout, and only an `LLMUnavailable` --
        auth rejection, exhausted credit, rate limit, transport error, or the
        provider's own timeout expiring -- moves to the next one.

        An earlier version also abandoned a hop after a short deadline, on the
        theory that a slow primary should not delay a working fallback. That
        was wrong and broke a healthy deployment: a real interpretation call
        takes 10.7-11.4s (5,600-character system prompt plus the live
        taxonomy), so the 8s deadline aborted a provider that was about to
        succeed, and every search fell through to keyword matching while the
        provider was fine. Slow is not the same as broken, and only the
        provider can report broken.
        """
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
