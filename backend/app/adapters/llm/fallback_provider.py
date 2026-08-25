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

    def __init__(
        self,
        chain: list[tuple[LLMProvider, str]],
        earlier_hop_timeout_s: float | None = None,
    ):
        """`chain` is ordered (provider, model) pairs, first tried first.

        `earlier_hop_timeout_s` caps how long every hop *except the last* may
        take before being abandoned. Without it a slow primary burns the whole
        request budget before the fallback gets a turn -- measured against a
        free-tier model that took 37-67s per interpretation, so the caller
        waited out the full timeout and only then started the call that would
        actually succeed. The last hop keeps the full timeout: there is nothing
        left to fall back to, so giving up early there only loses answers.
        """
        if not chain:
            raise ValueError("FallbackProvider needs at least one (provider, model) pair")
        self._chain = chain
        self._earlier_hop_timeout_s = earlier_hop_timeout_s

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
        last_index = len(self._chain) - 1
        for index, (provider, provider_model) in enumerate(self._chain):
            hop_timeout = timeout_s
            if index < last_index and self._earlier_hop_timeout_s is not None:
                # Never *raise* the caller's budget -- only shorten it, so a
                # deliberately tight timeout_s is still respected.
                hop_timeout = (
                    self._earlier_hop_timeout_s
                    if timeout_s is None
                    else min(timeout_s, self._earlier_hop_timeout_s)
                )
            try:
                return provider.structured(
                    system=system,
                    user=user,
                    schema=schema,
                    model=provider_model,
                    max_tokens=max_tokens,
                    timeout_s=hop_timeout,
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
