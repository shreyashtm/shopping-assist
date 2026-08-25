"""LLM transport interface.

Only transport lives here -- prompts and domain logic stay in services/. That
split is what lets the stub below be a genuine drop-in: it satisfies the same
interface without knowing anything about shopping.
"""

from typing import Any, Protocol


class LLMProvider(Protocol):
    name: str
    is_real: bool
    """False for the deterministic stub, so responses can be flagged degraded."""

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
        """Return JSON matching `schema`. Raises on transport failure."""
        ...


class LLMUnavailable(RuntimeError):
    """Raised when a real provider cannot serve a request.

    Callers degrade rather than fail: the app's contract is that it always
    returns products, and says when it did so without full reasoning.
    """
