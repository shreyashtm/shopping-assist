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


def post_json_with_deadline(client, url: str, deadline_s: float | None, **kwargs) -> Any:
    """POST and return the decoded JSON body, giving up after `deadline_s` in total.

    httpx's `timeout` limits each network step -- connecting, and the gap
    between bytes -- not the whole request. OpenRouter keeps slow requests
    open by trickling whitespace, so a 75s timeout never fired: a live search
    sat at "Reading your request" for over two minutes. Streaming the body
    and checking the clock between chunks makes the deadline real, so a slow
    model fails as LLMUnavailable and the caller degrades to keyword matching.

    Raises httpx.HTTPError for transport and HTTP-status failures (callers
    already translate those), and LLMUnavailable for the deadline and for a
    body that is not JSON.
    """
    import json
    import time

    started = time.monotonic()
    chunks: list[bytes] = []
    with client.stream("POST", url, **kwargs) as response:
        response.raise_for_status()
        for chunk in response.iter_bytes():
            chunks.append(chunk)
            if deadline_s is not None and time.monotonic() - started > deadline_s:
                raise LLMUnavailable(f"timed out after {deadline_s:.0f}s waiting for the model")
    try:
        return json.loads(b"".join(chunks))
    except ValueError as exc:
        raise LLMUnavailable("returned a response that is not JSON") from exc
