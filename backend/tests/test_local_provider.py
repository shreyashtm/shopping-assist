"""LocalProvider tests.

No live Ollama server needed: httpx.MockTransport substitutes for it, so
these pin the request shape and the error-to-LLMUnavailable contract
deterministically -- same contract AnthropicProvider and OpenRouterProvider
give the rest of the pipeline.
"""

import json

import httpx
import pytest

from app.adapters.llm.base import LLMUnavailable
from app.adapters.llm.local_provider import LocalProvider


def _provider_with_transport(handler) -> LocalProvider:
    provider = LocalProvider()
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    return provider


def test_structured_sends_json_schema_response_format_no_auth_header():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={"message": {"content": json.dumps({"ok": True})}},
        )

    provider = _provider_with_transport(handler)
    schema = {"type": "object", "properties": {}}
    result = provider.structured(system="sys", user="usr", schema=schema, model="llama3.2:3b")

    assert result == {"ok": True}
    assert "authorization" not in {k.lower() for k in captured["request"].headers.keys()}, (
        "local calls need no auth header"
    )
    body = captured["body"]
    assert body["model"] == "llama3.2:3b"
    assert body["format"] == schema
    assert body["think"] is False and body["stream"] is False


def test_the_context_window_fits_the_prompt():
    """Live: Ollama's default 4,096-token window cut the ~3,000-token prompt
    plus answer, and qwen3:8b planned thermals for a Goa beach trip."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"message": {"content": "{}"}})

    _provider_with_transport(handler).structured(system="s", user="u", schema={}, model="m")
    assert captured["body"]["options"]["num_ctx"] >= 8192


def test_an_old_openai_compatible_url_is_moved_to_the_native_endpoint():
    provider = LocalProvider(base_url="http://localhost:11434/v1/chat/completions")
    assert provider._base_url == "http://localhost:11434/api/chat"


def test_http_error_becomes_llm_unavailable_with_a_helpful_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="connection refused")

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailable, match="ollama serve"):
        provider.structured(system="s", user="u", schema={}, model="m")


def test_unparseable_json_content_becomes_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"message": {"content": "not json"}}
        )

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailable):
        provider.structured(system="s", user="u", schema={}, model="m")
