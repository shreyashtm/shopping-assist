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
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
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
    assert body["response_format"]["json_schema"]["schema"] == schema


def test_http_error_becomes_llm_unavailable_with_a_helpful_hint():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="connection refused")

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailable, match="ollama serve"):
        provider.structured(system="s", user="u", schema={}, model="m")


def test_unparseable_json_content_becomes_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "not json"}}]}
        )

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailable):
        provider.structured(system="s", user="u", schema={}, model="m")
