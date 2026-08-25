"""OpenRouterProvider tests.

No live network calls: httpx.MockTransport substitutes for the real
OpenRouter endpoint, so these pin the request shape and the
error-to-LLMUnavailable contract deterministically, the same contract
AnthropicProvider gives the rest of the pipeline.
"""

import json

import httpx
import pytest

from app.adapters.llm.base import LLMUnavailable
from app.adapters.llm.openrouter_provider import CHAT_COMPLETIONS_URL, OpenRouterProvider


def _provider_with_transport(handler) -> OpenRouterProvider:
    provider = OpenRouterProvider(api_key="test-key")
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    return provider


def test_structured_sends_json_schema_response_format_and_bearer_auth():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["request"] = request
        body = json.loads(request.content)
        captured["body"] = body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps({"ok": True})}}]},
        )

    provider = _provider_with_transport(handler)
    schema = {"type": "object", "properties": {}}
    result = provider.structured(
        system="sys", user="usr", schema=schema, model="some/model", effort="low"
    )

    assert result == {"ok": True}
    assert captured["request"].url == CHAT_COMPLETIONS_URL
    assert captured["request"].headers["authorization"] == "Bearer test-key"
    body = captured["body"]
    assert body["model"] == "some/model"
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "usr"},
    ]
    assert body["response_format"]["type"] == "json_schema"
    assert body["response_format"]["json_schema"]["schema"] == schema
    assert body["reasoning_effort"] == "low"


def test_effort_omitted_when_not_set():
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        assert "reasoning_effort" not in body
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": "{}"}}]},
        )

    provider = _provider_with_transport(handler)
    provider.structured(system="s", user="u", schema={}, model="m")


def test_http_error_becomes_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, text="upstream error")

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailable):
        provider.structured(system="s", user="u", schema={}, model="m")


def test_unexpected_response_shape_becomes_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"unexpected": "shape"})

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailable):
        provider.structured(system="s", user="u", schema={}, model="m")


def test_unparseable_json_content_becomes_llm_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "not json"}}]}
        )

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailable):
        provider.structured(system="s", user="u", schema={}, model="m")
