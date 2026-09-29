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


def test_an_error_inside_a_200_response_is_reported_by_its_own_message():
    """Live: OpenRouter answered HTTP 200 with {"error": {"message": "Upstream
    error from Nvidia: Service temporarily overloaded", "code": 503}} and the
    log said only "returned an unexpected response shape"."""
    from app.adapters.llm.base import LLMUnavailable

    def handler(request):
        return httpx.Response(200, json={"id": "gen-1", "error": {
            "message": "Upstream error from Nvidia: Service temporarily overloaded", "code": 503}})

    provider = _provider_with_transport(handler)
    with pytest.raises(LLMUnavailable, match="Service temporarily overloaded"):
        provider.structured(system="s", user="u", schema={"type": "object"}, model="m",
                            max_tokens=10, timeout_s=None, effort=None)


def test_only_endpoints_that_enforce_the_schema_are_used():
    """Live: nemotron-3-ultra supports neither response_format nor structured
    outputs, and returned unparseable JSON. require_parameters stops OpenRouter
    routing to an endpoint that silently ignores the schema."""
    captured = {}

    def handler(request):
        captured["body"] = json.loads(request.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "{}"}}]})

    _provider_with_transport(handler).structured(system="s", user="u", schema={}, model="m")
    assert captured["body"]["provider"] == {"require_parameters": True}


def _chain_provider(handler, fallback_models) -> OpenRouterProvider:
    provider = OpenRouterProvider(api_key="k", fallback_models=fallback_models)
    provider._client = httpx.Client(transport=httpx.MockTransport(handler))
    return provider


def test_a_failing_model_hands_over_to_the_next_one():
    """Live: free models failed with 429, an overloaded upstream, and empty
    replies, each sending the search to keyword mode though another free
    model was answering fine."""
    tried = []

    def handler(request):
        model = json.loads(request.content)["model"]
        tried.append(model)
        if model == "rate-limited":
            return httpx.Response(429, text="Too Many Requests")
        if model == "overloaded":
            return httpx.Response(200, json={"error": {"message": "overloaded", "code": 503}})
        if model == "empty":
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}}]})
        return httpx.Response(200, json={"choices": [{"message": {"content": '{"ok": 1}'}}]})

    provider = _chain_provider(handler, ["overloaded", "empty", "works", "never-reached"])
    result = provider.structured(system="s", user="u", schema={}, model="rate-limited", timeout_s=60)
    assert result == {"ok": 1}
    assert tried == ["rate-limited", "overloaded", "empty", "works"]


def test_when_every_model_fails_the_error_names_each_one():
    def handler(request):
        return httpx.Response(429, text="Too Many Requests")

    provider = _chain_provider(handler, ["b"])
    with pytest.raises(LLMUnavailable, match="every model failed") as info:
        provider.structured(system="s", user="u", schema={}, model="a", timeout_s=60)
    assert "a call failed" in str(info.value) and "b call failed" in str(info.value)


def test_a_later_model_is_not_started_without_time_to_answer():
    """The chain shares one deadline: a hop with seconds left cannot finish,
    and would only delay the keyword fallback."""
    tried = []

    def handler(request):
        tried.append(json.loads(request.content)["model"])
        return httpx.Response(429, text="Too Many Requests")

    provider = _chain_provider(handler, ["b"])
    with pytest.raises(LLMUnavailable, match="no time left"):
        provider.structured(system="s", user="u", schema={}, model="a", timeout_s=1)
    assert tried == ["a"]


def test_the_timeout_is_a_total_deadline_not_a_gap_between_bytes():
    """Live: a search sat at "Reading your request" for over two minutes
    with a 75s timeout. OpenRouter keeps slow requests open by trickling
    whitespace, and httpx's timeout only limits the gap between bytes, so
    it never fired. The request must give up at the deadline and let the
    caller degrade to keyword matching."""
    import time as _time

    from app.adapters.llm.base import LLMUnavailable

    def trickle():
        for _ in range(100):  # ~5s of keep-alive whitespace, 50ms apart
            _time.sleep(0.05)
            yield b" "

    def handler(request):
        return httpx.Response(200, content=trickle())

    provider = _provider_with_transport(handler)
    started = _time.monotonic()
    with pytest.raises(LLMUnavailable, match="timed out"):
        provider.structured(system="s", user="u", schema={"type": "object"}, model="m",
                            max_tokens=10, timeout_s=0.5, effort=None)
    assert _time.monotonic() - started < 1.5
