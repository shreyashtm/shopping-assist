"""Tests for the provider fallback chain.

Pure stub providers -- no network calls -- so these run unconditionally,
unlike test_api_contract.py/test_robustness.py which need a live key.
"""

import pytest

from app.adapters.llm.base import LLMUnavailable
from app.adapters.llm.fallback_provider import FallbackProvider

SCHEMA = {"type": "object", "properties": {}}


class StubProvider:
    """Records the model it was called with; returns a canned result or raises."""

    def __init__(self, name: str, result: dict | None = None, fails: bool = False):
        self.name = name
        self.is_real = True
        self.result = result
        self.fails = fails
        self.called_with_model: str | None = None
        self.call_count = 0

    def structured(
        self, *, system, user, schema, model, max_tokens=4000, timeout_s=None, effort=None
    ):
        self.call_count += 1
        self.called_with_model = model
        self.called_with_timeout = timeout_s
        if self.fails:
            raise LLMUnavailable(f"{self.name} is down")
        return self.result


def test_first_provider_success_short_circuits_the_rest():
    first = StubProvider("first", result={"ok": "first"})
    second = StubProvider("second", result={"ok": "second"})
    chain = FallbackProvider([(first, "model-a"), (second, "model-b")])

    result = chain.structured(system="s", user="u", schema=SCHEMA, model="ignored")

    assert result == {"ok": "first"}
    assert second.call_count == 0


def test_falls_through_to_next_provider_on_llm_unavailable():
    first = StubProvider("first", fails=True)
    second = StubProvider("second", result={"ok": "second"})
    chain = FallbackProvider([(first, "model-a"), (second, "model-b")])

    result = chain.structured(system="s", user="u", schema=SCHEMA, model="ignored")

    assert result == {"ok": "second"}
    assert first.call_count == 1


def test_raises_when_every_provider_in_the_chain_fails():
    first = StubProvider("first", fails=True)
    second = StubProvider("second", fails=True)
    chain = FallbackProvider([(first, "model-a"), (second, "model-b")])

    with pytest.raises(LLMUnavailable):
        chain.structured(system="s", user="u", schema=SCHEMA, model="ignored")


def test_each_hop_gets_its_own_model_not_the_callers():
    """The exact bug this exists to prevent: a model valid for one provider
    is meaningless on another, so the caller's `model` must never leak
    through to a hop that has its own configured model."""
    first = StubProvider("first", fails=True)
    second = StubProvider("second", result={"ok": True})
    chain = FallbackProvider([(first, "primary-model"), (second, "fallback-model")])

    chain.structured(system="s", user="u", schema=SCHEMA, model="caller-supplied-model")

    assert first.called_with_model == "primary-model"
    assert second.called_with_model == "fallback-model"


def test_empty_chain_is_rejected_at_construction():
    with pytest.raises(ValueError):
        FallbackProvider([])


def test_every_hop_gets_the_full_timeout():
    """Falling through is driven by failure, never by elapsed time.

    An earlier version cut earlier hops off at a short deadline so a slow
    primary could not delay a working fallback. That broke a healthy
    deployment: a real interpretation takes 10.7-11.4s, the deadline was 8s,
    so the primary was aborted just before succeeding and every search fell
    through to keyword matching while the provider was fine. Slow is not
    broken, and only the provider can report broken.
    """
    first = StubProvider("first", fails=True)
    last = StubProvider("last", result={"ok": True})
    chain = FallbackProvider([(first, "m1"), (last, "m2")])

    chain.structured(system="s", user="u", schema=SCHEMA, model="x", timeout_s=30.0)

    assert first.called_with_timeout == 30.0
    assert last.called_with_timeout == 30.0
