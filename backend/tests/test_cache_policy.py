"""What may and may not be written to the response cache.

Two classes of result are transient and must never be cached, because caching
one keeps serving it long after the cause has cleared:

* a degraded response (the LLM was unreachable), and
* a response that filled nothing.

The second was found in production. The flagship demo query -- a Hampta Pass
trek -- returned 0 groups and 9 unfilled slots because the model happened to
produce a plan whose buckets nothing could fill. That is non-determinism, not a
catalogue fact: rewording the same request returned 9 populated groups. But the
empty answer had been cached, so *every* later run of the original wording
served it back, turning a one-off bad plan into a permanent wrong answer for
that exact query.
"""

from app.schemas.recommend import RecommendResponse, ResponseMeta, UnfilledSlot
from app.services.recommend import _cache_and_return


def _meta(**overrides) -> ResponseMeta:
    fields = dict(
        latency_ms=100, llm_calls=1, cached=False, degraded_mode=False,
        catalogue_size=1738, notes=[],
    )
    fields.update(overrides)
    return ResponseMeta(**fields)


def _response(groups: list, unfilled: list | None = None, **meta) -> RecommendResponse:
    return RecommendResponse(
        query_id="test",
        mode="results",
        intent_summary="test",
        groups=groups,
        unfilled_slots=unfilled or [],
        meta=_meta(**meta),
    )


class _SpyCache:
    def __init__(self):
        self.written = []

    def set(self, key, value):
        self.written.append((key, value))


def test_a_response_that_filled_nothing_is_not_cached(monkeypatch):
    import app.services.recommend as recommend

    spy = _SpyCache()
    monkeypatch.setattr(recommend, "response_cache", spy)

    empty = _response(
        groups=[],
        unfilled=[UnfilledSlot(name="Footwear", role="required", reason="nothing matched")],
    )
    _cache_and_return("k", empty)

    assert spy.written == [], "cached an empty result; a bad plan becomes permanent"


def test_a_degraded_response_is_not_cached(monkeypatch):
    import app.services.recommend as recommend

    spy = _SpyCache()
    monkeypatch.setattr(recommend, "response_cache", spy)

    _cache_and_return("k", _response(groups=[], degraded_mode=True))

    assert spy.written == []


def test_a_response_with_groups_is_cached(monkeypatch):
    """The optimisation must still work -- only transient emptiness is skipped."""
    import app.services.recommend as recommend
    from app.schemas.recommend import RecommendationGroup

    spy = _SpyCache()
    monkeypatch.setattr(recommend, "response_cache", spy)

    good = _response(
        groups=[RecommendationGroup(name="Layering", why_needed="cold", items=[])]
    )
    _cache_and_return("k", good)

    assert len(spy.written) == 1


def test_a_clarify_response_is_still_cacheable(monkeypatch):
    """A clarify turn legitimately has no groups yet -- it is asking, not
    failing -- so the empty-result rule must not suppress it."""
    import app.services.recommend as recommend

    spy = _SpyCache()
    monkeypatch.setattr(recommend, "response_cache", spy)

    clarify = _response(groups=[])
    clarify = clarify.model_copy(update={"mode": "clarify"})
    _cache_and_return("k", clarify)

    assert len(spy.written) == 1
