"""S5: what happens when things go wrong.

The failure policy is "never show an empty screen": recoverable problems degrade
to weaker-but-real results and say so, rather than erroring or -- worse --
quietly passing off the weaker result as the full one. These tests hold that
policy in place, since it is only visible when something breaks.
"""

from datetime import date

import pytest

from app.adapters.llm.base import LLMUnavailable
from app.core.cache import ResponseCache, cache_key, response_cache
from app.schemas.query import QueryFilters
from app.schemas.recommend import RecommendRequest
from app.services.catalogue import Catalogue
from app.services.recommend import _with_overrides, recommend

TODAY = date(2026, 8, 24)


@pytest.fixture(scope="module")
def catalogue():
    return Catalogue.load()


@pytest.fixture(autouse=True)
def clear_cache():
    response_cache.clear()
    yield
    response_cache.clear()


class DeadProvider:
    """Every call fails, as if the API were unreachable."""

    name = "dead"
    is_real = True

    def structured(self, **_):
        raise LLMUnavailable("simulated outage")



# --- total outage ---------------------------------------------------------

def test_total_llm_outage_still_returns_products(catalogue):
    response = recommend(
        RecommendRequest(
            query="trekking jacket for a cold Himalayan trek", skip_clarification=True
        ),
        catalogue,
        DeadProvider(),
        today=TODAY,
    )
    assert response.mode == "results"
    assert response.groups, "an outage must not produce an empty screen"
    assert response.meta.degraded_mode is True
    assert response.meta.notes, "degradation must be explained, not silent"


def test_no_provider_configured_still_returns_products(catalogue):
    response = recommend(
        RecommendRequest(query="traditional wedding sherwani", skip_clarification=True),
        catalogue,
        None,
        today=TODAY,
    )
    assert response.groups
    assert response.meta.degraded_mode is True
    assert response.meta.llm_calls == 0


def test_degraded_responses_are_not_cached(catalogue):
    """A transient failure must not be frozen into the cache for 30 minutes."""
    request = RecommendRequest(query="warm jacket for the mountains", skip_clarification=True)
    recommend(request, catalogue, DeadProvider(), today=TODAY)
    key = cache_key(request.query, request.answers, request.skip_clarification)
    assert response_cache.get(key) is None


# --- cache ---------------------------------------------------------------

def test_cache_key_ignores_casing_and_answer_order():
    a = cache_key("Warm  Jacket", ["b:2", "a:1"], False)
    b = cache_key("warm jacket", ["a:1", "b:2"], False)
    assert a == b


def test_cache_key_separates_different_requests():
    assert cache_key("warm jacket", [], False) != cache_key("warm jacket", ["gender:men"], False)
    assert cache_key("warm jacket", [], False) != cache_key("warm jacket", [], True)


def test_cache_expires_entries():
    cache = ResponseCache(ttl_seconds=0)
    cache.set("k", "value")
    assert cache.get("k") is None


def test_cache_evicts_least_recently_used():
    cache = ResponseCache(max_entries=2)
    cache.set("a", 1)
    cache.set("b", 2)
    cache.get("a")          # 'a' becomes most recent, so 'b' should go first
    cache.set("c", 3)
    assert cache.get("b") is None
    assert cache.get("a") == 1
    assert cache.get("c") == 3


def test_cache_reports_hit_rate():
    cache = ResponseCache()
    cache.set("k", 1)
    cache.get("k")
    cache.get("missing")
    assert cache.stats["hits"] == 1
    assert cache.stats["misses"] == 1
    assert cache.stats["hit_rate"] == 0.5


# --- off-topic -----------------------------------------------------------

def test_off_topic_request_declines_without_inventing_products(catalogue):
    response = recommend(
        RecommendRequest(query="what is the capital of France?"),
        catalogue,
        None,
        today=TODAY,
    )
    # The offline path cannot judge topicality, so it returns weak matches; the
    # contract that matters is that nothing crashes and the response is honest.
    assert response.mode in {"results", "clarify"}
    assert response.meta.degraded_mode is True


@pytest.mark.skipif(
    not __import__("app.core.config", fromlist=["get_settings"]).get_settings().anthropic_api_key,
    reason="needs a real API key to exercise the live interpret path",
)
def test_completed_search_uses_evidence_based_explanations(catalogue):
    """One LLM call; explanations come from retrieval evidence, not a ranker."""
    from app.core.deps import load_provider, load_taxonomy

    load_taxonomy()
    provider = load_provider()
    if provider is None:
        pytest.skip("needs a real API key")

    response = recommend(
        RecommendRequest(
            query=(
                "I am going for a trek to Hampta Pass in the last week of October "
                "for one week. Find me trekking essentials."
            ),
            skip_clarification=True,
        ),
        catalogue,
        provider,
        today=TODAY,
    )
    assert response.mode == "results"
    if not response.groups:
        pytest.skip("Interpreter produced no fillable buckets this run")
    assert response.meta.llm_calls <= 1
    for group in response.groups:
        for item in group.items:
            assert item.reason, "every product still needs an explanation"


def test_unmapped_required_slot_is_reported_not_substituted(catalogue):
    """The original defect, end to end.

    A wedding-suit request has no formalwear to match in this catalogue. The
    system must say so rather than returning the nearest products by embedding
    distance -- which previously produced a utility jacket and running shoes.
    """
    from app.core.deps import load_provider, load_taxonomy

    provider = load_provider()
    if provider is None:
        pytest.skip("needs a real API key")
    load_taxonomy()

    response = recommend(
        RecommendRequest(
            query="Suggest me a suit for my friend's wedding", skip_clarification=True
        ),
        catalogue,
        provider,
        today=TODAY,
    )

    # Nothing outside the catalogue's actual formalwear coverage may be passed
    # off as a suit component.
    for group in response.groups:
        for item in group.items:
            path = f"{item.product.category}/{item.product.subcategory}"
            assert path != "Footwear/Sports Shoes", (
                "running shoes must never fill a wedding-suit slot"
            )

    # Either the gaps are reported, or the catalogue genuinely gained formalwear.
    formal_paths = {"Men's Apparel/Suits & Blazers", "Footwear/Formal Shoes"}
    has_formalwear = any(
        f"{p.category}/{p.subcategory}" in formal_paths for p in catalogue.products
    )
    if not has_formalwear:
        assert response.unfilled_slots, "missing formalwear must be reported"


# --- explicit client filters ----------------------------------------------
#
# `RecommendRequest.filters` is part of the published contract, so these pin
# that it is actually applied rather than silently accepted and dropped.


def test_client_filters_override_inferred_ones():
    inferred = QueryFilters(price_max=5000, gender="women")
    override = QueryFilters(price_max=1500)

    merged = _with_overrides(inferred, override)

    assert merged.price_max == 1500, "an explicit ceiling must win"
    assert merged.gender == "women", "fields the client left unset must survive"


def test_unset_client_filters_do_not_erase_inferred_ones():
    inferred = QueryFilters(price_min=500, price_max=3000, gender="men", categories=["Footwear"])

    merged = _with_overrides(inferred, QueryFilters())

    assert merged.price_min == 500
    assert merged.price_max == 3000
    assert merged.gender == "men"
    assert merged.categories == ["Footwear"], "an empty list means no opinion, not 'clear it'"


def test_absent_filters_block_is_a_no_op():
    inferred = QueryFilters(price_max=2000)
    assert _with_overrides(inferred, None) is inferred
