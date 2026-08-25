"""interpret() tests -- the boundary where raw model output enters the
system. Fuzzed against local models while chasing an intermittent
"zero results" bug; each test here pins one concrete way the raw output
needed normalizing before anything downstream could trust it.
"""

from datetime import date

from app.services.interpreter import interpret


class _FixedProvider:
    """Returns one canned StructuredQuery payload, ignoring the prompt."""

    name = "fixed"
    is_real = True

    def __init__(self, payload: dict):
        self._payload = payload

    def structured(self, **_):
        return self._payload


_BASE_PAYLOAD = {
    "intent_summary": "A jacket.",
    "is_shopping_request": True,
    "confidence": 0.7,
    "needs_clarification": False,
    "questions": [],
    "buckets": [
        {
            "name": "Jackets",
            "search_phrases": ["jacket"],
            "why_needed": "Requested.",
            "role": "required",
            "catalogue_paths": ["Men's Apparel/Jackets & Coats"],
            "priority": 1,
            "max_items": 4,
        }
    ],
    "filters": {"price_min": None, "price_max": None, "gender": None, "categories": []},
    "context": {
        "location": None, "location_lat": None, "location_lon": None,
        "elevation_estimate_m": None, "start_date": None, "end_date": None,
        "duration_days": None, "climate_note": None, "recipient": None,
    },
    "assumptions": [],
}


def test_interpret_normalizes_a_nonstandard_gender_word():
    """A local model returned filters.gender="neutral" directly -- not a
    tapped answer, the model's own first-pass output. merge_answers()
    already normalizes this class of value for chip answers; interpret()
    needs the same protection for the model's own output, or "neutral"
    reaches passes_filters() and matches no product's gender attribute at
    all, silently emptying every bucket that has a gender filter."""
    payload = {**_BASE_PAYLOAD, "filters": {**_BASE_PAYLOAD["filters"], "gender": "neutral"}}
    provider = _FixedProvider(payload)

    structured = interpret(provider, "any-model", "a jacket", date(2026, 8, 25))

    assert structured.filters.gender == "unisex"


def test_interpret_leaves_a_canonical_gender_value_alone():
    payload = {**_BASE_PAYLOAD, "filters": {**_BASE_PAYLOAD["filters"], "gender": "women"}}
    provider = _FixedProvider(payload)

    structured = interpret(provider, "any-model", "a jacket", date(2026, 8, 25))

    assert structured.filters.gender == "women"


def test_interpret_leaves_gender_unset_when_the_model_did_not_set_it():
    provider = _FixedProvider(_BASE_PAYLOAD)

    structured = interpret(provider, "any-model", "a jacket", date(2026, 8, 25))

    assert structured.filters.gender is None
