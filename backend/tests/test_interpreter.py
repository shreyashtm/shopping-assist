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


# --- Budget chips must be satisfiable by the catalogue ---------------------
#
# Real defect: a shopper asked for women's wedding wear, was offered a
# "Rs 2,000-5,000" budget chip, tapped it, and got zero products. The
# catalogue does hold 24 women's ethnic items -- but the most expensive is
# Rs 1,955, so the price filter excluded every one of them. The planner had
# invented a plausible-sounding range it had no way to check, because the
# prompt only ever showed it product *counts*. taxonomy.json already stores a
# price_range per subcategory; it just was not being passed through.


def _taxonomy_with_prices() -> dict:
    return {
        "categories": {
            "Ethnic Wear": {
                "Lehengas": {"count": 8, "price_range": [429, 1955], "viable": True},
                "Sarees": {"count": 8, "price_range": [500, 1431], "viable": True},
            },
            "Footwear": {
                "Heels": {"count": 0, "price_range": None, "viable": False},
            },
        }
    }


def test_taxonomy_prompt_includes_price_ranges():
    from app.services.interpreter import format_taxonomy

    rendered = format_taxonomy(_taxonomy_with_prices())

    assert "429" in rendered and "1955" in rendered, (
        f"price range missing from planner prompt:\n{rendered}"
    )


def test_taxonomy_prompt_still_shows_counts():
    from app.services.interpreter import format_taxonomy

    rendered = format_taxonomy(_taxonomy_with_prices())

    assert "Lehengas" in rendered
    assert "8" in rendered


def test_empty_paths_render_without_a_price_range():
    """A zero-count path has no price range to show, and must not crash or
    invent one -- it still has to appear so the model can report the gap."""
    from app.services.interpreter import format_taxonomy

    rendered = format_taxonomy(_taxonomy_with_prices())

    assert "Heels" in rendered
    heels_line = next(line for line in rendered.splitlines() if "Heels" in line)
    assert "None" not in heels_line


# --- Deterministic guard: drop budget chips the catalogue cannot satisfy ----
#
# The prompt now shows the planner each path's real price range, which
# materially improved the options it generates -- but a prompt cannot
# *guarantee* satisfiability. Two reasons it still slips: the model is asked
# several questions at once and cannot reason about the cross-product (women +
# Rs2,000-5,000 is empty even though each option is individually fine), and
# model output is not deterministic. So the same rule is enforced in code,
# matching the project's split: the model proposes, deterministic code checks.


def _price_taxonomy() -> dict:
    """Sarees and lehengas: nothing above Rs 1,955."""
    return {
        "categories": {
            "Ethnic Wear": {
                "Lehengas": {"count": 8, "price_range": [1245, 1955], "viable": True},
                "Sarees": {"count": 8, "price_range": [429, 1431], "viable": True},
            }
        }
    }


def test_unsatisfiable_budget_option_is_dropped():
    from app.services.interpreter import drop_unsatisfiable_budget_options

    question = {
        "slot": "budget",
        "question": "What is your budget?",
        "options": [
            {"label": "Under Rs 1,500", "value": "price_max:1500"},
            {"label": "Rs 1,500 - 3,000", "value": "price_min:1500,price_max:3000"},
            {"label": "Rs 3,000 and above", "value": "price_min:3000"},
        ],
    }

    kept = drop_unsatisfiable_budget_options(
        [question], ["Ethnic Wear/Lehengas", "Ethnic Wear/Sarees"], _price_taxonomy()
    )

    values = [o["value"] for o in kept[0]["options"]]
    assert "price_min:3000" not in values, "kept a chip no product can satisfy"
    assert "price_max:1500" in values
    assert "price_min:1500,price_max:3000" in values


def test_budget_question_is_dropped_entirely_when_no_option_survives():
    """Better to assume than to ask a question whose every answer is empty."""
    from app.services.interpreter import drop_unsatisfiable_budget_options

    question = {
        "slot": "budget",
        "question": "What is your budget?",
        "options": [
            {"label": "Rs 5,000+", "value": "price_min:5000"},
            {"label": "Rs 10,000+", "value": "price_min:10000"},
        ],
    }

    kept = drop_unsatisfiable_budget_options(
        [question], ["Ethnic Wear/Sarees"], _price_taxonomy()
    )

    assert kept == []


def test_non_budget_questions_are_never_touched():
    from app.services.interpreter import drop_unsatisfiable_budget_options

    question = {
        "slot": "occasion",
        "question": "What is the occasion?",
        "options": [{"label": "Wedding", "value": "occasion:wedding"}],
    }

    kept = drop_unsatisfiable_budget_options([question], ["Ethnic Wear/Sarees"], _price_taxonomy())

    assert kept == [question]


def test_no_paths_or_no_taxonomy_leaves_questions_alone():
    """With nothing to check against, dropping options would be guessing."""
    from app.services.interpreter import drop_unsatisfiable_budget_options

    question = {
        "slot": "budget",
        "question": "Budget?",
        "options": [{"label": "Rs 9,000+", "value": "price_min:9000"}],
    }

    assert drop_unsatisfiable_budget_options([question], [], _price_taxonomy()) == [question]
    assert drop_unsatisfiable_budget_options([question], ["Ethnic Wear/Sarees"], {}) == [question]


# --- Chip values must mean what their prefix claims ------------------------
#
# Observed live: for "trip to goa" the model offered occasion:beach_casual and
# occasion:dining. Neither is in the closed OCCASIONS vocabulary, so neither
# can ever match product.attributes.occasion or earn BOOST_OCCASION. The value
# still widens bucket search phrases, so it is not inert -- but an `occasion:`
# chip that cannot act as an occasion is not doing what the prefix claims, and
# the shopper has no way to see the difference.
#
# Same family as the budget chips that could not be satisfied: the model
# proposes, and deterministic code has to check the proposal against the
# vocabulary the runtime actually understands.


def test_out_of_vocabulary_occasion_is_normalised_not_silently_kept():
    from app.services.interpreter import canonical_answer_value

    assert canonical_answer_value("occasion:beach_casual") == "use_case:beach casual"
    assert canonical_answer_value("occasion:dining") == "use_case:dining"


def test_in_vocabulary_occasion_is_left_alone():
    from app.services.interpreter import canonical_answer_value

    assert canonical_answer_value("occasion:wedding") == "occasion:wedding"
    assert canonical_answer_value("occasion:party") == "occasion:party"


def test_non_occasion_values_pass_through_untouched():
    from app.services.interpreter import canonical_answer_value

    for value in ("gender:men", "price_max:500", "start_date:2026-10-25", "use_case:trekking"):
        assert canonical_answer_value(value) == value


# --- plans that are valid JSON but not a valid plan -----------------------
#
# Found by a scripted situation matrix: each of these reached the shopper as
# HTTP 500 "Internal server error". Free OpenRouter models do not enforce the
# response schema, so parseable-but-wrong output is a live failure mode.

import pytest  # noqa: E402

from app.adapters.llm.base import LLMUnavailable  # noqa: E402


def test_a_wrong_shape_plan_is_a_model_failure_not_a_crash():
    """Raised as LLMUnavailable so recommend() degrades to keyword matching,
    exactly as it already does for unparseable JSON."""
    provider = _FixedProvider({**_BASE_PAYLOAD, "buckets": "not a list"})
    with pytest.raises(LLMUnavailable):
        interpret(provider, "any-model", "a jacket", date(2026, 8, 25))


def test_a_shopping_plan_with_no_buckets_is_still_rejected():
    provider = _FixedProvider({**_BASE_PAYLOAD, "buckets": []})
    with pytest.raises(LLMUnavailable):
        interpret(provider, "any-model", "a jacket", date(2026, 8, 25))


def test_a_declined_request_may_have_no_buckets():
    """The prompt tells the model to set is_shopping_request=false for
    off-topic input; the natural way to do that has no buckets at all."""
    payload = {**_BASE_PAYLOAD, "is_shopping_request": False, "buckets": []}
    structured = interpret(_FixedProvider(payload), "any-model", "capital of France", date(2026, 8, 25))
    assert structured.is_shopping_request is False
    assert structured.buckets == []


def test_buckets_sharing_a_name_are_merged():
    """Two "Shoes" buckets rendered as two identically named groups sharing
    products, and the frontend keys groups by name."""
    running = {**_BASE_PAYLOAD["buckets"][0], "name": "Shoes", "role": "recommended",
               "search_phrases": ["running shoes"], "catalogue_paths": ["Footwear/Sports Shoes"]}
    hiking = {**_BASE_PAYLOAD["buckets"][0], "name": " shoes ", "role": "required",
              "search_phrases": ["hiking boots", "running shoes"], "catalogue_paths": ["Footwear/Boots"]}
    structured = interpret(
        _FixedProvider({**_BASE_PAYLOAD, "buckets": [running, hiking]}),
        "any-model", "shoes", date(2026, 8, 25),
    )
    assert len(structured.buckets) == 1
    merged = structured.buckets[0]
    assert merged.name == "Shoes"
    assert merged.search_phrases == ["running shoes", "hiking boots"]
    assert merged.catalogue_paths == ["Footwear/Sports Shoes", "Footwear/Boots"]
    assert merged.role == "required"


# Live, with the model path: "a trek near Leh ... thermals and warm socks,
# budget 3000" intermittently came back "Nothing in the catalogue fits",
# although 34 men's thermals and dozens of socks were in budget. Shelf paths
# were compared character for character, and a curly apostrophe, lower case,
# "and" for "&" or spaces around "/" each gated out every product.

PATH_VARIANTS = [
    "Men’s Apparel/Thermals & Base Layers",
    "men's apparel/thermals & base layers",
    "Men's Apparel/Thermals and Base Layers",
    "Men's Apparel / Thermals & Base Layers",
]


@pytest.mark.parametrize("variant", PATH_VARIANTS)
def test_model_shelf_paths_are_mapped_to_the_real_catalogue_path(variant):
    bucket = {**_BASE_PAYLOAD["buckets"][0], "catalogue_paths": [variant]}
    payload = {**_BASE_PAYLOAD, "buckets": [bucket],
               "filters": {**_BASE_PAYLOAD["filters"], "categories": [variant, "men’s apparel"]}}
    structured = interpret(_FixedProvider(payload), "any-model", "thermals", date(2026, 9, 29))
    assert structured.buckets[0].catalogue_paths == ["Men's Apparel/Thermals & Base Layers"]
    assert structured.filters.categories == ["Men's Apparel/Thermals & Base Layers", "Men's Apparel"]


def test_an_unknown_path_is_left_for_retrieval_to_report():
    bucket = {**_BASE_PAYLOAD["buckets"][0], "catalogue_paths": ["Snowshoes/Carbon"]}
    structured = interpret(_FixedProvider({**_BASE_PAYLOAD, "buckets": [bucket]}), "m", "x", date(2026, 9, 29))
    assert structured.buckets[0].catalogue_paths == ["Snowshoes/Carbon"]


def test_a_shelf_written_the_way_the_prompt_displays_it_still_maps():
    """Live, qwen3:8b copied the taxonomy display instead of the path:
    "Gifting > Hampers (₹1299–₹5299)" and "Footwear: Sandals & Floaters"
    matched no shelf, so those needs found nothing."""
    from app.services.interpreter import _canonical_path

    assert _canonical_path("Gifting > Hampers (₹1299–₹5299)") == "Gifting/Hampers"
    assert _canonical_path("Footwear: Sandals & Floaters") == "Footwear/Sandals & Floaters"
    assert _canonical_path("Men's Apparel/Jackets & Coats") == "Men's Apparel/Jackets & Coats"
