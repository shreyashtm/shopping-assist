"""State-machine regression tests for the ask-then-recommend flow.

These pin the conversational contract end to end, against the real catalogue,
with a scripted fake LLM standing in for the live model so the flow is
deterministic. They exist because of a real regression: turns could loop on
follow-up questions -- sometimes repeating one, sometimes asking a fresh one
each time -- and never reach the recommendation stage, or reach it with an
empty screen because a filter silently matched nothing. See
`test_retrieval.py::test_category_filter_also_accepts_a_full_taxonomy_path`
and `test_context_slots.py::test_non_canonical_slot_question_is_not_reasked_once_answered`
for the two root causes these guard from the other direction.

The example flow below (gender, then budget, then a style/occasion question,
then a recommendation) is one instance of the contract, not the contract
itself: a system that always asks exactly those three questions in that order
would also fail these tests if it ever asked a fourth, or reached the cap
without producing products, or re-asked something already answered.
"""

from datetime import date

import pytest

from app.core.cache import response_cache
from app.schemas.recommend import RecommendRequest
from app.services.catalogue import Catalogue
from app.services.recommend import MAX_CLARIFY_ANSWERS, recommend

TODAY = date(2026, 8, 25)


@pytest.fixture(autouse=True)
def clear_cache():
    # `response_cache` is a process-global singleton (see test_robustness.py),
    # and both tests below query the same catalogue text with overlapping
    # answer prefixes -- without clearing it, one test's cached response can
    # silently serve a later test's turn.
    response_cache.clear()
    yield
    response_cache.clear()


_TSHIRT_PLAN = {
    "intent_summary": "A t-shirt.",
    "is_shopping_request": True,
    "confidence": 0.6,
    "buckets": [
        {
            "name": "T-Shirts",
            "search_phrases": ["cotton t-shirt", "everyday t-shirt"],
            "why_needed": "What you asked for.",
            "role": "required",
            "catalogue_paths": ["Men's Apparel/T-Shirts"],
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


def _question(slot: str, question: str, options: list[tuple[str, str]]) -> dict:
    return {
        "slot": slot,
        "question": question,
        "options": [{"label": label, "value": value} for label, value in options],
        "allow_multiple": False,
    }


class ScriptedProvider:
    """Returns one canned StructuredQuery payload per call, in order.

    Mirrors `DeadProvider` in test_robustness.py, but returns real (if
    scripted) structured output instead of raising. `calls` records every
    prompt so a test can assert what the interpreter was actually asked.
    """

    name = "scripted"
    is_real = True

    def __init__(self, responses: list[dict]):
        self._responses = list(responses)
        self.calls: list[str] = []

    def structured(self, *, system, user, schema, model, max_tokens, timeout_s, effort):
        self.calls.append(user)
        if not self._responses:
            raise AssertionError("ScriptedProvider ran out of scripted turns")
        return self._responses.pop(0)


@pytest.fixture(scope="module")
def catalogue():
    return Catalogue.load()


def _plan(needs_clarification: bool, questions: list[dict]) -> dict:
    return {**_TSHIRT_PLAN, "needs_clarification": needs_clarification, "questions": questions}


def _answer_for(question) -> str:
    """Pick an option that won't accidentally hit a real, unrelated catalogue
    gap. This catalogue's Men's Apparel/T-Shirts starts at Rs.613, so the
    budget question's own first option ("Under Rs.500") is a genuine empty
    result for this category -- correct behaviour from the system, but not
    what these tests are about. Preferring the option with the loosest price
    ceiling keeps the flow assertions about the state machine, not the
    catalogue's real price distribution.
    """
    if question.slot == "budget":
        return max(
            question.options,
            key=lambda o: "price_max" not in o.value,
        ).value
    return question.options[0].value


def test_gender_budget_style_then_recommendation(catalogue):
    """One instance of the contract: three answered rounds, then products.

    Only the first turn's questions are asserted against a script -- the
    interpreter re-runs from scratch every turn (see
    `interpreter.interpret`, which discards the model's own follow-up
    questions once any answer exists), so turns 2+ are driven by the
    deterministic audit in `context_slots.py`, which is exactly the part
    under test.
    """
    provider = ScriptedProvider([
        _plan(True, [_question(
            "gender", "Who is this for?",
            [("Men", "gender:men"), ("Women", "gender:women")],
        )]),
        _plan(False, []),
        _plan(False, []),
        _plan(False, []),
    ])

    seen_slots: list[set[str]] = []
    answers: list[str] = []
    response = None

    for _ in range(MAX_CLARIFY_ANSWERS + 1):
        response = recommend(
            RecommendRequest(query="I need a t-shirt", answers=list(answers)),
            catalogue, provider, today=TODAY,
        )
        slots_this_turn = {q.slot for q in response.questions}

        # The regression: a slot already answered in an earlier turn must
        # never come back.
        answered_keys = {a.split(":", 1)[0] for a in answers}
        assert not (slots_this_turn & answered_keys), (
            f"already-answered slot re-asked: {slots_this_turn & answered_keys}"
        )
        seen_slots.append(slots_this_turn)

        if response.mode == "results":
            break

        assert response.groups, (
            "a clarify turn must still carry a recommendation when the "
            "catalogue can offer one, not just a question"
        )

        # Answer whatever was asked and continue.
        for q in response.questions:
            answers.append(_answer_for(q))

    assert response is not None
    assert response.mode == "results", "must reach the recommendation stage, not loop forever"
    assert response.groups, "the recommendation stage must produce actual products"
    for group in response.groups:
        assert group.items, "a shown group must not be empty"
        for item in group.items:
            assert item.product.id
            assert item.reason


def test_provider_that_never_stops_asking_still_reaches_results(catalogue):
    """The explicit stop condition: even a model that always wants to ask
    one more thing must not be allowed to loop forever. Every scripted turn
    proposes a *different* supported-key question, so this cannot pass by
    accident of the redundancy check -- only the round cap
    (`MAX_CLARIFY_ANSWERS` in `services/recommend.py`) can terminate it."""
    endless_questions = [
        _question("gender", "Who's it for?", [("Men", "gender:men")]),
        _question("occasion", "What's the occasion?", [("Daily", "occasion:daily-wear")]),
        _question("use", "What will you use it for?", [("Casual", "use_case:daily-wear")]),
        _question("budget", "What's your budget?", [("Any", "price_min:0")]),
        _question("timing", "When do you need it?", [("Soon", "timing:soon")]),
        _question("category", "Any category preference?", [("Shirts", "category:Men's Apparel")]),
    ]
    provider = ScriptedProvider(
        [_plan(True, [q]) for q in endless_questions] + [_plan(True, [endless_questions[0]])] * 4
    )

    answers: list[str] = []
    response = None
    for turn in range(MAX_CLARIFY_ANSWERS + 2):
        response = recommend(
            RecommendRequest(query="I need a t-shirt", answers=list(answers)),
            catalogue, provider, today=TODAY,
        )
        if response.mode == "results":
            break
        for q in response.questions:
            answers.append(_answer_for(q))
        assert turn < MAX_CLARIFY_ANSWERS + 1, "must not still be asking past the round cap"

    assert response is not None
    assert response.mode == "results", (
        "an endlessly-asking model must still be forced to a recommendation"
    )
    assert response.groups, "forcing results must not mean forcing an empty screen"
