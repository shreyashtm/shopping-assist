"""Unit tests for the provisional-questions guesser.

Pure text heuristics -- no LLM, no network, no catalogue. These pin the
deliberately conservative behavior: guess only what the real context audit
already treats as high-confidence signal, and stay silent otherwise.
"""

from datetime import date

from app.services.context_slots import guess_preview_questions

TODAY = date(2026, 9, 28)


def test_trek_request_guesses_dates_and_gender():
    slots = {q.slot for q in guess_preview_questions(
        "I'm trekking Hampta Pass the last week of October for a week", TODAY
    )}
    assert slots == {"dates", "gender"}


def test_dated_trip_without_gear_words_still_guesses_dates():
    slots = {q.slot for q in guess_preview_questions("planning a trip to Goa", TODAY)}
    assert "dates" in slots


def test_apparel_request_guesses_gender_but_not_dates():
    questions = guess_preview_questions("find me a good pair of formal trousers", TODAY)
    slots = {q.slot for q in questions}
    assert slots == {"gender"}


def test_gift_request_guesses_nothing():
    """The conservative design point: gifting has no high-confidence text
    signal for dates or gender, so it must not guess wrong just to guess."""
    questions = guess_preview_questions(
        "I need a premium gifting hamper for my parents' anniversary", TODAY
    )
    assert questions == []


def test_never_guesses_budget_or_occasion():
    """Budget and occasion are explicitly out of scope for the guesser --
    narrower signals than this heuristic can support with confidence."""
    questions = guess_preview_questions(
        "trekking gear for a wedding gift trip to Goa", TODAY
    )
    slots = {q.slot for q in questions}
    assert "budget" not in slots
    assert "occasion" not in slots


def test_dates_question_resolves_to_a_concrete_range():
    questions = guess_preview_questions("hiking trip next month", TODAY)
    dates_q = next(q for q in questions if q.slot == "dates")
    assert dates_q.options[0].value.startswith("start_date:")
