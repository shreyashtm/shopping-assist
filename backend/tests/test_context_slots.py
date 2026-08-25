"""Context slot audit tests."""

from datetime import date

from app.schemas.query import (
    Bucket,
    ClimateContext,
    ClarifyingQuestion,
    QueryFilters,
    ResolvedContext,
    StructuredQuery,
)
from app.services.context_slots import apply_context_audit, build_context_variables
from app.services.interpreter import merge_answers


def _trek_query(**ctx_overrides) -> StructuredQuery:
    ctx_fields = {
        "location": "Hampta Pass",
        "start_date": date(2026, 10, 25),
        "end_date": date(2026, 11, 1),
        "duration_days": 7,
        "climate": ClimateContext(
            source="climatological",
            temp_min_c=-15,
            temp_max_c=-2,
            summary="Typical conditions: nights to -15 C.",
        ),
    }
    ctx_fields.update(ctx_overrides)
    ctx = ResolvedContext(**ctx_fields)
    return StructuredQuery(
        intent_summary="Week-long trek to Hampta Pass in late October",
        buckets=[
            Bucket(
                name="Layering",
                search_phrases=["insulated jacket for trekking"],
                why_needed="Cold nights at altitude.",
                catalogue_paths=["Men's Apparel/Jackets & Coats"],
            )
        ],
        context=ctx,
    )


def test_specific_trek_does_not_need_budget_or_gender():
    slots = build_context_variables(_trek_query(), [])
    names = {s.name: s.status for s in slots}
    assert names.get("dates") == "known"
    assert names.get("climate") == "known"
    assert "budget" not in names or names["budget"] != "needed"
    assert "gender" not in names or names["gender"] != "needed"


def test_trek_without_dates_marks_dates_needed():
    q = _trek_query(start_date=None, end_date=None, duration_days=None)
    slots = build_context_variables(q, [])
    assert any(s.name == "dates" and s.status == "needed" for s in slots)


def test_tshirt_request_marks_budget_and_occasion_needed():
    structured = StructuredQuery(
        intent_summary="Suggest me some t-shirts",
        buckets=[
            Bucket(
                name="T-Shirts",
                search_phrases=["cotton t-shirt"],
                why_needed="Everyday tops.",
                catalogue_paths=["Men's Apparel/T-Shirts"],
            )
        ],
    )
    slots = build_context_variables(structured, [])
    needed = {s.name for s in slots if s.status == "needed"}
    assert "budget" in needed
    assert "occasion" in needed


def test_audit_adds_budget_question_for_thin_request():
    structured = StructuredQuery(
        intent_summary="Suggest me some t-shirts",
        buckets=[
            Bucket(
                name="T-Shirts",
                search_phrases=["cotton t-shirt"],
                why_needed="Everyday tops.",
                catalogue_paths=["Men's Apparel/T-Shirts"],
            )
        ],
        needs_clarification=False,
        questions=[],
    )
    updated, slots = apply_context_audit(structured, [])
    assert updated.needs_clarification
    assert any(q.slot == "budget" for q in updated.questions)
    assert slots


def test_answered_budget_removes_needed_slot():
    structured = StructuredQuery(
        intent_summary="Suggest me some t-shirts",
        buckets=[
            Bucket(
                name="T-Shirts",
                search_phrases=["cotton t-shirt"],
                why_needed="Everyday tops.",
                catalogue_paths=["Men's Apparel/T-Shirts"],
            )
        ],
        filters=QueryFilters(price_max=1500),
    )
    slots = build_context_variables(structured, ["price_max:1500"])
    assert any(s.name == "budget" and s.status == "known" for s in slots)


def test_known_wedding_drops_redundant_model_occasion_question():
    structured = StructuredQuery(
        intent_summary="Traditional wear for a friend's wedding in March",
        buckets=[
            Bucket(
                name="Traditional Wear",
                search_phrases=["wedding clothing"],
                why_needed="Formal clothing for a wedding.",
                catalogue_paths=["Ethnic Wear/Kurta Sets"],
            )
        ],
        needs_clarification=True,
        questions=[
            ClarifyingQuestion(
                slot="occasion_type",
                question="What is the occasion?",
                options=[],
            )
        ],
    )

    updated, _ = apply_context_audit(structured, [])

    assert updated.needs_clarification
    assert all(q.slot not in {"occasion", "occasion_type"} for q in updated.questions)


def test_combined_gender_question_does_not_create_duplicate_gender_question():
    structured = StructuredQuery(
        intent_summary="Traditional wear for a friend's wedding",
        buckets=[
            Bucket(
                name="Traditional Wear",
                search_phrases=["wedding traditional wear"],
                why_needed="Formal clothing for a wedding.",
                catalogue_paths=["Ethnic Wear/Kurta Sets"],
            )
        ],
        needs_clarification=True,
        questions=[
            ClarifyingQuestion(
                slot="gender_recipient",
                question="Is this for yourself or your friend? And what gender?",
                options=[
                    {"label": "Men", "value": "gender:men,recipient:friend"},
                    {"label": "Women", "value": "gender:women,recipient:friend"},
                ],
            ),
            ClarifyingQuestion(
                slot="formality_level",
                question="How formal should the look be?",
                options=[{"label": "Formal", "value": "formality:formal"}],
            ),
        ],
    )

    updated, _ = apply_context_audit(structured, [])

    slots = [q.slot for q in updated.questions]
    assert slots.count("gender") == 0
    assert "gender_recipient" in slots
    assert "formality_level" not in slots


def test_dates_question_offers_resolvable_dates():
    """The old options ("Within 2 weeks", "Next 1-3 months") were moods, not
    dates -- merge_answers could only turn them into a prose assumption, so
    answering never actually unlocked weather. Every option except the
    explicit "not sure" one must carry a real start_date."""
    from datetime import date as date_cls

    from app.services.context_slots import _dates_question

    question = _dates_question(date_cls(2026, 9, 1))
    resolvable = [o for o in question.options if o.value != "timing:unknown"]
    assert resolvable
    for option in resolvable:
        assert "start_date:2026-" in option.value
        assert "duration_days:" in option.value


def test_answering_dates_question_resolves_real_dates():
    """merge_answers must set actual context.start_date/end_date from the
    dates question's answer, not just record an assumption string -- that's
    the whole point of the fix (see interpreter.merge_answers)."""
    structured = StructuredQuery(
        intent_summary="Trekking gear for a pass",
        buckets=[
            Bucket(
                name="Layering",
                search_phrases=["insulated jacket"],
                why_needed="Cold at altitude.",
                catalogue_paths=["Men's Apparel/Jackets & Coats"],
            )
        ],
    )
    merge_answers(structured, ["start_date:2026-09-15,duration_days:7"])
    assert structured.context.start_date == date(2026, 9, 15)
    assert structured.context.duration_days == 7
    assert structured.context.end_date == date(2026, 9, 21)


def test_dates_slot_is_known_once_resolved_and_not_reasked():
    structured = StructuredQuery(
        intent_summary="Trekking gear for Hampta Pass",
        buckets=[
            Bucket(
                name="Layering",
                search_phrases=["insulated jacket for trekking"],
                why_needed="Cold nights at altitude.",
                catalogue_paths=["Men's Apparel/Jackets & Coats"],
            )
        ],
        context=ResolvedContext(location="Hampta Pass"),
        needs_clarification=True,
        questions=[],
    )
    merge_answers(structured, ["start_date:2026-09-15,duration_days:7"])
    updated, slots = apply_context_audit(
        structured, ["start_date:2026-09-15,duration_days:7"], date(2026, 8, 25)
    )
    assert all(q.slot != "dates" for q in updated.questions)
    assert any(s.name == "dates" and s.status == "known" for s in slots)


def test_gender_answer_alias_is_normalized_to_product_enum():
    structured = StructuredQuery(
        intent_summary="Wedding wear",
        buckets=[
            Bucket(
                name="Traditional Wear",
                search_phrases=["wedding sherwani"],
                why_needed="Wedding clothing.",
                catalogue_paths=["Ethnic Wear/Sherwanis"],
            )
        ],
    )

    merge_answers(structured, ["gender:male"])

    assert structured.filters.gender == "men"


def test_non_canonical_slot_question_is_not_reasked_once_answered():
    """A question the model labels with a slot name outside the five
    canonical ones (e.g. "style") must still be dropped once its answer has
    been supplied -- otherwise it is re-asked on every turn forever and the
    conversation never reaches the recommendation stage. Reproduces the
    reported regression: gender/budget answered, then a "style" question
    loops instead of yielding a recommendation."""
    structured = StructuredQuery(
        intent_summary="Something for a casual weekend",
        buckets=[
            Bucket(
                name="Tops",
                search_phrases=["casual t-shirt"],
                why_needed="Everyday tops.",
                catalogue_paths=["Men's Apparel/T-Shirts"],
            )
        ],
        filters=QueryFilters(price_max=1500, gender="men"),
        needs_clarification=True,
        questions=[
            ClarifyingQuestion(
                slot="style",
                question="What style are you going for?",
                options=[
                    {"label": "Casual", "value": "use_case:daily-wear"},
                    {"label": "Sporty", "value": "use_case:sports"},
                ],
            )
        ],
    )

    updated, _ = apply_context_audit(
        structured,
        ["price_max:1500", "gender:men", "use_case:daily-wear"],
    )

    assert all(q.slot != "style" for q in updated.questions), (
        f"style question re-asked after being answered: {updated.questions}"
    )
    assert not updated.needs_clarification, (
        "all needed slots are known, so the recommendation stage should be reached"
    )
