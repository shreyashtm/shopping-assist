"""Context variable audit — what we know, what we still need.

The interpreter's `needs_clarification` flag judges whether a request is
*textually thin*. That misses cases where the prose is specific but variables
that would change the answer are still missing — budget and gender on a trek,
dates on a pass with no timing, and so on.

These slots are symmetric to product buckets: each names a variable, tags where
its value came from, and marks it known, needed, or unobtainable. Clarifying
questions are generated deterministically for `needed` slots so the decision
does not depend on the model happening to ask.
"""

from datetime import date, timedelta

from app.schemas.query import (
    ClarifyingQuestion,
    ContextVariable,
    QuestionOption,
    StructuredQuery,
)
from app.services.context import needs_place_climate

BUDGET_QUESTION = ClarifyingQuestion(
    slot="budget",
    question="Roughly what budget?",
    options=[
        QuestionOption(label="Under ₹500", value="price_max:500"),
        QuestionOption(label="₹500 – 1,500", value="price_min:500,price_max:1500"),
        QuestionOption(label="₹1,500 – 3,000", value="price_min:1500,price_max:3000"),
        QuestionOption(label="Premium (₹3,000+)", value="price_min:3000"),
    ],
)

GENDER_QUESTION = ClarifyingQuestion(
    slot="gender",
    question="Who is this for?",
    options=[
        QuestionOption(label="Men", value="gender:men"),
        QuestionOption(label="Women", value="gender:women"),
        QuestionOption(label="Anyone / unisex", value="gender:unisex"),
    ],
)

OCCASION_QUESTION = ClarifyingQuestion(
    slot="occasion",
    question="What's the occasion?",
    options=[
        QuestionOption(label="Everyday wear", value="occasion:daily-wear"),
        QuestionOption(label="Office / formal", value="occasion:office"),
        QuestionOption(label="Sports / active", value="use_case:sports"),
        QuestionOption(label="Trekking / outdoors", value="use_case:trekking"),
    ],
)

def _dates_question(today: date) -> ClarifyingQuestion:
    """When the trip is, as a resolvable date range rather than a mood.

    The previous options ("Within 2 weeks", "Next 1-3 months") could not be
    turned into a date for the Open-Meteo lookup -- `merge_answers` could
    only record them as a prose assumption, so answering this question never
    actually unlocked measured weather. Each option here carries a concrete
    `start_date`/`duration_days` pair anchored to today, so an answer resolves
    climate on the same turn instead of costing another round trip.
    """
    def _at(days_out: int) -> str:
        return f"start_date:{(today + timedelta(days=days_out)).isoformat()},duration_days:7"

    return ClarifyingQuestion(
        slot="dates",
        question="When is the trip? Season and weather drive what you need.",
        options=[
            QuestionOption(label="This week", value=_at(0)),
            QuestionOption(label="In 2–4 weeks", value=_at(21)),
            QuestionOption(label="1–3 months out", value=_at(60)),
            QuestionOption(label="Not sure yet", value="timing:unknown"),
        ],
    )

_TREK_HINTS = ("trek", "hike", "hiking", "mountain", "pass", "expedition", "camp", "altitude")
_APPAREL_HINTS = (
    "shirt", "t-shirt", "tshirt", "top", "dress", "jeans", "trouser", "jacket", "wear",
)
_GIFT_HINTS = ("gift", "present", "anniversary", "hamper")


def _answered_keys(answers: list[str]) -> set[str]:
    keys: set[str] = set()
    for answer in answers:
        for pair in answer.split(","):
            key, _, _ = pair.partition(":")
            if key.strip():
                keys.add(key.strip())
    return keys


def _text_blob(structured: StructuredQuery) -> str:
    parts = [structured.intent_summary.lower()]
    for bucket in structured.buckets:
        parts.extend([bucket.name.lower(), bucket.why_needed.lower()])
        parts.extend(p.lower() for p in bucket.search_phrases)
    ctx = structured.context
    if ctx.location:
        parts.append(ctx.location.lower())
    if ctx.climate_note:
        parts.append(ctx.climate_note.lower())
    return " ".join(parts)


def _implies_trek(text: str) -> bool:
    return any(token in text for token in _TREK_HINTS)


def _implies_apparel(text: str) -> bool:
    return any(token in text for token in _APPAREL_HINTS)


def _implies_gift(text: str) -> bool:
    return any(token in text for token in _GIFT_HINTS)


def _has_budget(structured: StructuredQuery, answered: set[str]) -> bool:
    filters = structured.filters
    return bool(
        filters.price_min is not None
        or filters.price_max is not None
        or "price_min" in answered
        or "price_max" in answered
        or "budget" in answered
    )


def _has_gender(structured: StructuredQuery, answered: set[str]) -> bool:
    return bool(structured.filters.gender or "gender" in answered)


def _has_occasion_signal(structured: StructuredQuery, answered: set[str], text: str) -> bool:
    if "occasion" in answered or "use_case" in answered:
        return True
    if _implies_trek(text) or _implies_gift(text):
        return True
    if any("wedding" in p.lower() for b in structured.buckets for p in b.search_phrases):
        return True
    return False


def _is_specific_trip(structured: StructuredQuery) -> bool:
    """Location + dates + outdoor activity — enough to recommend without budget chips."""
    ctx = structured.context
    text = _text_blob(structured)
    return bool(ctx.location and ctx.start_date and _implies_trek(text))


def build_context_variables(
    structured: StructuredQuery, answers: list[str]
) -> list[ContextVariable]:
    """Tag each planning variable with status and provenance."""
    ctx = structured.context
    answered = _answered_keys(answers)
    text = _text_blob(structured)
    slots: list[ContextVariable] = []

    if ctx.location:
        slots.append(
            ContextVariable(
                name="location",
                label="Place",
                status="known",
                source="inferred",
                value=ctx.location,
            )
        )
    elif _implies_trek(text):
        slots.append(ContextVariable(name="location", label="Place", status="needed"))

    if ctx.start_date:
        date_label = ctx.start_date.isoformat()
        if ctx.end_date and ctx.end_date != ctx.start_date:
            date_label = f"{ctx.start_date.isoformat()} – {ctx.end_date.isoformat()}"
        slots.append(
            ContextVariable(
                name="dates",
                label="Dates",
                status="known",
                source="user" if ctx.duration_days else "inferred",
                value=date_label,
            )
        )
    elif ctx.location and _implies_trek(text):
        slots.append(ContextVariable(name="dates", label="Dates", status="needed"))

    climate = ctx.climate
    if climate is not None:
        if climate.has_numbers:
            source = "external" if climate.source in ("measured", "climatological") else "user"
            slots.append(
                ContextVariable(
                    name="climate",
                    label="Conditions",
                    status="known",
                    source=source,
                    value=climate.summary or ctx.climate_note,
                )
            )
        elif climate.source == "unobtainable":
            slots.append(
                ContextVariable(
                    name="climate",
                    label="Conditions",
                    status="unobtainable",
                    value=climate.summary or "Could not verify",
                )
            )
    elif needs_place_climate(ctx):
        slots.append(ContextVariable(name="climate", label="Conditions", status="needed"))

    if _has_budget(structured, answered):
        value = _budget_label(structured)
        slots.append(
            ContextVariable(
                name="budget",
                label="Budget",
                status="known",
                source="user",
                value=value,
            )
        )
    elif not _is_specific_trip(structured):
        slots.append(ContextVariable(name="budget", label="Budget", status="needed"))

    if _has_gender(structured, answered):
        slots.append(
            ContextVariable(
                name="gender",
                label="For",
                status="known",
                source="user",
                value=structured.filters.gender or "unspecified",
            )
        )
    elif _implies_apparel(text) and not _is_specific_trip(structured):
        slots.append(ContextVariable(name="gender", label="For", status="needed"))

    if _has_occasion_signal(structured, answered, text):
        slots.append(
            ContextVariable(
                name="occasion",
                label="Occasion",
                status="known",
                source="inferred" if "occasion" not in answered else "user",
                value="from your request",
            )
        )
    elif _implies_apparel(text) and not _is_specific_trip(structured):
        slots.append(ContextVariable(name="occasion", label="Occasion", status="needed"))

    if ctx.recipient:
        slots.append(
            ContextVariable(
                name="recipient",
                label="Recipient",
                status="known",
                source="user",
                value=ctx.recipient,
            )
        )
    elif _implies_gift(text):
        slots.append(ContextVariable(name="recipient", label="Recipient", status="needed"))

    return slots


def _budget_label(structured: StructuredQuery) -> str:
    filters = structured.filters
    if filters.price_min and filters.price_max:
        return f"₹{filters.price_min:,} – ₹{filters.price_max:,}"
    if filters.price_max:
        return f"Under ₹{filters.price_max:,}"
    if filters.price_min:
        return f"₹{filters.price_min:,}+"
    return "Set"


def _question_for_slot(name: str, today: date) -> ClarifyingQuestion | None:
    return {
        "budget": BUDGET_QUESTION,
        "gender": GENDER_QUESTION,
        "occasion": OCCASION_QUESTION,
        "dates": _dates_question(today),
    }.get(name)


def _question_keys(question: ClarifyingQuestion) -> set[str]:
    """Return canonical request keys represented by a model question."""
    keys: set[str] = set()
    for option in question.options:
        for value in option.value.split(","):
            key, _, _ = value.partition(":")
            if key.strip():
                keys.add(key.strip())
    slot = question.slot.strip().lower().replace("-", "_")
    if slot in {"budget", "budget_tier", "budget_range"}:
        keys.update({"budget", "price_min", "price_max"})
    elif slot in {"gender", "audience"}:
        keys.add("gender")
    elif slot in {"occasion", "occasion_type", "event", "event_type"}:
        keys.add("occasion")
    elif slot in {"dates", "timing", "when"}:
        keys.update({"timing", "start_date", "duration_days"})
    return keys


def is_specific_trip(structured: StructuredQuery) -> bool:
    """Location + dates + outdoor activity — enough to recommend without more chips."""
    return _is_specific_trip(structured)


def apply_context_audit(
    structured: StructuredQuery, answers: list[str], today: date | None = None
) -> tuple[StructuredQuery, list[ContextVariable]]:
    """Fill context slots and append deterministic questions for gaps."""
    today = today or date.today()
    slots = build_context_variables(structured, answers)
    answered = _answered_keys(answers)
    text = _text_blob(structured)

    def is_redundant(question: ClarifyingQuestion) -> bool:
        slot = question.slot.strip().lower().replace("-", "_")
        if slot in {"occasion", "occasion_type", "event", "event_type"}:
            return _has_occasion_signal(structured, answered, text)
        if slot in {"budget", "budget_tier", "budget_range"}:
            return _has_budget(structured, answered)
        if slot in {"gender", "audience"}:
            return _has_gender(structured, answered)
        if slot in {"recipient", "who_for"}:
            return structured.context.recipient is not None
        if slot in {"dates", "timing", "when"}:
            return structured.context.start_date is not None
        # The model is free to label a question anything -- "style", "fit",
        # "colour" -- and it is not required to reuse the same label between
        # turns. Falling through to False here is what let an unrecognised
        # slot name loop forever: the same gap gets re-asked every turn
        # because nothing above ever matches it, even after it is answered.
        # The general rule is label-independent: a question is redundant once
        # every machine-value key its own options would set is already
        # answered, regardless of what its author decided to call it.
        keys = _question_keys(question)
        return bool(keys) and keys.issubset(answered)

    # Keep only questions whose machine values are understood by the runtime.
    # Showing a model-generated control such as ``formality:formal`` would be
    # misleading because no schema field or ranking rule consumes it.
    supported_keys = {
        "budget", "price_min", "price_max", "gender", "occasion", "use_case",
        "timing", "category", "start_date", "duration_days",
    }
    questions = [
        q for q in structured.questions
        if _question_keys(q) & supported_keys and not is_redundant(q)
    ]

    # The model may name the budget slot more specifically, such as
    # ``budget_tier``. Treat a question whose options set price constraints as
    # covering the canonical budget slot, otherwise the audit appends a second
    # budget question to the same response.
    existing = {q.slot for q in questions}
    for question in questions:
        existing.update(_question_keys(question) & supported_keys)
    if any(
        q.slot == "budget"
        or q.slot in {"budget_tier", "budget_range"}
        or any(
            option.value.startswith(("price_min:", "price_max:", "budget:"))
            for option in q.options
        )
        for q in questions
    ):
        existing.add("budget")
    extra: list[ClarifyingQuestion] = []

    for slot in slots:
        if slot.status != "needed":
            continue
        if slot.name in answered or slot.name in existing:
            continue
        question = _question_for_slot(slot.name, today)
        if question is not None:
            extra.append(question)

    merged = questions + [q for q in extra if q.slot not in existing]
    needs = bool(merged)
    if needs:
        structured = structured.model_copy(
            update={
                "needs_clarification": True,
                "questions": merged[:4],
            }
        )
    elif structured.needs_clarification or structured.questions:
        structured = structured.model_copy(
            update={"needs_clarification": False, "questions": []}
        )

    return structured, slots
