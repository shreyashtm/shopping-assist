"""Turn resolved context into an explicit requirement set.

This is the first half of the generalized suitability mechanism scope.md
calls for. `derive_constraints()` reads the same evidence retrieval already
has -- measured climate, the request's own wording -- and produces one
`ContextConstraints` per request, checked against each candidate by
`services/suitability.py::evaluate()`.

Deliberately conservative: a constraint is only set when there is a real
signal for it. A request with no rain evidence gets `min_water_resistance =
None`, not a default, because "no opinion" and "requires no rain protection"
are different claims -- the first lets a product's water_resistance be
irrelevant to its score, the second would need a whole separate constraint
this module does not model.
"""

from app.schemas.query import ContextConstraints, StructuredQuery

# Precipitation thresholds, in mm/day -- deliberately separate from
# retrieval.WET_MM_PER_DAY (which only asks "is this monsoon-ish, for season
# tagging") rather than importing it, because this module asks a stricter
# question: is rain protection actually load-bearing for this trip. The
# higher bar keeps an ordinary passing-shower forecast from demanding
# waterproof gear.
LIGHT_RAIN_MM_PER_DAY = 3.0
HEAVY_RAIN_MM_PER_DAY = 15.0

_RAIN_NOTE_HINTS = ("monsoon", "rain", "rainy", "wet weather")

_FORMAL_HINTS = (
    "wedding", "formal", "office", "interview", "black-tie", "black tie",
    "sherwani", "blazer", "suit", "gala",
)


def _implies_formal(structured: StructuredQuery) -> bool:
    parts = [structured.intent_summary.lower()]
    for bucket in structured.buckets:
        parts.extend([bucket.name.lower(), bucket.why_needed.lower()])
        parts.extend(p.lower() for p in bucket.search_phrases)
    text = " ".join(parts)
    return any(hint in text for hint in _FORMAL_HINTS)


def derive_constraints(structured: StructuredQuery) -> ContextConstraints:
    """Build the requirement set for one request. Called once per turn."""
    reasons: list[str] = []
    min_water_resistance = None

    climate = structured.context.climate
    if climate is not None and climate.has_numbers:
        rate = climate.precipitation_mm_per_day
        if rate is not None and rate >= HEAVY_RAIN_MM_PER_DAY:
            min_water_resistance = "waterproof"
            reasons.append(f"heavy rain expected ({rate:.0f}mm/day) -- rain protection matters")
        elif rate is not None and rate >= LIGHT_RAIN_MM_PER_DAY:
            min_water_resistance = "repellent"
            reasons.append(f"rain expected ({rate:.0f}mm/day)")
    elif structured.context.climate_note:
        # No measured numbers: fall back to reading the sentence, coarsely --
        # same fallback pattern as retrieval.score_product for season.
        lowered = structured.context.climate_note.lower()
        if any(hint in lowered for hint in _RAIN_NOTE_HINTS):
            min_water_resistance = "repellent"
            reasons.append("rain mentioned in trip conditions")

    required_formality = None
    if _implies_formal(structured):
        required_formality = "formal"
        reasons.append("occasion implies formal wear")

    return ContextConstraints(
        min_water_resistance=min_water_resistance,
        required_formality=required_formality,
        reasons=reasons,
    )
