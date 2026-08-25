"""Structured interpretation of a shopping request.

This is the output contract of LLM call 1 (the interpreter). It is the bridge
between free text and retrieval: everything downstream reads this object, never
the raw user string. Keeping that boundary strict means retrieval stays testable
without an LLM, and the LLM can be swapped or stubbed without touching search.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.product import Formality, WaterResistance

ClimateSource = Literal[
    "measured",         # forecast, within the 16-day horizon
    "climatological",   # downscaled climate model, beyond the forecast horizon
    "user",             # the shopper told us directly
    "inferred",         # the language model's guess -- flagged, never silent
    "unobtainable",     # we could not find out, and say so
]

ContextVariableStatus = Literal["known", "needed", "unobtainable"]
ContextVariableSource = Literal["user", "external", "inferred"]


class ContextVariable(BaseModel):
    """One planning variable and whether we actually have it.

    Symmetric to product buckets: the interpreter may think the request is
    specific enough, but if budget is still `needed` the answer would change
    materially once it is known.
    """

    name: str = Field(description="Machine key: location, dates, climate, budget, …")
    label: str = Field(description="Short display label.")
    status: ContextVariableStatus
    source: ContextVariableSource | None = None
    value: str | None = Field(default=None, description="Display value when known.")


class ClimateContext(BaseModel):
    """Measured conditions for the trip, with the provenance attached.

    This replaces ranking on a sentence the language model wrote. That sentence
    was load-bearing -- `retrieval.temperature_fit()` string-matched it to
    decide which jacket to recommend -- and it was wrong by 5-12C for the one
    trip it was tested against, recommending a -5C jacket for -14.7C nights.

    So the numbers here come from Open-Meteo and the ranking reads the numbers.
    `summary` is rendered *from* them for display; it is never the source of
    truth, and it is never what anything ranks on.

    `source` is part of the contract rather than an internal detail, because a
    measured forecast and a model's guess must never look alike to the person
    deciding what to pack. When nothing can be established, the honest value is
    `unobtainable` -- there is no code path that fills these numbers in from
    imagination.
    """

    source: ClimateSource
    summary: str = Field(
        default="",
        description="Rendered from the numbers for display. Derived, not authoritative.",
    )

    unobtainable_reason: Literal["no dates", "lookup failed"] | None = Field(
        default=None,
        description="Why conditions are unavailable, when source is 'unobtainable'. "
        "A request that named no date never had a lookup attempted, which is a "
        "different thing to tell the shopper than one that was attempted and "
        "failed -- and only the second is a fault.",
    )

    place_resolved: str | None = Field(
        default=None,
        description="What we actually looked up, which may differ from what the "
        "shopper typed. Shown so a wrong match is visible rather than silent.",
    )
    latitude: float | None = None
    longitude: float | None = None
    elevation_m: float | None = Field(
        default=None,
        description="Measured ground elevation at the coordinates. At altitude this "
        "drives everything, and it is also what corroborates a coordinate the "
        "language model proposed for a place no geocoder holds.",
    )

    # Reduced over the trip window: the coldest night and the warmest day, since
    # those are what kit has to cover.
    temp_min_c: float | None = None
    temp_max_c: float | None = None
    precipitation_mm: float | None = Field(
        default=None, description="Total across the window."
    )

    window_start: date | None = None
    window_end: date | None = None
    as_of: date | None = Field(
        default=None, description="When this was fetched. Forecasts go stale."
    )

    @property
    def has_numbers(self) -> bool:
        """Whether anything downstream may rank on this."""
        return self.temp_min_c is not None or self.temp_max_c is not None

    @property
    def precipitation_mm_per_day(self) -> float | None:
        """Rainfall as a rate, which is the only form that means anything.

        `precipitation_mm` is a total over the trip window, so the same number
        describes drizzle across a fortnight and a downpour in an afternoon.
        Comparing that total against a fixed threshold made wetness depend on
        how long the trip was: Mumbai in monsoon reported 4.3 mm because the
        request named no dates and the window collapsed to a single day, which
        then failed a 20 mm threshold calibrated for a week.
        """
        if self.precipitation_mm is None:
            return None
        days = 1
        if self.window_start is not None and self.window_end is not None:
            days = max((self.window_end - self.window_start).days + 1, 1)
        return self.precipitation_mm / days


class ResolvedContext(BaseModel):
    """Real-world facts inferred from the request.

    The interpreter is given today's date, so relative phrases ("last week of
    October", "next month") land on absolute dates. `climate_note` is where the
    model records the inference that keyword search can never make -- that a
    Hampta Pass trek in late October means sub-zero nights.
    """

    location: str | None = None
    # Proposed by the interpreter for places no geocoder holds (mountain passes,
    # trails). Checked against measured elevation before any weather is fetched.
    location_lat: float | None = None
    location_lon: float | None = None
    elevation_estimate_m: int | None = None
    start_date: date | None = None
    end_date: date | None = None
    duration_days: int | None = None
    climate_note: str | None = Field(
        default=None,
        description="Display sentence for the conditions. Rendered from ClimateContext "
        "when real data was obtained; only on the `inferred` path is this the model's "
        "own words, and there it is flagged as such.",
    )
    recipient: str | None = Field(
        default=None, description="Who the purchase is for, if not the user."
    )
    climate: ClimateContext | None = Field(
        default=None,
        description="Resolved conditions with provenance. None when the request "
        "implies no location, e.g. 'suggest me some t-shirts'.",
    )


class Bucket(BaseModel):
    """One shopping need derived from the request.

    Buckets are the unit of retrieval *and* the unit of display: search runs once
    per bucket, so results stay diverse. A single global search for a trek query
    returns ten jackets; per-bucket search returns jackets, boots, and a headlamp.

    `catalogue_paths` is what stops a bucket being filled by something plausible
    but wrong. The planner picks paths from the catalogue's actual taxonomy, so
    a "formal trousers" need either resolves to a real path or resolves to
    nothing -- and nothing is reported as a gap rather than quietly filled with
    the nearest embedding neighbour, which is how women's jeans once answered a
    request for wedding suit trousers.
    """

    name: str = Field(description="Display heading, e.g. 'Layering & Insulation'.")
    search_phrases: list[str] = Field(
        default_factory=list,
        description="Catalogue-language queries for this bucket. These carry the "
        "expansion from intent to product vocabulary.",
    )
    why_needed: str = Field(description="One line tying this bucket to the request.")

    role: Literal["required", "recommended", "optional"] = Field(
        default="recommended",
        description="required = the request is not satisfied without it (trousers "
        "for a suit); recommended = expected but not essential; optional = a nice "
        "addition. Drives whether an unfillable slot is reported as a failure.",
    )
    catalogue_paths: list[str] = Field(
        default_factory=list,
        description="Canonical 'Category/Subcategory' paths that may fill this slot, "
        "chosen from the catalogue taxonomy supplied in the prompt. EMPTY means the "
        "catalogue has no product type for this need -- a deliberate, reportable gap, "
        "never a reason to substitute something unrelated.",
    )

    priority: int = Field(default=2, description="1 = essential, 3 = nice to have.")
    max_items: int = Field(default=4)

    # Clamped rather than rejected. These fields are produced by a language
    # model, and a slightly out-of-range hint ("show 10 items" instead of 8) is
    # not a reason to fail an entire request -- it is a reason to use 8. Strict
    # bounds here turn a cosmetic overshoot into a 500.
    @field_validator("priority")
    @classmethod
    def _clamp_priority(cls, value: int) -> int:
        return max(1, min(3, value))

    @field_validator("max_items")
    @classmethod
    def _clamp_max_items(cls, value: int) -> int:
        return max(1, min(8, value))


class QueryFilters(BaseModel):
    """Hard constraints applied before semantic ranking."""

    price_min: int | None = None
    price_max: int | None = None
    gender: str | None = None
    categories: list[str] = Field(default_factory=list)


class QuestionOption(BaseModel):
    """One tappable answer to a clarifying question."""

    label: str = Field(description="Short chip text, e.g. 'Under Rs.500' or 'Everyday wear'.")
    value: str = Field(
        description="Machine value merged back into the query, e.g. 'price_max:500' "
        "or 'occasion:daily-wear'."
    )


class ClarifyingQuestion(BaseModel):
    """A targeted question asked before recommending.

    Options are model-generated but the *answers* are structured selections, so
    merging them back is a dictionary update rather than another inference. That
    is what keeps the clarification round-trip at zero additional LLM calls --
    the whole reason questions are cheap to add.
    """

    slot: str = Field(
        description="Which gap this fills: occasion, budget, gender, fit, colour, "
        "material, season, size, recipient."
    )
    question: str = Field(description="Asked in second person, one line.")
    options: list[QuestionOption] = Field(
        description="Mutually exclusive choices unless allow_multiple is set.",
    )
    allow_multiple: bool = False


class StructuredQuery(BaseModel):
    intent_summary: str = Field(description="One sentence restating the need, for UI echo.")
    buckets: list[Bucket] = Field(min_length=1)
    filters: QueryFilters = Field(default_factory=QueryFilters)
    context: ResolvedContext = Field(default_factory=ResolvedContext)

    assumptions: list[str] = Field(
        default_factory=list,
        description="Gaps the model filled in. Surfaced in the UI so the user can correct them.",
    )

    needs_clarification: bool = Field(
        default=False,
        description="True when a missing detail would materially change which products "
        "are right. Deliberately adaptive: a request that already states the trip, the "
        "dates and the activity should go straight to results, while 'suggest me some "
        "t-shirts' should not.",
    )
    questions: list[ClarifyingQuestion] = Field(
        default_factory=list,
        description="Asked only when needs_clarification is true. Keep to the gaps that "
        "actually change the answer -- never ask for detail already given.",
    )
    confidence: float = Field(
        default=0.8,
        description="How well the request pins down a recommendation. Drives "
        "needs_clarification.",
    )

    @field_validator("confidence")
    @classmethod
    def _clamp_confidence(cls, value: float) -> float:
        return max(0.0, min(1.0, value))

    @field_validator("questions")
    @classmethod
    def _cap_questions(cls, value: list) -> list:
        # Four is the point past which a "quick question" stops feeling quick.
        return value[:4]
    is_shopping_request: bool = Field(
        default=True,
        description="False for off-topic input, so the API can decline without inventing results.",
    )


class ContextConstraints(BaseModel):
    """Explicit requirement set derived from resolved context, independent of
    any one bucket or product type. Built once per request by
    `services/constraints.py::derive_constraints()`; checked against each
    candidate by `services/suitability.py::evaluate()`.

    This is the generalized mechanism scope.md calls for: thermal mismatch
    already has its own signed scoring in `retrieval.temperature_fit()` and
    is deliberately left alone there (it is well-tested and correct); this
    model covers the axes that had no mechanism at all -- rain suitability
    and formality -- rather than folding everything into one undifferentiated
    penalty.
    """

    min_water_resistance: WaterResistance | None = Field(
        default=None,
        description="'waterproof' for heavy/sustained rain, 'repellent' for light "
        "or possible rain. None means the trip gives no rain signal at all.",
    )
    required_formality: Formality | None = Field(
        default=None,
        description="Set only when the request has a genuine formal-occasion "
        "signal (wedding, office, interview). None otherwise -- a casual or "
        "outdoor request does not get a 'casual' floor, because there is "
        "nothing wrong with a shopper owning something nicer than they asked for.",
    )
    reasons: list[str] = Field(
        default_factory=list,
        description="Why each constraint was set, for traceability -- e.g. "
        "'monsoon: 18mm/day expected' or \"occasion implies formal wear\".",
    )
