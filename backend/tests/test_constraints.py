"""Constraint derivation tests.

`derive_constraints()` must stay conservative: a constraint is only set when
there is a real signal, because "no opinion" and "requires none" are
different claims -- see the module docstring.
"""

from app.schemas.query import Bucket, ClimateContext, ResolvedContext, StructuredQuery
from app.services.constraints import derive_constraints


def _query(context: ResolvedContext, bucket_name: str = "Layering", why_needed: str = "") -> StructuredQuery:
    return StructuredQuery(
        intent_summary="A trip",
        buckets=[
            Bucket(
                name=bucket_name,
                search_phrases=["jacket"],
                why_needed=why_needed,
                catalogue_paths=["Men's Apparel/Jackets & Coats"],
            )
        ],
        context=context,
    )


def test_no_climate_signal_leaves_water_resistance_unset():
    constraints = derive_constraints(_query(ResolvedContext()))
    assert constraints.min_water_resistance is None


def test_heavy_measured_rain_requires_waterproof():
    climate = ClimateContext(source="measured", temp_min_c=20, temp_max_c=28, precipitation_mm=126, window_start=None, window_end=None)
    constraints = derive_constraints(_query(ResolvedContext(climate=climate)))
    assert constraints.min_water_resistance == "waterproof"
    assert constraints.reasons


def test_light_measured_rain_requires_only_repellent():
    climate = ClimateContext(source="measured", temp_min_c=20, temp_max_c=28, precipitation_mm=8, window_start=None, window_end=None)
    constraints = derive_constraints(_query(ResolvedContext(climate=climate)))
    assert constraints.min_water_resistance == "repellent"


def test_dry_measured_climate_sets_no_water_resistance_requirement():
    climate = ClimateContext(source="measured", temp_min_c=15, temp_max_c=25, precipitation_mm=1, window_start=None, window_end=None)
    constraints = derive_constraints(_query(ResolvedContext(climate=climate)))
    assert constraints.min_water_resistance is None


def test_unmeasured_climate_note_mentioning_monsoon_implies_repellent():
    constraints = derive_constraints(
        _query(ResolvedContext(climate_note="Monsoon season in Mumbai."))
    )
    assert constraints.min_water_resistance == "repellent"


def test_wedding_request_implies_formal():
    constraints = derive_constraints(
        _query(ResolvedContext(), bucket_name="Traditional Wear", why_needed="For a friend's wedding.")
    )
    assert constraints.required_formality == "formal"


def test_trek_request_does_not_imply_a_formality_floor():
    """A casual/outdoor request gets no formality constraint at all -- there is
    nothing wrong with a shopper owning something nicer than they asked for,
    so there is no 'casual' floor to violate."""
    constraints = derive_constraints(
        _query(ResolvedContext(), bucket_name="Trekking Essentials", why_needed="Cold nights at altitude.")
    )
    assert constraints.required_formality is None
