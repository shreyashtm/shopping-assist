"""Context acquisition tests.

Pins the judgements that make measured weather safe to rank on: elevation
corroboration rejects wrong coordinates, Manali disambiguation prefers the
proposal over population, and unobtainable never fabricates numbers.
"""

from datetime import date

import httpx
import pytest

from app.adapters.weather.open_meteo import OpenMeteoClient, Place
from app.schemas.query import ResolvedContext
from app.services.context import (
    apply_climate_from_answers,
    build_climate_question,
    elevation_agrees,
    has_climate_answers,
    pick_place,
    render_summary,
    resolve_climate,
)


def test_elevation_agrees_within_tolerance():
    assert elevation_agrees(4393, 4200)
    assert not elevation_agrees(765, 4200)


def test_pick_place_prefers_nearest_to_proposal():
    """Manali, Tamil Nadu has higher population but is wrong for a Himalayan trek."""
    candidates = [
        Place("Manali", 13.17, 80.27, 6.0, "India", "Tamil Nadu", 35000),
        Place("Manali", 32.26, 77.17, 2108.0, "India", "Himachal Pradesh", 8000),
    ]
    chosen = pick_place(candidates, 32.24, 77.37)
    assert chosen is not None
    assert chosen.admin1 == "Himachal Pradesh"


# Without model coordinates the first geocoder hit used to win outright. Both
# candidate lists below are what Open-Meteo actually returned on 2026-09-28.

LEH = [
    Place("Le Havre", 49.49, 0.11, 5.0, "France", "Normandy", 185972),
    Place("Leh", 34.16, 77.58, 3500.0, "India", "Ladakh", 37475),
    Place("Leh", 47.2, 11.0, None, "Austria", None, None),
]
GOA = [
    Place("Genoa", 44.41, 8.93, 20.0, "Italy", "Liguria", 580097),
    Place("Goa", 13.55, 123.28, None, "Philippines", None, 20936),
    Place("Goa", 55.0, 38.0, None, "Russia", None, None),
]


def test_without_coordinates_an_indian_exact_match_beats_the_first_hit():
    """"Leh" resolved to Le Havre, France: a December Leh trek ranked for
    3.7-10.9C instead of nights near -15C."""
    chosen = pick_place(LEH, None, None, "Leh")
    assert chosen is not None and chosen.country == "India" and chosen.name == "Leh"


def test_without_coordinates_no_indian_candidate_is_unobtainable_not_a_guess():
    """"Goa" is a state, so the geocoder has no Indian entry at all and
    offered Genoa. Better to report conditions as unobtainable."""
    assert pick_place(GOA, None, None, "Goa") is None


def test_without_coordinates_an_indian_non_exact_name_is_still_accepted():
    """"Ooty" geocodes only to its official name, Udhagamandalam."""
    ooty = [Place("Udhagamandalam", 11.41, 76.7, 2240.0, "India", "Tamil Nadu", 233426)]
    assert pick_place(ooty, None, None, "Ooty") is ooty[0]


def test_with_coordinates_foreign_places_are_still_allowed():
    """The India preference applies only when there is no proposal to match
    against; a model-supplied point still picks the nearest candidate."""
    chosen = pick_place(LEH, 49.5, 0.1)
    assert chosen is not None and chosen.country == "France"


def test_render_summary_includes_provenance():
    from app.schemas.query import ClimateContext

    climate = ClimateContext(
        source="climatological",
        place_resolved="Hampta Pass",
        elevation_m=4393,
        temp_min_c=-14.8,
        temp_max_c=-2.1,
    )
    text = render_summary(climate)
    assert "Typical" in text
    assert "-15" in text or "-14" in text


def test_has_climate_answers_detects_chips():
    assert has_climate_answers(["temp_min:-10,temp_max:5"])
    assert not has_climate_answers(["gender:men"])


def test_apply_climate_from_answers_sets_user_source():
    ctx = ResolvedContext(location="Hampta Pass")
    updated = apply_climate_from_answers(ctx, ["temp_min:-10,temp_max:5"])
    assert updated.climate is not None
    assert updated.climate.source == "user"
    assert updated.climate.temp_min_c == -10


def test_build_climate_question_has_options():
    q = build_climate_question(ResolvedContext(location="Hampta Pass"))
    assert q.slot == "climate"
    assert len(q.options) >= 2


class _MockTransport(httpx.BaseTransport):
    """Stub Open-Meteo responses for Hampta-like coordinates."""

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if "geocoding" in url:
            return httpx.Response(200, json={"results": []})
        if "elevation" in url:
            lat = float(request.url.params["latitude"])
            elev = 4393.0 if lat > 30 else 765.0
            return httpx.Response(200, json={"elevation": [elev]})
        if "climate-api" in url:
            return httpx.Response(
                200,
                json={
                    "elevation": 4393.0,
                    "daily": {
                        "time": ["2026-10-25", "2026-10-26"],
                        "temperature_2m_min": [-14.8, -15.0],
                        "temperature_2m_max": [-2.2, -1.0],
                        "precipitation_sum": [0.0, 0.0],
                    },
                },
            )
        return httpx.Response(404, json={"error": True})


@pytest.fixture
def mock_client():
    return OpenMeteoClient(transport=_MockTransport())


def test_resolve_climate_accepts_corroborated_proposal(mock_client):
    ctx = ResolvedContext(
        location="Hampta Pass",
        start_date=date(2026, 10, 25),
        end_date=date(2026, 11, 1),
        duration_days=7,
    )
    climate = resolve_climate(
        ctx,
        mock_client,
        date(2026, 8, 24),
        proposed_lat=32.24,
        proposed_lon=77.37,
        proposed_elevation_m=4270,
    )
    assert climate is not None
    assert climate.source == "climatological"
    assert climate.temp_min_c == pytest.approx(-15.0)
    assert climate.has_numbers


def test_resolve_climate_rejects_bad_proposal(mock_client):
    ctx = ResolvedContext(
        location="Hampta Pass",
        start_date=date(2026, 10, 25),
        end_date=date(2026, 11, 1),
    )
    climate = resolve_climate(
        ctx,
        mock_client,
        date(2026, 8, 24),
        proposed_lat=13.65,
        proposed_lon=75.83,
        proposed_elevation_m=4200,
    )
    assert climate is not None
    assert climate.source == "unobtainable"


def test_resolve_climate_none_without_place():
    client = OpenMeteoClient(transport=_MockTransport())
    try:
        assert resolve_climate(ResolvedContext(), client, date.today()) is None
    finally:
        client.close()


# --- "Could not verify" must say *why* -------------------------------------
#
# Real report: "suggest dress for my trip to goa" returned "Conditions for Goa
# could not be verified." Nothing had been attempted -- the request carried no
# date, so trip_window() returned None and resolve_climate() bailed before any
# lookup. The message implied a failed lookup and read as a broken app, when
# the truth was that the same beach is monsoon in August and peak season in
# December and the system was refusing to guess.
#
# The cause was already computed and passed to unobtainable() as `reason`, then
# logged and thrown away. These pin it into the user-visible sentence.


def test_missing_dates_says_so_rather_than_implying_a_failed_lookup():
    from app.services.context import unobtainable

    climate = unobtainable(ResolvedContext(location="Goa"), "no dates in the request")

    assert "Goa" in climate.summary
    assert "could not be verified" not in climate.summary
    assert "date" in climate.summary.lower()


def test_a_genuine_lookup_failure_still_reads_as_a_failure():
    from app.services.context import unobtainable

    climate = unobtainable(
        ResolvedContext(location="Hampta Pass"), "could not establish coordinates"
    )

    assert "Hampta Pass" in climate.summary
    assert "could not" in climate.summary.lower()


def test_unobtainable_never_invents_numbers():
    """Whatever the wording, the contract holds: no fabricated temperatures."""
    from app.services.context import unobtainable

    climate = unobtainable(ResolvedContext(location="Goa"), "no dates in the request")

    assert climate.source == "unobtainable"
    assert climate.temp_min_c is None
    assert climate.temp_max_c is None


# Same-name places in different states, as Open-Meteo returned them on
# 2026-09-28. Without coordinates only a clearly dominant one is trusted.


def _india(name, admin1, population, lat=20.0, lon=78.0):
    return Place(name, lat, lon, None, "India", admin1, population)


def test_without_coordinates_a_cross_state_tie_is_ambiguous():
    """Name-only "Manali" went to Tamil Nadu (20.9-29.2C in December) rather
    than the Himalayan town, which is the smaller of the two."""
    manali = [_india("Manali", "Tamil Nadu", 35248), _india("Manali", "Himachal Pradesh", 8096)]
    assert pick_place(manali, None, None, "Manali") is None
    auli = [_india("Auli", "Himachal Pradesh", None), _india("Auli", "Uttarakhand", None)]
    assert pick_place(auli, None, None, "Auli") is None


def test_without_coordinates_a_dominant_same_name_place_is_trusted():
    shimla = [_india("Shimla", "Himachal Pradesh", 173503), _india("Shimla", "Rajasthan", None)]
    assert pick_place(shimla, None, None, "Shimla").admin1 == "Himachal Pradesh"
    udaipur = [_india("Udaipur", "Rajasthan", 451100), _india("Udaipur", "Tripura", 32758)]
    assert pick_place(udaipur, None, None, "Udaipur").admin1 == "Rajasthan"


def test_without_coordinates_same_state_duplicates_are_not_ambiguous():
    darjeeling = [_india("Darjeeling", "West Bengal", 123797), _india("Darjeeling", "West Bengal", None)]
    assert pick_place(darjeeling, None, None, "Darjeeling").population == 123797
