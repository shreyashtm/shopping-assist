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


# --- ranked geocoding --------------------------------------------------------
#
# Live checks on 2026-09-28: with no model coordinates, Open-Meteo's unranked
# settlement list sent "Leh" to Le Havre and "Goa" to Genoa. Nominatim ranks by
# prominence and covers states and regions, so its first hit is trusted for a
# bare name; Open-Meteo is only ever used to match a proposed point.

from app.adapters.weather.nominatim import NominatimClient  # noqa: E402
from app.services import context as context_module  # noqa: E402


class _GazetteerTransport(httpx.BaseTransport):
    """Nominatim returns `ranked`; Open-Meteo geocoding returns `unranked`."""

    def __init__(self, ranked=None, nominatim_down=False, unranked=None):
        self.ranked = ranked or []
        self.nominatim_down = nominatim_down
        self.unranked = unranked or []

    def handle_request(self, request):
        url = str(request.url)
        if "nominatim" in url:
            if self.nominatim_down:
                return httpx.Response(503)
            return httpx.Response(200, json=self.ranked)
        if "geocoding" in url:
            return httpx.Response(200, json={"results": self.unranked})
        if "elevation" in url:
            return httpx.Response(200, json={"elevation": [3500.0]})
        return _MockTransport().handle_request(request)


def _nominatim(name, lat, lon, state, country):
    return {"name": name, "lat": str(lat), "lon": str(lon),
            "address": {"state": state, "country": country}}


def _resolve(place, transport, lat=None, lon=None):
    context_module._place_cache.clear()
    context_module._miss_cache.clear()
    ctx = ResolvedContext(location=place, start_date=date(2026, 12, 20), end_date=date(2026, 12, 27))
    client = OpenMeteoClient(transport=transport)
    places = NominatimClient(transport=transport)
    try:
        return resolve_climate(ctx, client, date(2026, 9, 28), proposed_lat=lat,
                               proposed_lon=lon, places=places)
    finally:
        client.close(); places.close()


@pytest.fixture(autouse=False)
def no_throttle(monkeypatch):
    monkeypatch.setattr("app.adapters.weather.nominatim.MIN_INTERVAL_S", 0.0)


def test_name_only_uses_the_ranked_gazetteer_first_hit(no_throttle):
    transport = _GazetteerTransport(
        ranked=[_nominatim("Leh", 34.0, 77.66, "Ladakh", "India")],
        unranked=[{"name": "Le Havre", "latitude": 49.49, "longitude": 0.11, "country": "France"}],
    )
    climate = _resolve("Leh", transport)
    assert climate.place_resolved == "Leh, Ladakh, India"


def test_name_only_foreign_destination_resolves_abroad(no_throttle):
    """No country preference: a shopper flying to Dubai means the UAE."""
    transport = _GazetteerTransport(ranked=[_nominatim("Dubai", 25.07, 55.19, "Dubai", "United Arab Emirates")])
    assert _resolve("Dubai", transport).place_resolved.endswith("United Arab Emirates")


def test_name_only_never_guesses_from_the_unranked_list(no_throttle):
    """Nominatim down: Open-Meteo's first hit is exactly the Le Havre failure."""
    transport = _GazetteerTransport(
        nominatim_down=True,
        unranked=[{"name": "Le Havre", "latitude": 49.49, "longitude": 0.11, "country": "France"}],
    )
    climate = _resolve("Leh", transport)
    assert climate.source == "unobtainable"


def test_model_coordinates_pick_the_matching_candidate(no_throttle):
    """A bare "Auli" ranks a Ukrainian village first; the model's point near
    Joshimath selects the Uttarakhand one from the same list."""
    transport = _GazetteerTransport(ranked=[
        _nominatim("Auly", 48.54, 34.45, "Dnipropetrovsk Oblast", "Ukraine"),
        _nominatim("Auli", 60.03, 11.35, "Akershus", "Norway"),
        _nominatim("Auli", 30.54, 79.57, "Uttarakhand", "India"),
    ])
    climate = _resolve("Auli", transport, lat=30.53, lon=79.56)
    assert climate.place_resolved == "Auli, Uttarakhand, India"


def test_nominatim_adapter_parses_and_tolerates_bad_rows(no_throttle):
    transport = _GazetteerTransport(ranked=[
        {"name": "Goa", "lat": "15.3", "lon": "74.08", "address": {"state": "Goa", "country": "India"}},
        {"name": "broken", "lat": "not-a-number", "lon": "1"},
    ])
    client = NominatimClient(transport=transport)
    try:
        places = client.search("Goa")
    finally:
        client.close()
    assert [(p.name, p.country, p.latitude) for p in places] == [("Goa", "India", 15.3)]


def test_place_display_does_not_repeat_a_region_named_after_its_state():
    from app.adapters.weather.open_meteo import Place
    assert Place("Goa", 15.3, 74.1, None, "India", "Goa", None).display == "Goa, India"
    assert Place("Manali", 32.2, 77.2, None, "India", "Himachal Pradesh", None).display == "Manali, Himachal Pradesh, India"


# --- areas are not points -----------------------------------------------------
#
# Live on 2026-09-28, after switching to Nominatim: "Europe" resolved to one
# point (-1.4..7.8C), "India" to 4.8..23.5C, "Himachal Pradesh" to a single
# reading although Shimla and Spiti differ by 20C. Nominatim reports each hit's
# bounding box; an area wider than a climate reading can represent needs a
# point from the model, or it is unobtainable.


def _area(name, lat, lon, state, country, south, north, west, east):
    row = _nominatim(name, lat, lon, state, country)
    row["boundingbox"] = [str(south), str(north), str(west), str(east)]
    return row


LADAKH = _area("Ladakh", 33.95, 77.66, "Ladakh", "India", 32.3, 35.6, 75.3, 79.4)


def test_a_broad_area_without_a_model_point_is_unobtainable(no_throttle):
    assert _resolve("Ladakh", _GazetteerTransport(ranked=[LADAKH])).source == "unobtainable"


def test_a_broad_area_uses_the_model_point_inside_it(no_throttle):
    """Model says Ladakh at Leh's coordinates: measure Leh, not the centroid."""
    climate = _resolve("Ladakh", _GazetteerTransport(ranked=[LADAKH]), lat=34.16, lon=77.58)
    assert climate.source != "unobtainable"
    assert (climate.latitude, climate.longitude) == (34.16, 77.58)
    assert climate.place_resolved.startswith("Ladakh")


def test_a_model_point_outside_the_area_is_not_trusted(no_throttle):
    """Near enough to select Ladakh (189 km from its centre) but south of its
    bounding box: the point contradicts the area it claims to be in."""
    climate = _resolve("Ladakh", _GazetteerTransport(ranked=[LADAKH]), lat=32.25, lon=77.66)
    assert climate.source == "unobtainable"


def test_a_small_area_is_still_a_point(no_throttle):
    goa = _area("Goa", 15.3, 74.08, "Goa", "India", 14.9, 15.8, 73.7, 74.3)
    assert _resolve("Goa", _GazetteerTransport(ranked=[goa])).place_resolved == "Goa, India"


class _CountingTransport(_GazetteerTransport):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.nominatim_calls = 0

    def handle_request(self, request):
        if "nominatim" in str(request.url):
            self.nominatim_calls += 1
        return super().handle_request(request)


def test_a_name_the_gazetteer_does_not_know_is_not_looked_up_again(no_throttle):
    transport = _CountingTransport(ranked=[])
    context_module._miss_cache.clear()
    assert _resolve("Qwxzplorbia", transport).source == "unobtainable"
    ctx = ResolvedContext(location="Qwxzplorbia", start_date=date(2026, 12, 20), end_date=date(2026, 12, 27))
    client, places = OpenMeteoClient(transport=transport), NominatimClient(transport=transport)
    try:
        resolve_climate(ctx, client, date(2026, 9, 28), places=places)
    finally:
        client.close(); places.close()
    assert transport.nominatim_calls == 1


def test_a_failed_lookup_is_not_cached_as_a_miss(no_throttle):
    """Nominatim down and Open-Meteo empty: nothing was learned, so ask again."""
    transport = _CountingTransport(nominatim_down=True)
    _resolve("Leh", transport)  # clears both caches first
    ctx = ResolvedContext(location="Leh", start_date=date(2026, 12, 20), end_date=date(2026, 12, 27))
    client, places = OpenMeteoClient(transport=transport), NominatimClient(transport=transport)
    try:
        resolve_climate(ctx, client, date(2026, 9, 28), places=places)
    finally:
        client.close(); places.close()
    assert transport.nominatim_calls == 2


def test_a_district_ranked_first_resolves_to_its_namesake_town(no_throttle):
    """Live: Nominatim ranks Leh district (3.3 degrees) above Leh town."""
    district = _area("Leh", 34.0, 77.66, "Ladakh", "India", 32.3, 35.6, 75.3, 79.4)
    town = _area("Leh", 34.16, 77.58, "Ladakh", "India", 34.0, 34.32, 77.4, 77.72)
    swiss = _area("Leh", 47.47, 9.26, "St. Gallen", "Switzerland", 47.45, 47.49, 9.24, 9.28)
    climate = _resolve("Leh", _GazetteerTransport(ranked=[district, town, swiss]))
    assert (climate.latitude, climate.longitude) == (34.16, 77.58)
