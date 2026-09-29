"""Context acquisition: turn a place and some dates into measured conditions.

This exists because the previous answer to "how cold will it be?" was a
sentence the language model wrote from memory, and that sentence was ranked on.
For Hampta Pass in late October it claimed nights of -5C to -10C and days of
+5C to +15C. The measured values are -17.5C and +0.5C. The system then offered
a -5C jacket as its top pick, which for a real trip is not a weak suggestion,
it is a dangerous one.

The fix is not a better prompt. It is to stop asking a model for a number that
can be looked up.

## Who supplies what

    where a place is       -> the model may propose it, but only as a proposal
    what elevation it is   -> Open-Meteo, measured
    what the weather is    -> Open-Meteo, measured or modelled
    whether that is cold   -> arithmetic

## Why a proposed coordinate is still acceptable

No free geocoder holds "Hampta Pass". Open-Meteo's is GeoNames-backed and
returns settlements; OpenStreetMap has Rohtang Pass but not Hampta. So for
exactly the kind of place this product is interesting for, geocoding fails and
something has to propose a point.

A language model can do that -- but it must not be believed. So it is checked:
the elevation endpoint reports the real ground height at the proposed point,
and a claimed 4,200m pass that lands at 765m is rejected. Measured against a
local 3B model that put Hampta Pass in Karnataka, that check fires. Against a
36B model 26km off the true saddle, it passes -- and the weather returned for
that point is within ~2.4C of the weather at the true one, which is an order of
magnitude better than the 5-12C error from asking a model to state temperatures
directly.

That is the whole argument: an approximate place with measured weather beats an
exact-sounding sentence with invented weather.

## When nothing can be established

`unobtainable`. There is no branch in this module that invents a number, and
none that quietly downgrades to the model's guess without labelling it.
"""

import logging
import math
from datetime import date, timedelta

from app.adapters.weather.nominatim import NominatimClient
from app.adapters.weather.open_meteo import (
    FORECAST_HORIZON_DAYS,
    DailySeries,
    OpenMeteoClient,
    Place,
    WeatherUnavailable,
)
from app.core.cache import ResponseCache
from app.schemas.query import ClarifyingQuestion, ClimateContext, QuestionOption, ResolvedContext

logger = logging.getLogger(__name__)

# How far a measured elevation may sit from the model's estimate before the
# coordinate is treated as wrong. Generous in absolute terms because estimates
# of mountain passes are rough, and proportional as well so the tolerance means
# something at sea level too.
ELEVATION_TOLERANCE_M = 600.0
ELEVATION_TOLERANCE_FRACTION = 0.25

# A geocoding hit further than this from the proposed point is a different
# place with the same name, not a refinement of it. "Manali" returns Manali in
# Tamil Nadu (6m) ahead of Manali in Himachal Pradesh (2,108m), because the
# Tamil Nadu one has four times the population -- so first-result-wins is wrong
# and population ranking is wrong. Proximity to the proposal settles it.
MAX_GEOCODE_MATCH_KM = 200.0

# Geography does not change and elevation does not change, so these are cached
# for the life of the process. Climate is cached for a day: a forecast moves,
# and a climatology does not move fast enough to matter within one.
_place_cache = ResponseCache(ttl_seconds=60 * 60 * 24 * 30, max_entries=512)
_weather_cache = ResponseCache(ttl_seconds=60 * 60 * 24, max_entries=512)
# Names the gazetteer genuinely answered "nothing usable" for. Kept apart from
# _place_cache and for a day, not a month: a miss is cheaper to be wrong about,
# and Nominatim allows one request a second, so re-asking about the same
# unknown name on every request is what this prevents. A failed lookup is
# never stored here -- that is "we could not ask", not "there is nothing".
_miss_cache = ResponseCache(ttl_seconds=60 * 60 * 24, max_entries=512)

# How wide a gazetteer hit can be and still stand for one climate reading.
# Goa (about 1 degree) is one reading; Himachal Pradesh (3 degrees) spans
# Shimla and Spiti, 20C apart in December, and "Europe" or "India" are not a
# place to measure at all. Past this, only a model-proposed point inside the
# area is measured.
MAX_POINT_EXTENT_DEG = 1.5

# The widest area a model's point may pin down. Within a state (Ladakh is 4
# degrees wide, Rajasthan 8) the point says which part is meant. Live, qwen3:8b
# gave "Europe" a point at 36 m, and the page showed one reading for a whole
# continent; for a country or continent any point is arbitrary.
MAX_AREA_FOR_MODEL_POINT_DEG = 10.0


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    radius = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return 2 * radius * math.asin(math.sqrt(a))


# Popular treks no gazetteer holds: Nominatim and Open-Meteo both return
# nothing for "Hampta Pass", so its weather rested on the model recalling
# coordinates -- right once on Claude Haiku, wrong on qwen3:8b. These points
# still go through the elevation check, so a bad entry can only fall back to
# "unverified", never to wrong weather. (lat, lon, known elevation in metres;
# Open-Meteo measured 4516, 3733, 4660, 3396, 2532, 2919 m at them.)
KNOWN_TREK_POINTS: dict[str, tuple[float, float, float]] = {
    "hampta pass": (32.27, 77.37, 4270),
    "kedarkantha": (31.02, 78.17, 3800),
    "roopkund": (30.26, 79.73, 5029),
    "sandakphu": (27.10, 88.00, 3636),
    "triund": (32.26, 76.35, 2850),
    "kheerganga": (31.99, 77.51, 2960),
}


def known_trek_point(location: str | None) -> tuple[float, float, float] | None:
    if not location:
        return None
    head = location.split(",")[0].strip().lower()
    return next((point for name, point in KNOWN_TREK_POINTS.items() if head.startswith(name)), None)


def elevation_agrees(measured_m: float, estimated_m: float | None) -> bool:
    """Whether a measured elevation corroborates the model's estimate.

    With no estimate to check against there is nothing to disagree with, so
    this passes -- the coordinate is then resting on the geocoder, which is a
    different and better kind of evidence.
    """
    if estimated_m is None:
        return True
    tolerance = max(ELEVATION_TOLERANCE_M, abs(estimated_m) * ELEVATION_TOLERANCE_FRACTION)
    return abs(measured_m - estimated_m) <= tolerance


def pick_place(
    candidates: list[Place],
    proposed_lat: float | None,
    proposed_lon: float | None,
) -> Place | None:
    """Choose among same-named places.

    Nearest to the proposed point when there is one, because that is the only
    signal that actually distinguishes them. Without a proposal the first
    candidate wins, so callers must only pass a list whose order means
    something -- see `_locate`, which never does this with Open-Meteo's
    unranked settlement list.
    """
    if not candidates:
        return None
    if proposed_lat is None or proposed_lon is None:
        return candidates[0]

    nearest = min(
        candidates,
        key=lambda p: haversine_km(proposed_lat, proposed_lon, p.latitude, p.longitude),
    )
    distance = haversine_km(proposed_lat, proposed_lon, nearest.latitude, nearest.longitude)
    return nearest if distance <= MAX_GEOCODE_MATCH_KM else None


def trip_window(context: ResolvedContext, today: date) -> tuple[date, date] | None:
    """The date range to ask about, or None when the request pins down neither.

    A start with no end is treated as a single day rather than guessed at,
    unless a duration was given. Past dates clamp to today: the forecast
    endpoint refuses ranges outside its window, and a shopper buying for a trip
    means the trip ahead of them.
    """
    start = context.start_date
    end = context.end_date

    if start is None and end is None:
        return None
    if start is None:
        start = end
    if end is None:
        end = start + timedelta(days=max((context.duration_days or 1) - 1, 0))

    start = max(start, today)
    end = max(end, start)
    return start, end


def _reduce(series: DailySeries) -> tuple[float | None, float | None, float | None]:
    """Coldest night, warmest day, total precipitation over the window.

    The minimum rather than the mean, because kit has to cover the worst night
    of the trip, not the average one.
    """
    tmin = min(series.temp_min_c) if series.temp_min_c else None
    tmax = max(series.temp_max_c) if series.temp_max_c else None
    precipitation = sum(series.precipitation_mm) if series.precipitation_mm else None
    return tmin, tmax, precipitation


def render_summary(climate: ClimateContext) -> str:
    """A display sentence built from the numbers, never the other way round.

    Provenance is stated in the sentence itself rather than left to a badge, so
    the caveat survives being copied, screenshotted or read aloud.
    """
    place = climate.place_resolved or "this location"
    where = place
    if climate.elevation_m is not None:
        where = f"{place} ({climate.elevation_m:,.0f} m)"

    if climate.source == "unobtainable":
        # The cause matters to the reader. "Could not be verified" implies a
        # lookup was attempted and failed; when the request simply carried no
        # date, nothing was attempted at all, and saying otherwise reads as a
        # broken app rather than as the system declining to guess.
        if climate.unobtainable_reason == "no dates":
            return (
                f"Add a date for {place} and conditions can be checked -- "
                f"the same place needs different clothing in different seasons."
            )
        return f"Conditions for {place} could not be verified."

    if not climate.has_numbers:
        return f"No temperature data available for {place}."

    lead = {
        "measured": f"Forecast for {where}:",
        "climatological": f"Typical conditions for {where} at this time of year:",
        "user": "Conditions you told us to expect:",
        "inferred": f"Estimated conditions for {where}, not verified:",
    }[climate.source]

    parts = []
    if climate.temp_min_c is not None:
        parts.append(f"nights to {climate.temp_min_c:.0f} C")
    if climate.temp_max_c is not None:
        parts.append(f"days to {climate.temp_max_c:.0f} C")
    if climate.precipitation_mm is not None and climate.precipitation_mm >= 1:
        parts.append(f"{climate.precipitation_mm:.0f} mm precipitation expected")
    return f"{lead} {', '.join(parts)}."


def unobtainable(context: ResolvedContext, reason: str) -> ClimateContext:
    """Report that conditions are unavailable, and why.

    `reason` used to be logged and discarded, so every cause produced the same
    sentence. A request with no date and a genuinely failed lookup are very
    different things to tell a shopper about, and only one of them is a fault.
    """
    logger.info("Climate unobtainable for %r: %s", context.location, reason)
    climate = ClimateContext(
        source="unobtainable",
        place_resolved=context.location,
        unobtainable_reason="no dates" if "no dates" in reason else "lookup failed",
    )
    return climate.model_copy(update={"summary": render_summary(climate)})


def resolve_climate(
    context: ResolvedContext,
    client: OpenMeteoClient,
    today: date,
    proposed_lat: float | None = None,
    proposed_lon: float | None = None,
    proposed_elevation_m: float | None = None,
    places: NominatimClient | None = None,
) -> ClimateContext | None:
    """Resolve conditions for a request, or report that we could not.

    Returns None when the request implies no place at all -- "suggest me some
    t-shirts" has no conditions to look up, and inventing a climate strip for
    it would be noise.
    """
    if not context.location and proposed_lat is None:
        return None

    window = trip_window(context, today)
    if window is None:
        # A place with no dates is genuinely unanswerable: the same pass needs
        # different kit in June and October. The caller turns this into a
        # question rather than a guess.
        return unobtainable(context, "no dates in the request")

    start, end = window

    try:
        located = _locate(
            context.location, proposed_lat, proposed_lon, proposed_elevation_m, client, places
        )
    except WeatherUnavailable as exc:
        return unobtainable(context, f"location lookup failed: {exc}")
    if located is None:
        return unobtainable(context, "could not establish coordinates")

    latitude, longitude, elevation_m, place_name = located

    horizon_days = (start - today).days
    use_forecast = horizon_days <= FORECAST_HORIZON_DAYS
    source = "measured" if use_forecast else "climatological"

    cache_key = f"{source}:{latitude:.3f},{longitude:.3f}:{start}:{end}"
    series = _weather_cache.get(cache_key)
    if series is None:
        try:
            series = (
                client.forecast(latitude, longitude, start, end)
                if use_forecast
                else client.climatology(latitude, longitude, start, end)
            )
        except WeatherUnavailable as exc:
            return unobtainable(context, f"weather lookup failed: {exc}")
        _weather_cache.set(cache_key, series)

    tmin, tmax, precipitation = _reduce(series)
    if tmin is None and tmax is None:
        return unobtainable(context, "weather source returned no temperatures")

    climate = ClimateContext(
        source=source,
        place_resolved=place_name,
        latitude=latitude,
        longitude=longitude,
        elevation_m=series.elevation_m if series.elevation_m is not None else elevation_m,
        temp_min_c=tmin,
        temp_max_c=tmax,
        precipitation_mm=precipitation,
        window_start=start,
        window_end=end,
        as_of=today,
    )
    return climate.model_copy(update={"summary": render_summary(climate)})


def _is_area(place: Place) -> bool:
    return (place.extent_deg or 0) > MAX_POINT_EXTENT_DEG


def _contains(area: Place, latitude: float, longitude: float) -> bool:
    south, north, west, east = area.bbox
    return south <= latitude <= north and west <= longitude <= east


def _named_town_inside(area: Place, candidates: list[Place]) -> Place | None:
    """The measurable place an area shares its name with, if it contains one.

    A district is often ranked above the town it is named after -- Nominatim
    gives "Leh" as Leh district (3.3 degrees wide), then Leh town inside it --
    and that town is what the bare name means. Ladakh, Himachal Pradesh or
    India contain no namesake, so they stay unmeasurable without a point.
    """
    return next(
        (
            c for c in candidates
            if c is not area
            and c.name.lower() == area.name.lower()
            and not _is_area(c)
            and _contains(area, c.latitude, c.longitude)
        ),
        None,
    )


def _locate(
    location: str | None,
    proposed_lat: float | None,
    proposed_lon: float | None,
    proposed_elevation_m: float | None,
    client: OpenMeteoClient,
    places: NominatimClient | None = None,
) -> tuple[float, float, float | None, str] | None:
    """Settle on coordinates, or return None.

    Geocoding is preferred when it produces a hit that the proposal agrees
    with, because a real gazetteer entry is stronger evidence than a model's
    recall. The proposal is the fallback, and only survives if the measured
    elevation at that point corroborates it.

    Candidates come from Nominatim, which ranks by prominence and knows states
    and regions, so with no proposal its first result is a sound reading of a
    bare name. Open-Meteo's settlement list is the fallback when Nominatim is
    unreachable, but it is unranked -- its first hit sent "Leh" to Le Havre --
    so it is only ever used to match a proposed point, never to guess one.
    """
    key = f"{location}|{proposed_lat}|{proposed_lon}"
    cached = _place_cache.get(key)
    if cached is not None:
        return cached
    if _miss_cache.get(key):
        return None

    candidates: list[Place] = []
    ranked = False
    lookup_failed = False  # "could not ask" -- never cached as a miss
    if location and places is not None:
        try:
            candidates = places.search(location)
            ranked = True
        except WeatherUnavailable as exc:
            lookup_failed = True
            logger.info("Nominatim lookup of %r failed, trying Open-Meteo: %s", location, exc)
    if location and not ranked:
        try:
            candidates = client.geocode(location)
        except WeatherUnavailable as exc:
            lookup_failed = True
            logger.info("Geocoding %r failed, falling back to proposal: %s", location, exc)

    def miss() -> None:
        if not lookup_failed:
            _miss_cache.set(key, True)

    has_proposal = proposed_lat is not None and proposed_lon is not None
    # A name whose best reading is a continent or country has no one climate,
    # whatever point the model offers. Checked before matching the proposal:
    # live, the model's point for "Europe" was far from Europe's centre, so
    # the match below rejected the gazetteer entry and the bare point was used.
    if ranked and candidates and (candidates[0].extent_deg or 0) > MAX_AREA_FOR_MODEL_POINT_DEG:
        logger.info("%r is %.0f deg wide, too broad for one point", location, candidates[0].extent_deg)
        miss()
        return None
    chosen = pick_place(candidates, proposed_lat, proposed_lon) if (ranked or has_proposal) else None

    if chosen is not None and _is_area(chosen):
        # An area, not a place: its centroid's weather is nobody's trip.
        if has_proposal:
            # A model point inside the area, whose elevation checks out, says
            # which part is meant -- unless the area is too wide for any one
            # point to stand for it.
            if (chosen.extent_deg or 0) > MAX_AREA_FOR_MODEL_POINT_DEG:
                logger.info("%r is %.0f deg wide, too broad for one point", location, chosen.extent_deg)
                miss()
                return None
            if not _contains(chosen, proposed_lat, proposed_lon):
                miss()
                return None
            measured = client.elevation(proposed_lat, proposed_lon)
            if not elevation_agrees(measured, proposed_elevation_m):
                miss()
                return None
            result = (proposed_lat, proposed_lon, measured, chosen.display)
            _place_cache.set(key, result)
            return result
        town = _named_town_inside(chosen, candidates)
        if town is None:
            logger.info("%r is an area %.1f deg wide with no point in it", location, chosen.extent_deg)
            miss()
            return None
        chosen = town

    if chosen is not None:
        elevation = chosen.elevation_m
        if elevation is None:
            try:
                elevation = client.elevation(chosen.latitude, chosen.longitude)
            except WeatherUnavailable:
                elevation = None
        result = (chosen.latitude, chosen.longitude, elevation, chosen.display)
        _place_cache.set(key, result)
        return result

    if proposed_lat is None or proposed_lon is None:
        miss()
        return None

    # No gazetteer entry. The proposal is all we have, so check it against the
    # ground before trusting it.
    measured = client.elevation(proposed_lat, proposed_lon)
    if not elevation_agrees(measured, proposed_elevation_m):
        logger.info(
            "Rejecting proposed coordinates for %r: estimated %sm, measured %.0fm",
            location, proposed_elevation_m, measured,
        )
        miss()
        return None

    result = (proposed_lat, proposed_lon, measured, location or "the given coordinates")
    _place_cache.set(key, result)
    return result


def climate_from_answers(
    context: ResolvedContext, temp_min_c: float | None, temp_max_c: float | None
) -> ClimateContext:
    """Conditions the shopper supplied themselves.

    Used when lookup failed and we asked. Their answer is real information, so
    it ranks like any other -- but it is sourced as `user`, not laundered into
    looking like a measurement.
    """
    climate = ClimateContext(
        source="user",
        place_resolved=context.location,
        temp_min_c=temp_min_c,
        temp_max_c=temp_max_c,
        window_start=context.start_date,
        window_end=context.end_date,
    )
    return climate.model_copy(update={"summary": render_summary(climate)})


def has_climate_answers(answers: list[str]) -> bool:
    return any("temp_min:" in answer or "temp_max:" in answer for answer in answers)


def apply_climate_from_answers(
    context: ResolvedContext, answers: list[str]
) -> ResolvedContext:
    """Fold user-supplied temperature chips into the context."""
    temp_min: float | None = None
    temp_max: float | None = None
    for answer in answers:
        for pair in answer.split(","):
            key, _, value = pair.partition(":")
            key, value = key.strip(), value.strip()
            if not value:
                continue
            try:
                if key == "temp_min":
                    temp_min = float(value)
                elif key == "temp_max":
                    temp_max = float(value)
            except ValueError:
                continue

    if temp_min is None and temp_max is None:
        return context

    climate = climate_from_answers(context, temp_min, temp_max)
    return context.model_copy(update={"climate": climate, "climate_note": climate.summary})


def build_climate_question(context: ResolvedContext) -> ClarifyingQuestion:
    """Ask when we could not verify conditions for a named place."""
    place = context.location or "that place"
    return ClarifyingQuestion(
        slot="climate",
        question=(
            f"We couldn't verify weather for {place}. "
            "Roughly what temperatures are you expecting?"
        ),
        options=[
            QuestionOption(label="Mild (10–25 °C)", value="temp_min:10,temp_max:25"),
            QuestionOption(label="Cool (0–10 °C)", value="temp_min:0,temp_max:10"),
            QuestionOption(label="Freezing nights (below 0 °C)", value="temp_min:-10,temp_max:5"),
            QuestionOption(label="Extreme cold (below −10 °C)", value="temp_min:-20,temp_max:-5"),
        ],
    )


def needs_place_climate(context: ResolvedContext) -> bool:
    """Whether this request implies weather we should look up."""
    return bool(
        context.location
        or context.location_lat is not None
        or context.location_lon is not None
    )
