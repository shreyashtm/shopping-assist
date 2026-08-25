"""Open-Meteo transport.

Transport only -- no provenance decisions, no corroboration, no shopping
knowledge. Those are judgements and they live in services/context.py. What
happens here is: build a URL, handle the ways Open-Meteo fails, return the
numbers it gave us.

Four endpoints, all free and keyless:

  geocode      name -> candidate places (GeoNames-backed: settlements only)
  elevation    lat/lon -> metres, from a 90m digital elevation model
  forecast     lat/lon + dates -> daily values, up to ~16 days ahead
  climatology  lat/lon + dates -> daily values from a downscaled climate model,
               covering 1940-2050, which is what makes a trip four months out
               answerable with real numbers instead of a guess

The elevation endpoint is the interesting one. It is what lets a coordinate
proposed by a language model be *checked* rather than trusted: a model that
places Hampta Pass in Karnataka proposes a point at 765m, and a claimed 4,200m
pass at 765m is a rejection, not a rounding error.
"""

import logging
from dataclasses import dataclass
from datetime import date

import httpx

logger = logging.getLogger(__name__)

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
ELEVATION_URL = "https://api.open-meteo.com/v1/elevation"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
CLIMATE_URL = "https://climate-api.open-meteo.com/v1/climate"

# Beyond this the forecast endpoint refuses the date range outright:
# "Parameter 'start_date' is out of allowed range". Past it, climatology is the
# only source of real numbers.
FORECAST_HORIZON_DAYS = 16

# Downscaled to 10km, which matters in mountains: a coarse model averages a
# 4,400m pass with the valley next to it and reports neither.
CLIMATE_MODEL = "MRI_AGCM3_2_S"

DAILY_FIELDS = "temperature_2m_max,temperature_2m_min,precipitation_sum"


class WeatherUnavailable(RuntimeError):
    """Raised when Open-Meteo cannot answer.

    Callers must treat this as "we do not know" and say so. It is never a
    licence to fall back to an invented number -- that failure is the entire
    reason this module exists.
    """


@dataclass(frozen=True)
class Place:
    """One geocoding candidate."""

    name: str
    latitude: float
    longitude: float
    elevation_m: float | None
    country: str | None
    admin1: str | None
    population: int | None

    @property
    def display(self) -> str:
        parts = [p for p in (self.name, self.admin1, self.country) if p]
        return ", ".join(parts)


@dataclass(frozen=True)
class DailySeries:
    """Raw daily values over the requested window.

    Returned unaggregated on purpose. "Coldest night of the trip" is a
    shopping judgement, not a property of the data, so the reduction happens
    in the service layer where that judgement belongs.
    """

    dates: list[date]
    temp_min_c: list[float]
    temp_max_c: list[float]
    precipitation_mm: list[float]
    elevation_m: float | None


class OpenMeteoClient:
    """Sync HTTP client for Open-Meteo.

    Sync because the recommendation pipeline is a sync generator; an async
    client here would mean colouring the whole call chain for three requests
    that take a few hundred milliseconds.
    """

    def __init__(self, timeout_s: float = 6.0, transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(
            timeout=timeout_s,
            transport=transport,
            headers={"User-Agent": "shopping-assistant/0.1"},
        )

    def close(self) -> None:
        self._client.close()

    def _get(self, url: str, params: dict) -> dict:
        """One request, one retry.

        Open-Meteo answers overload with HTTP 200 and `{"error": true,
        "reason": "The service is overloaded"}` -- observed live during
        development -- so the status code alone is not the health signal.
        """
        last: Exception | None = None
        for attempt in (1, 2):
            try:
                response = self._client.get(url, params=params)
                response.raise_for_status()
                payload = response.json()
            except (httpx.HTTPError, ValueError) as exc:
                last = exc
                logger.debug("Open-Meteo %s attempt %d failed: %s", url, attempt, exc)
                continue
            if payload.get("error"):
                last = WeatherUnavailable(str(payload.get("reason", "unknown error")))
                logger.debug("Open-Meteo %s attempt %d: %s", url, attempt, last)
                continue
            return payload
        raise WeatherUnavailable(f"{url}: {last}") from last

    def geocode(self, name: str, count: int = 5) -> list[Place]:
        """Candidate places for a name.

        GeoNames-backed, so it holds settlements and not natural features:
        "Manali" resolves, "Hampta Pass" returns nothing at all. An empty list
        is a normal answer here, not an error.
        """
        payload = self._get(GEOCODE_URL, {"name": name, "count": count, "format": "json"})
        return [
            Place(
                name=entry.get("name", name),
                latitude=float(entry["latitude"]),
                longitude=float(entry["longitude"]),
                elevation_m=_as_float(entry.get("elevation")),
                country=entry.get("country"),
                admin1=entry.get("admin1"),
                population=entry.get("population"),
            )
            for entry in payload.get("results", [])
            if entry.get("latitude") is not None and entry.get("longitude") is not None
        ]

    def elevation(self, latitude: float, longitude: float) -> float:
        """Ground elevation in metres at a point."""
        payload = self._get(ELEVATION_URL, {"latitude": latitude, "longitude": longitude})
        values = payload.get("elevation") or []
        if not values or values[0] is None:
            raise WeatherUnavailable("elevation endpoint returned no value")
        return float(values[0])

    def forecast(self, latitude: float, longitude: float, start: date, end: date) -> DailySeries:
        """Measured forecast. Only valid inside FORECAST_HORIZON_DAYS."""
        return self._series(
            FORECAST_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "daily": DAILY_FIELDS,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "timezone": "auto",
            },
        )

    def climatology(
        self, latitude: float, longitude: float, start: date, end: date
    ) -> DailySeries:
        """Modelled normals for dates beyond the forecast horizon."""
        return self._series(
            CLIMATE_URL,
            {
                "latitude": latitude,
                "longitude": longitude,
                "daily": DAILY_FIELDS,
                "start_date": start.isoformat(),
                "end_date": end.isoformat(),
                "models": CLIMATE_MODEL,
            },
        )

    def _series(self, url: str, params: dict) -> DailySeries:
        payload = self._get(url, params)
        daily = payload.get("daily") or {}
        dates = [date.fromisoformat(d) for d in daily.get("time", [])]
        if not dates:
            raise WeatherUnavailable(f"{url}: no daily values returned")
        return DailySeries(
            dates=dates,
            temp_min_c=_floats(daily.get("temperature_2m_min")),
            temp_max_c=_floats(daily.get("temperature_2m_max")),
            precipitation_mm=_floats(daily.get("precipitation_sum")),
            elevation_m=_as_float(payload.get("elevation")),
        )


def _floats(values: list | None) -> list[float]:
    """Drop nulls rather than substituting zero.

    A missing temperature is not 0C, and a series with a fabricated zero in it
    would quietly drag the trip's minimum in the wrong direction.
    """
    return [float(v) for v in (values or []) if v is not None]


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
