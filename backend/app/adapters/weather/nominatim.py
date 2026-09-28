"""OpenStreetMap Nominatim transport: place name -> ranked candidates.

Transport only, like open_meteo.py. It exists because Open-Meteo's geocoder
indexes settlements alone and returns them in no meaningful order: a live check
sent "Leh" to Le Havre, "Goa" (a state, so absent) to Genoa, and "Manali" to a
Tamil Nadu town rather than the Himalayan one. Nominatim covers states, regions
and countries and orders results by prominence, so its first result is the
place a bare name usually means -- Leh, Ladakh; Goa, India; Dubai, UAE.

Usage policy (https://operations.osmfoundation.org/policies/nominatim/): an
identifying User-Agent and at most one request per second. Results are cached
upstream in services/context.py, so a repeated place costs nothing.
"""

import logging
import threading
import time

import httpx

from app.adapters.weather.open_meteo import Place, WeatherUnavailable

logger = logging.getLogger(__name__)

SEARCH_URL = "https://nominatim.openstreetmap.org/search"
USER_AGENT = "PersonalShoppingAssistant/1.0 (trip climate lookup)"
MIN_INTERVAL_S = 1.0

_throttle_lock = threading.Lock()
_last_request_at = 0.0


def _wait_for_slot() -> None:
    """Hold the process to Nominatim's one-request-per-second limit."""
    global _last_request_at
    with _throttle_lock:
        wait = MIN_INTERVAL_S - (time.monotonic() - _last_request_at)
        if wait > 0:
            time.sleep(wait)
        _last_request_at = time.monotonic()


class NominatimClient:
    def __init__(self, timeout_s: float = 6.0, transport: httpx.BaseTransport | None = None):
        self._client = httpx.Client(
            timeout=timeout_s, transport=transport, headers={"User-Agent": USER_AGENT}
        )

    def close(self) -> None:
        self._client.close()

    def search(self, name: str, limit: int = 5) -> list[Place]:
        """Candidates for a name, most prominent first. Empty is a normal answer."""
        _wait_for_slot()
        try:
            response = self._client.get(
                SEARCH_URL,
                params={
                    "q": name, "format": "jsonv2", "limit": limit,
                    "addressdetails": 1, "accept-language": "en",
                },
            )
            response.raise_for_status()
            results = response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise WeatherUnavailable(f"{SEARCH_URL}: {exc}") from exc
        if not isinstance(results, list):
            raise WeatherUnavailable(f"{SEARCH_URL}: unexpected response shape")

        places = []
        for entry in results:
            try:
                latitude, longitude = float(entry["lat"]), float(entry["lon"])
            except (KeyError, TypeError, ValueError):
                continue
            address = entry.get("address") or {}
            try:
                bbox = tuple(float(v) for v in entry["boundingbox"])
                bbox = bbox if len(bbox) == 4 else None
            except (KeyError, TypeError, ValueError):
                bbox = None
            places.append(
                Place(
                    name=entry.get("name") or name,
                    latitude=latitude,
                    longitude=longitude,
                    elevation_m=None,
                    country=address.get("country"),
                    admin1=address.get("state"),
                    population=None,
                    bbox=bbox,
                )
            )
        return places
