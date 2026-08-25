"""Response cache for identical searches.

Every search costs two LLM calls, and identical requests are common in practice:
a shopper reloads, taps back, or re-runs the same example from the landing page.
Serving those from memory is the single cheapest optimisation available -- it
removes 100% of the API cost for a repeat rather than some fraction of it.

Deliberately a bounded in-process dict rather than Redis. The app is stateless
and horizontally scalable; a per-instance cache trades a lower hit rate across
replicas for having no extra service to run, which is the right trade at this
size. The seam to swap it is this module.
"""

import hashlib
import time
from collections import OrderedDict
from typing import Any

DEFAULT_TTL_SECONDS = 60 * 30
DEFAULT_MAX_ENTRIES = 256


def cache_key(query: str, answers: list[str], skip_clarification: bool) -> str:
    """Stable key for a request.

    Query text is normalised (lowercased, whitespace-collapsed) so trivial
    differences in typing do not miss the cache, and answers are sorted because
    tapping the same chips in a different order is the same request.
    """
    normalised = " ".join(query.lower().split())
    payload = "|".join([normalised, ",".join(sorted(answers)), str(skip_clarification)])
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


class ResponseCache:
    def __init__(
        self,
        ttl_seconds: int = DEFAULT_TTL_SECONDS,
        max_entries: int = DEFAULT_MAX_ENTRIES,
    ):
        self.ttl_seconds = ttl_seconds
        self.max_entries = max_entries
        self._entries: OrderedDict[str, tuple[float, Any]] = OrderedDict()
        self.hits = 0
        self.misses = 0

    def get(self, key: str) -> Any | None:
        entry = self._entries.get(key)
        if entry is None:
            self.misses += 1
            return None
        stored_at, value = entry
        if time.time() - stored_at > self.ttl_seconds:
            # Expired. Catalogue prices drift, so stale recommendations are
            # worse than a cache miss.
            del self._entries[key]
            self.misses += 1
            return None
        self._entries.move_to_end(key)
        self.hits += 1
        return value

    def set(self, key: str, value: Any) -> None:
        self._entries[key] = (time.time(), value)
        self._entries.move_to_end(key)
        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = self.misses = 0

    @property
    def stats(self) -> dict[str, int | float]:
        total = self.hits + self.misses
        return {
            "entries": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": round(self.hits / total, 3) if total else 0.0,
        }


response_cache = ResponseCache()
