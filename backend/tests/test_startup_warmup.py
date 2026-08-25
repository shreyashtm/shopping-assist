"""Tests for startup warm-up.

`load_embedder()` exists so the first real search does not pay the ~7s
sentence-transformers load (measured: 22.6s first request vs 7.5s warm). Two
properties matter and both are easy to regress:

1. it must actually populate the cache `recommend.py` later reads, and
2. it must never raise -- a container that refuses to boot is worse than one
   serving weaker matches with `degraded_mode` set.
"""

from app.adapters.embeddings.local import get_embedder
from app.core import deps


def test_load_embedder_populates_the_cache_recommend_reads():
    get_embedder.cache_clear()
    assert get_embedder.cache_info().currsize == 0

    deps.load_embedder()

    assert get_embedder.cache_info().currsize == 1


def test_load_embedder_survives_a_failing_embedder(monkeypatch):
    """Startup must not die because the model could not be built."""
    import app.adapters.embeddings.local as local_module

    def boom(*_args, **_kwargs):
        raise RuntimeError("no model available")

    monkeypatch.setattr(local_module, "get_embedder", boom)

    deps.load_embedder()  # must not raise
