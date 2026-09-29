"""sentence-transformers embeddings, run locally.

Local rather than hosted for a specific reason: Anthropic has no embedding
endpoint, so a hosted embedder would mean a second vendor and a second API key
for what is a solved, cheap, offline problem. all-MiniLM-L6-v2 is ~90MB, encodes
the whole 1,725-product catalogue in a second or two, and costs nothing per query.
"""

import logging
import os
import time
from functools import lru_cache

import numpy as np

from app.adapters.embeddings.base import l2_normalise


logger = logging.getLogger(__name__)

# Phrases like a real search's, for timing the encoder at startup.
_PROBE_PHRASES = [
    "insulated down jacket", "thermal base layer", "waterproof trekking boots",
    "wool hiking socks", "headlamp", "trekking backpack", "fleece jacket",
    "gift hamper", "formal shirt", "running shoes", "kurta set", "sunglasses",
]


def container_cpus() -> int:
    """CPUs this process may actually use, which in a container is its quota.

    torch sizes its thread pool from the host's core count. On Railway the
    host has many cores but the service is capped at 2 vCPU, so torch ran a
    dozens-strong pool on two CPUs: a search stage that takes 0.2s locally
    took 84s deployed, CPU pinned at the cap, no I/O in the logs.
    """
    try:
        with open("/sys/fs/cgroup/cpu.max") as f:  # cgroup v2: "<quota> <period>"
            quota, period = f.read().split()
        if quota != "max":
            return max(1, int(int(quota) / int(period)))
    except (OSError, ValueError):
        pass
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # macOS
        return max(1, os.cpu_count() or 1)


def _time_encode(model) -> float:
    started = time.perf_counter()
    model.encode(_PROBE_PHRASES, batch_size=32, show_progress_bar=False, convert_to_numpy=True)
    return (time.perf_counter() - started) * 1000


class LocalEmbeddings:
    name = "all-MiniLM-L6-v2"
    dimension = 384
    is_semantic = True

    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        # Imported lazily: importing sentence_transformers pulls in torch, which
        # is slow enough to notice on process start and pointless when the
        # fallback is in use.
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(model_name)
        self.name = model_name.split("/")[-1]
        # Renamed in sentence-transformers 5.x; fall back for older installs.
        get_dim = getattr(
            self._model, "get_embedding_dimension", None
        ) or self._model.get_sentence_embedding_dimension
        self.dimension = get_dim()
        self._fit_threads_to_container()

    def _fit_threads_to_container(self) -> None:
        """Size torch's thread pool to the CPUs actually available, and log
        the encode time, so a slow host shows in the deploy log.

        Measured on Railway (30 Sep 2026) with a before-and-after run: torch
        started 48 threads on a 2-CPU container, and 12 phrases took 15.1s;
        capped at 2 threads they took 0.21s. The "before" run is not repeated
        at every startup -- it cost those 15 seconds each time.
        """
        import torch

        default_threads = torch.get_num_threads()
        cpus = container_cpus()
        if cpus < default_threads:  # only ever narrow the pool
            torch.set_num_threads(cpus)
        _time_encode(self._model)  # first call pays one-off setup
        took = _time_encode(self._model)
        logger.info(
            "Encoder threads: %d -> %d (container CPUs %d); %d phrases took %.0f ms",
            default_threads, torch.get_num_threads(), cpus, len(_PROBE_PHRASES), took,
        )

    def embed(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, show_progress_bar=False, convert_to_numpy=True
        )
        return l2_normalise(np.asarray(vectors, dtype=np.float32))


@lru_cache(maxsize=1)
def get_embedder(prefer_local: bool = True):
    """Return the best available provider, falling back without raising."""
    if prefer_local:
        try:
            return LocalEmbeddings()
        except Exception as exc:  # noqa: BLE001 - any failure means fall back
            import logging

            logging.getLogger(__name__).warning(
                "Local embedding model unavailable (%s); using hashing fallback. "
                "Retrieval will be keyword-ish, not semantic.",
                exc,
            )
    from app.adapters.embeddings.hashing import HashingEmbeddings

    return HashingEmbeddings()
