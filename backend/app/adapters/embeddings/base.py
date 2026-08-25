"""Embedding provider interface.

Two implementations sit behind this: a real sentence-transformers model, and a
deterministic hashing fallback. The fallback exists so the app boots and serves
requests on a machine that has never downloaded the model -- retrieval quality
degrades, but nothing crashes and the failure is visible rather than silent.
"""

from typing import Protocol

import numpy as np


class EmbeddingProvider(Protocol):
    """Turns text into L2-normalised vectors.

    Normalisation is part of the contract, not an implementation detail: it lets
    the retrieval layer use a plain dot product for cosine similarity, which is
    a single matrix multiply over the whole catalogue.
    """

    name: str
    dimension: int
    is_semantic: bool
    """False for the hashing fallback, so callers can flag degraded results."""

    def embed(self, texts: list[str]) -> np.ndarray:
        """Return an (len(texts), dimension) float32 array of unit vectors."""
        ...


def l2_normalise(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    # Zero vectors would divide by zero; map them to zero rather than NaN so a
    # degenerate input cannot poison a whole similarity matrix.
    norms[norms == 0] = 1.0
    return (vectors / norms).astype(np.float32)
