"""sentence-transformers embeddings, run locally.

Local rather than hosted for a specific reason: Anthropic has no embedding
endpoint, so a hosted embedder would mean a second vendor and a second API key
for what is a solved, cheap, offline problem. all-MiniLM-L6-v2 is ~90MB, encodes
the whole 1,738-product catalogue in a second or two, and costs nothing per query.
"""

from functools import lru_cache

import numpy as np

from app.adapters.embeddings.base import l2_normalise


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
