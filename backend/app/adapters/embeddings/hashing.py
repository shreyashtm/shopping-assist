"""Deterministic embedding fallback -- no model download, no network.

This is not a good text encoder and does not pretend to be. It hashes character
n-grams into a fixed vector space, which captures literal token overlap and
nothing else: "down jacket" and "insulated parka" score near zero together.

It exists so that a fresh clone can run the app end to end before pulling a
~90MB model, and so that a missing model degrades loudly (`is_semantic = False`
propagates to `degraded_mode` in the API response) rather than crashing.
"""

import hashlib

import numpy as np

from app.adapters.embeddings.base import l2_normalise

DIMENSION = 384


class HashingEmbeddings:
    name = "hashing-fallback"
    dimension = DIMENSION
    is_semantic = False

    def _ngrams(self, text: str) -> list[str]:
        words = text.lower().split()
        grams = list(words)
        grams += [f"{a}_{b}" for a, b in zip(words, words[1:], strict=False)]
        return grams

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dimension), dtype=np.float32)
        for row, text in enumerate(texts):
            for gram in self._ngrams(text):
                digest = hashlib.md5(gram.encode("utf-8")).digest()
                index = int.from_bytes(digest[:4], "little") % self.dimension
                # Sign from a second byte keeps unrelated grams from always
                # accumulating in the same direction.
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, index] += sign
        return l2_normalise(out)
