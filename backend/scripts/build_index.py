"""Precompute the catalogue embedding matrix.

Run once after the catalogue changes:

    uv run python scripts/build_index.py

Embeddings are computed at build time rather than per request for the obvious
reason -- encoding 1,738 products on every search would dominate latency -- and
written next to the catalogue as a plain .npy plus an id list. The id list is
what lets the loader detect a stale index instead of silently scoring products
against the wrong vectors.
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.embeddings.local import get_embedder  # noqa: E402
from app.schemas.product import Product  # noqa: E402
from app.services.catalogue import (  # noqa: E402
    EMBEDDING_IDS_FILE,
    EMBEDDINGS_FILE,
    PRODUCTS_FILE,
    embedding_text,
)


def main() -> int:
    if not PRODUCTS_FILE.exists():
        print(f"No catalogue at {PRODUCTS_FILE}. Run scripts/build_catalogue.py first.")
        return 1

    products = [Product.model_validate(item) for item in json.loads(PRODUCTS_FILE.read_text())]
    embedder = get_embedder()
    if not embedder.is_semantic:
        print(
            "WARNING: falling back to hashing embeddings -- retrieval will match on "
            "literal wording only. Install sentence-transformers for semantic search."
        )

    texts = [embedding_text(p) for p in products]
    print(f"encoding {len(texts)} products with {embedder.name} ({embedder.dimension}d)...")
    vectors = embedder.embed(texts)

    EMBEDDINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_FILE, vectors)
    EMBEDDING_IDS_FILE.write_text(json.dumps([p.id for p in products]))

    print(f"wrote {vectors.shape} -> {EMBEDDINGS_FILE.name}, {EMBEDDING_IDS_FILE.name}")

    # A quick sanity probe: the nearest neighbour of a plain-language need
    # should be a plausible product, not a random one.
    probe = "something warm for freezing nights on a Himalayan trek"
    scores = vectors @ embedder.embed([probe])[0]
    top = np.argsort(-scores)[:3]
    print(f'\nprobe: "{probe}"')
    for rank, index in enumerate(top, 1):
        print(f"  {rank}. {scores[index]:.3f}  {products[index].title[:64]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
