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
from collections import Counter
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

    # Unique ids are a correctness requirement, not a nicety: Catalogue.load()
    # maps id -> embedding row, so a repeated id silently collapses to one row
    # and the other products carrying that id get scored against a different
    # product's vector. Caught here rather than shipped, because the symptom
    # downstream is invisible -- slightly wrong ranking, plus a duplicate-key
    # warning in the UI if two of them land in the same group.
    counts = Counter(p.id for p in products)
    duplicates = {pid: n for pid, n in counts.items() if n > 1}
    if duplicates:
        print(f"ERROR: {len(duplicates)} duplicate product ids in {PRODUCTS_FILE.name}:")
        for pid, n in sorted(duplicates.items()):
            print(f"  {pid} x{n}")
        print("\nRe-run scripts/normalize.py -- assign_ids() disambiguates collisions.")
        return 1

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
