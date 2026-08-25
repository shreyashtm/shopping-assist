"""In-memory product catalogue.

At 289 products the entire catalogue and its embedding matrix fit comfortably in
process memory, so there is no database here on purpose: a JSON file plus a
numpy array answers every query this app makes, with no connection pool, no
migration, and no second service to run. The trade is documented rather than
hidden -- past roughly 50k products this should move to pgvector, and the
loader below is the seam where that swap would happen.
"""

import json
import logging
from pathlib import Path

import numpy as np

from app.core.config import DATA_DIR
from app.core.errors import CatalogueNotReady
from app.schemas.product import Product

logger = logging.getLogger(__name__)

PRODUCTS_FILE = DATA_DIR / "products.json"
EMBEDDINGS_FILE = DATA_DIR / "embeddings.npy"
EMBEDDING_IDS_FILE = DATA_DIR / "embedding_ids.json"


def embedding_text(product: Product) -> str:
    """The text a product is indexed under.

    Deliberately more than the title. A shopper asking for "something warm for
    sub-zero nights" matches on the description and attribute tags, never on a
    product name like "Mens Arctic Crest" -- so the attributes that enrichment
    inferred are folded into the indexed text rather than used only as filters.
    """
    attrs = product.attributes
    parts = [
        product.title,
        product.brand,
        product.category,
        product.subcategory,
        product.description,
    ]
    if attrs.use_case:
        parts.append("for " + ", ".join(attrs.use_case))
    if attrs.occasion:
        parts.append("occasion " + ", ".join(attrs.occasion))
    if attrs.season:
        parts.append("season " + ", ".join(attrs.season))
    if attrs.material:
        parts.append(attrs.material)
    if attrs.temp_rating_c is not None:
        parts.append(f"rated to {attrs.temp_rating_c} degrees celsius")
    if attrs.is_giftable:
        parts.append("suitable as a gift")
    return " | ".join(p for p in parts if p)


class Catalogue:
    """Products plus their embedding matrix, kept row-aligned."""

    def __init__(self, products: list[Product], vectors: np.ndarray | None = None):
        self.products = products
        self.by_id = {p.id: p for p in products}
        self.vectors = vectors
        if vectors is not None and len(vectors) != len(products):
            raise ValueError(
                f"Embedding matrix has {len(vectors)} rows but catalogue has "
                f"{len(products)} products; rebuild the index."
            )

    def __len__(self) -> int:
        return len(self.products)

    @property
    def has_vectors(self) -> bool:
        return self.vectors is not None

    @classmethod
    def load(cls, products_file: Path = PRODUCTS_FILE) -> "Catalogue":
        if not products_file.exists():
            raise CatalogueNotReady(
                f"No catalogue at {products_file}. "
                "Run: uv run python scripts/build_catalogue.py"
            )
        raw = json.loads(products_file.read_text())
        products = [Product.model_validate(item) for item in raw]

        vectors = None
        if EMBEDDINGS_FILE.exists() and EMBEDDING_IDS_FILE.exists():
            ids = json.loads(EMBEDDING_IDS_FILE.read_text())
            matrix = np.load(EMBEDDINGS_FILE)
            # Reorder the matrix to match product order rather than trusting the
            # two files to have been written in lockstep. A stale index that is
            # merely out of order should not silently mis-attribute scores.
            position = {pid: i for i, pid in enumerate(ids)}
            if all(p.id in position for p in products):
                vectors = np.vstack([matrix[position[p.id]] for p in products])
            else:
                logger.warning(
                    "Embedding index is stale (%d ids, %d products); "
                    "ignoring it until rebuilt.",
                    len(ids),
                    len(products),
                )
        return cls(products, vectors)
