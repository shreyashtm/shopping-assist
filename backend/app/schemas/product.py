"""Product catalogue schema.

Products are real, purchasable items sourced from retailer sites. Every record
carries provenance (`retailer`, `product_url`, `link_verified_at`) so a stale or
dead link can be traced back and re-verified by scripts/verify_links.py.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, HttpUrl

Gender = Literal["men", "women", "unisex", "kids"]
LinkStatus = Literal["verified", "blocked", "archival"]
Season = Literal["summer", "monsoon", "winter", "all-season"]

# Suitability axes (see services/constraints.py and services/suitability.py).
# Each is None until enriched -- unlike temp_rating_c there is no "does not
# apply" null; None means "not evaluated yet", and evaluate() must treat that
# as no opinion, not as a conflict, exactly like an unrated temp_rating_c.
WaterResistance = Literal["none", "repellent", "waterproof"]
Layer = Literal["base", "mid", "outer", "standalone"]
Formality = Literal["casual", "smart_casual", "formal"]
Breathability = Literal["low", "medium", "high"]


class ProductAttributes(BaseModel):
    """Structured facets used for filtering and for grounding LLM explanations.

    These are deliberately shopping-domain specific rather than generic tags:
    the retrieval layer filters on them directly, and the ranker cites them when
    justifying a pick (e.g. "rated to -10C for your late-October night temps").
    """

    gender: Gender = "unisex"
    season: list[Season] = Field(default_factory=lambda: ["all-season"])
    use_case: list[str] = Field(
        default_factory=list,
        description="Activity tags, e.g. 'trekking', 'wedding', 'gifting', 'daily-wear'.",
    )
    occasion: list[str] = Field(
        default_factory=list,
        description="Event tags, e.g. 'anniversary', 'festive', 'travel'.",
    )
    material: str | None = None
    temp_rating_c: int | None = Field(
        default=None,
        description="Lowest comfortable temperature in Celsius. Set for insulation gear only.",
    )
    is_giftable: bool = False

    water_resistance: WaterResistance | None = Field(
        default=None,
        description="Rain protection level. 'none' is a positive judgement (this "
        "garment offers no rain protection), distinct from null (not evaluated).",
    )
    layer: Layer | None = Field(
        default=None, description="Where this sits in a layering system, for apparel."
    )
    formality: Formality | None = Field(
        default=None, description="How dressy the item reads, for occasion matching."
    )
    breathability: Breathability | None = Field(
        default=None, description="Comfort signal only -- never a hard or strong penalty."
    )


class Product(BaseModel):
    id: str = Field(description="Stable slug, e.g. 'dcth-forclaz-mt100-jacket'.")
    title: str
    brand: str
    category: str = Field(description="Top-level group, e.g. 'Outdoor & Trekking'.")
    subcategory: str = Field(description="Leaf group, e.g. 'Insulated Jackets'.")

    price_inr: int
    mrp_inr: int | None = None

    description: str
    attributes: ProductAttributes = Field(default_factory=ProductAttributes)

    rating: float | None = Field(default=None, ge=0, le=5)
    review_count: int | None = Field(default=None, ge=0)

    retailer: str = Field(description="Source site, e.g. 'Decathlon India'.")
    product_url: HttpUrl
    image_url: HttpUrl | None = None

    in_stock: bool = True

    link_status: LinkStatus = Field(
        default="blocked",
        description="How well the product URL could be checked at build time. "
        "'verified' means a live product page was confirmed; 'blocked' means the "
        "retailer's bot protection prevented a check, so the link rests on "
        "provenance (it came from that retailer's own live search results); "
        "'archival' means the record came from a historical dataset rather than "
        "a live scrape, so the link and price are illustrative and were never "
        "checked at all -- never live-verifiable by construction, and never to "
        "be presented as though it were.",
    )
    link_verified_at: datetime | None = Field(
        default=None,
        description="Set only when link_status is 'verified'. Null otherwise, so "
        "nothing downstream can claim a link was checked when it was not.",
    )

    def to_prompt_line(self) -> str:
        """Compact one-line form fed to the ranker LLM.

        Kept terse on purpose: the ranker sees up to ~60 candidates, so every
        token per product multiplies. Only fields that can justify a pick are
        included.
        """
        attrs = self.attributes
        bits = [f"[{self.id}] {self.brand} {self.title}", f"{self.category}/{self.subcategory}"]
        bits.append(f"Rs.{self.price_inr}")
        if attrs.gender != "unisex":
            bits.append(attrs.gender)
        if attrs.temp_rating_c is not None:
            bits.append(f"rated {attrs.temp_rating_c}C")
        if attrs.material:
            bits.append(attrs.material)
        if attrs.use_case:
            bits.append("use:" + ",".join(attrs.use_case))
        if self.rating:
            bits.append(f"{self.rating}*")
        return " | ".join(bits) + f" :: {self.description[:110]}"
