"""Tests for catalogue cleaning rules.

Every case here is taken from data the scrapers actually returned, not invented
edge cases -- these are the specific ways real retailer listings were broken.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from normalize import (  # noqa: E402
    clean_title,
    dedupe_key,
    guess_brand,
    https,
    norm_amazon,
    sane_mrp,
)


def test_clean_title_trims_seo_keyword_stuffing():
    raw = (
        "Men Polyester Sweatshirt Stylish Sweatshirts Breathable Jacket Full Zipper "
        "Sweatshirt Casual Sweat Shirt Full Sleeves Sweaters All Day Winter wear Jackets Stylish"
    )
    assert len(clean_title(raw)) <= 80


def test_clean_title_cuts_at_separator():
    raw = "DRIFT Jacket – Black | Premium Polyester Fleece with Ecolite Fabric | Men"
    assert clean_title(raw) == "DRIFT Jacket – Black"


def test_guess_brand_skips_descriptive_prefix():
    assert guess_brand("Men Lace Up Hiking Boot") != "Men"
    assert guess_brand("Boldfit Trekking Shoes for Man Outdoor") == "Boldfit"
    assert guess_brand("Lux Cottswool Men's Cotton Thermal Set") == "Lux"


def test_sane_mrp_rejects_misplaced_decimal():
    # Observed: price 1891, originalPrice 189100.
    assert sane_mrp(1891, 189100) is None


def test_sane_mrp_rejects_mrp_below_price():
    # Observed: price 799, originalPrice 399.5 -- would render as a fake discount.
    assert sane_mrp(799, 399.5) is None


def test_sane_mrp_keeps_plausible_discount():
    assert sane_mrp(1469, 3999) == 3999


def test_https_upgrades_myntra_images():
    url = "http://assets.myntassets.com/assets/images/2026/AUGUST/14/abc.jpg"
    assert https(url).startswith("https://")


def test_dedupe_key_matches_relisted_product():
    # Two ASINs, identical title and image -- the same jacket relisted.
    a = {"title": "Men Polyester Sweatshirt", "image_url": "https://x/1.jpg", "id": "amz-a"}
    b = {"title": "Men  Polyester   Sweatshirt", "image_url": "https://x/1.jpg", "id": "amz-b"}
    assert dedupe_key(a) == dedupe_key(b)


def test_norm_amazon_drops_listing_without_price():
    item = {
        "asin": "X1",
        "title": "Some Jacket",
        "url": "https://a/x",
        "keyword": "fleece jacket men",
    }
    assert norm_amazon(item) is None


def test_norm_amazon_builds_valid_record():
    item = {
        "asin": "B0D1KMC1VL",
        "title": "Boldfit Trekking Shoes for Man Outdoor Hiking Shoes",
        "url": "https://www.amazon.in/dp/B0D1KMC1VL",
        "price": 1469,
        "originalPrice": 3999,
        "rating": 4,
        "reviewsCount": 1800,
        "imageUrl": "https://m.media-amazon.com/images/I/61BduAeitgL._AC_UL320_.jpg",
        "keyword": "trekking shoes men waterproof",
    }
    record = norm_amazon(item)
    assert record["brand"] == "Boldfit"
    assert record["category"] == "Outdoor & Trekking"
    assert record["subcategory"] == "Footwear"
    assert record["price_inr"] == 1469
    assert record["mrp_inr"] == 3999
    assert record["product_url"].startswith("https://www.amazon.in/")
