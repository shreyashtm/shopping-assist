"""Turn raw retailer listings into validated Product records.

Real listing data is messy in specific, repeatable ways, and each cleaning rule
below exists because the scraped data actually exhibited the problem:

* Amazon has no brand field at all, and titles are SEO keyword stuffing
  ("Men Polyester Sweatshirt Stylish Sweatshirts Breathable Jacket Full Zipper
  ...").  Brand is recovered from the leading token, title is truncated at the
  first separator.
* MRP is frequently nonsense -- ``originalPrice: 189100`` against a price of
  1891 (a misplaced decimal), or an MRP *below* the selling price. Both are
  dropped rather than shown as a fake discount.
* The same product appears under several ASINs/option codes with identical
  title and image. Deduplicated on (normalised title, image).
* Myntra serves images over http://, which the frontend's image host allowlist
  rejects. Upgraded to https.
* Ratings come back as 0 for "no ratings yet", which would otherwise render as
  a genuine zero-star score.
"""

import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from slugify import slugify  # noqa: E402
from sources import ALL_QUERIES, DEFAULT_KEEP, KEEP_PER_QUERY  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
RAW_DIR = BASE / "data" / "raw"
OUT_FILE = BASE / "data" / "products_normalized.json"

# Taxonomy is derived from sources.py rather than duplicated here. A second
# hand-written copy drifts the moment a query is added -- and with 41 queries
# across 12 categories, drift would be silent and hard to spot.
_BY_QUERY = {q.query: (q.category, q.subcategory) for q in ALL_QUERIES}

# Amazon tags each row with the `keyword` it came from; Myntra tags it with
# `searchQuery`. Both resolve through the same map.
KEYWORD_TAXONOMY = _BY_QUERY
SEARCH_TAXONOMY = _BY_QUERY

# Leading tokens that are descriptive rather than a brand name.
# Only used when the enrichment pass has not run; enrich.py recovers the real
# brand semantically, which a token heuristic fundamentally cannot do
# ("Winter Jacket For Men" has no brand, but its first token is capitalised).
NON_BRAND_PREFIXES = {
    "men", "mens", "men's", "men\u2019s", "women", "womens", "women's",
    "women\u2019s", "unisex", "boys", "girls", "kids", "premium", "new", "the",
    "pack", "set", "combo", "winter", "summer", "thermal", "waterproof",
    "sleeping", "camping", "hiking", "trekking", "rucksack", "fleece", "down",
    "puffer", "powder", "soft", "warm", "luxury", "scented", "silver", "perfume",
    "gift", "dry", "hand", "anti", "super", "best", "original", "genuine",
}


def clean_title(raw: str) -> str:
    """Trim SEO keyword stuffing down to a readable product name."""
    title = re.sub(r"\s+", " ", raw).strip()
    # Cut at the first structural separator -- everything after it is padding.
    for sep in ("|", "||", " - ", " – ", " — ", ","):
        if sep in title:
            head = title.split(sep)[0].strip()
            if len(head) >= 15:
                title = head
                break
    # Still absurd? Keep the first eight words, which is where the real name is.
    if len(title) > 80:
        title = " ".join(title.split()[:8]).rstrip(" ,-|")
    return title


def guess_brand(title: str) -> str:
    """Best-effort brand from a title that has no brand field (Amazon).

    Only the *first* token is considered. Scanning further into the title looks
    smarter but is reliably wrong -- "Men's Black Quilted Winter Jackets" would
    yield "Black" -- so when the leading token is descriptive this returns
    "Generic" and lets the enrichment pass recover the real brand semantically.
    """
    tokens = title.split()
    if not tokens:
        return "Generic"
    word = tokens[0].strip("|,-\u2013\u2014").strip()
    if len(word) <= 2 or word.lower() in NON_BRAND_PREFIXES or not word[0].isupper():
        return "Generic"
    return word


def sane_mrp(price: float, mrp: float | None) -> int | None:
    """Reject MRPs that cannot be real.

    An MRP at or below the selling price is not a discount, and one more than
    5x the price is a data error (observed: 189100 against a price of 1891).
    """
    if not mrp or mrp <= price:
        return None
    if mrp > price * 5:
        return None
    return int(round(mrp))


def https(url: str | None) -> str | None:
    if not url:
        return None
    return url.replace("http://", "https://", 1)


def norm_amazon(item: dict) -> dict | None:
    keyword = item.get("keyword", "")
    taxonomy = KEYWORD_TAXONOMY.get(keyword)
    if not taxonomy or not item.get("price") or not item.get("url"):
        return None
    title = clean_title(item.get("title", ""))
    if len(title) < 8:
        return None
    price = float(item["price"])
    rating = item.get("rating") or None
    return {
        "id": f"amz-{item['asin'].lower()}",
        "title": title,
        "brand": guess_brand(item.get("title", "")),
        "category": taxonomy[0],
        "subcategory": taxonomy[1],
        "price_inr": int(round(price)),
        "mrp_inr": sane_mrp(price, item.get("originalPrice")),
        "rating": float(rating) if rating else None,
        "review_count": item.get("reviewsCount") or None,
        "retailer": "Amazon.in",
        "product_url": item["url"],
        "image_url": https(item.get("imageUrl")),
        "source_query": keyword,
        "raw_title": item.get("title", ""),
    }


def norm_myntra(item: dict) -> dict | None:
    taxonomy = SEARCH_TAXONOMY.get(item.get("searchQuery", ""))
    if not taxonomy or not item.get("price") or not item.get("url"):
        return None
    price = float(item["price"])
    rating = item.get("rating") or None
    gender = (item.get("gender") or "").lower()
    return {
        "id": f"myn-{item['productId']}",
        "title": clean_title(item.get("name", "")),
        "brand": item.get("brand") or "Generic",
        "category": taxonomy[0],
        "subcategory": taxonomy[1],
        "price_inr": int(round(price)),
        "mrp_inr": sane_mrp(price, item.get("mrp")),
        "rating": round(float(rating), 1) if rating else None,
        "review_count": item.get("ratingCount") or None,
        "retailer": "Myntra",
        "product_url": item["url"],
        "image_url": https(item.get("imageUrl")),
        "gender_hint": "men" if gender == "men" else "women" if gender == "women" else None,
        "season_hint": (item.get("season") or "").lower() or None,
        "source_query": item.get("searchQuery", ""),
        "raw_title": item.get("name", ""),
        # True for rows gathered by browsing Myntra directly rather than via an
        # Apify actor (the free-tier scrape budget ran out mid-build).
        "curated": bool(item.get("_curated")),
    }


NORMALIZERS = {"Amazon.in": norm_amazon, "Myntra": norm_myntra}


def assign_ids(records: list[dict]) -> None:
    """Give every record a stable, readable, **unique** id, in place.

    The readable form is a retailer prefix plus the slugified title, truncated
    to 40 characters. That truncation is what made ids collide: three distinct
    Roadster watches ("...RDSTR-8008 Gold", "...RDSTR-8047 Black",
    "...RDSTR-1545D White Black") are identical well past 40 characters, so all
    three were assigned the same id. Dedupe did not catch them because it keys
    on (title, image) and these genuinely differ -- they are separate products
    that should both exist.

    Duplicate ids were not cosmetic. `Catalogue.load()` builds
    `{id: row}` from `embedding_ids.json`, so a repeated id collapses to the
    last row and the earlier products get scored against *another product's*
    embedding.

    On collision every colliding record is suffixed with a short hash of its
    product URL (which carries the retailer's own unique id). Suffixing all of
    them rather than "all but the first" keeps ids independent of iteration
    order, so a rebuild reproduces the committed catalogue instead of churning
    ids whenever the scrape order shifts.
    """
    bases = [f"{r['id'][:4]}{slugify(r['title'])[:40]}".strip("-") for r in records]
    collisions = {base for base, count in Counter(bases).items() if count > 1}

    for record, base in zip(records, bases, strict=True):
        if base in collisions:
            digest = hashlib.sha1(record["product_url"].encode()).hexdigest()[:6]
            record["id"] = f"{base}-{digest}"
        else:
            record["id"] = base


def dedupe_key(record: dict) -> tuple[str, str]:
    """Same visible name + same image = the same product relisted."""
    title = re.sub(r"[^a-z0-9]", "", record["title"].lower())
    return (title, record.get("image_url") or record["id"])


def main() -> int:
    if not RAW_DIR.exists():
        print(f"No raw data at {RAW_DIR}. Run scripts/fetch_raw.py first.")
        return 1

    per_query: dict[str, list[dict]] = {}
    seen: set[tuple[str, str]] = set()
    stats = {"read": 0, "dropped": 0, "deduped": 0}

    for path in sorted(RAW_DIR.glob("*.json")):
        items = json.loads(path.read_text())
        for item in items:
            stats["read"] += 1
            normalizer = NORMALIZERS.get(item.get("_retailer", ""))
            record = normalizer(item) if normalizer else None
            if not record:
                stats["dropped"] += 1
                continue
            key = dedupe_key(record)
            if key in seen:
                stats["deduped"] += 1
                continue
            seen.add(key)
            per_query.setdefault(record["source_query"], []).append(record)

    # Cap per search term so one over-scraped query cannot dominate the
    # catalogue and skew retrieval toward whatever Amazon returned most of.
    products: list[dict] = []
    for _query, records in sorted(per_query.items()):
        # Prefer listings with social proof, then cheaper -- both correlate with
        # being a real product rather than a drop-shipped relist.
        records.sort(key=lambda r: (-(r.get("review_count") or 0), r["price_inr"]))
        retailer = records[0]["retailer"]
        cap = KEEP_PER_QUERY.get(retailer, DEFAULT_KEEP)
        products.extend(records[: min(cap, len(records))])

    # Stable, readable, unique ids.
    assign_ids(products)

    OUT_FILE.write_text(json.dumps(products, indent=2, ensure_ascii=False))
    print(
        f"read={stats['read']} dropped={stats['dropped']} "
        f"deduped={stats['deduped']} kept={len(products)}"
    )
    by_cat: dict[str, int] = {}
    for record in products:
        by_cat[record["category"]] = by_cat.get(record["category"], 0) + 1
    for cat, count in sorted(by_cat.items()):
        print(f"  {cat:<22} {count}")
    print(f"-> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
