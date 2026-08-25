"""Sample a fashion slice from the Kaggle Amazon-products archive.

This is a second catalogue source, alongside the Apify scrape in fetch_raw.py --
not a replacement. The Apify scrape targets Amazon.in and Myntra directly and
stays the primary, INR-native path; this script draws from a bulk historical US
dataset to fill taxonomy paths the scrape never reached (Section: which gaps).

## Why a second source at all

34 of the catalogue's taxonomy paths hold zero products -- no formal shirts, no
jeans, no dresses, no formal shoes, no wallets. Reaching them via Apify means
designing and paying for dozens more scrape queries. This dataset already has
them: 1.4M US Amazon listings across 249 categories, dumped once in Feb 2024.

## Why not just use it for everything

Three reasons this stays a secondary source, not a wholesale replacement:

* **It is not live.** Prices, stock and links are frozen at scrape time and were
  never checked against a real page. `link_status="archival"` records that
  honestly (see app/schemas/product.py) rather than presenting a 2024 US price
  as a verified INR listing.
* **It is US-priced.** Converted at a fixed, documented rate (INR_PER_USD)
  rather than a live FX lookup -- the rate is for catalogue realism, not for
  telling anyone what a rupee is worth today.
* **It has almost no Indian ethnic wear.** 0.08% of the fashion slice matches
  sherwani/kurta/saree/lehenga vocabulary at all. Ethnic Wear stays Myntra's.

## Which gaps this closes

Only apparel, footwear, bags and jewellery -- the categories this dataset
actually carries. Electronics, Home & Kitchen and Beauty & Personal Care are
also empty in the taxonomy but this source has nothing for them; filling those
needs its own scrape queries, not more of this one.

## How selection works

1. Restrict to the archive's fashion + travel-adjacent category ids.
2. Quality-filter: priced, rated, imaged, linked, a real title.
3. Classify each title against the SAME canonical taxonomy the rest of the
   catalogue uses (app.services.taxonomy), by keyword -- crude on purpose. This
   is a *sampling* filter, not the final classification: enrich_kaggle.py makes
   the authoritative call with an LLM that can see the whole title and is told
   to override a wrong guess. Keyword-matching only decides who gets *offered*
   to that pass, not who ends up where.
4. Stratify: every currently-empty apparel/footwear/bag/jewellery path gets up
   to PER_PATH samples; monsoon-relevant items (rain/waterproof/quick-dry) are
   separately oversampled across the paths this catalogue got most wrong,
   because that failure mode -- a winter jacket recommended for monsoon wear --
   is what this ingestion exists to fix.

Output feeds enrich_kaggle.py, not normalize.py: this source has no per-query
grouping to dedupe against and its category assignment is provisional, so it
skips straight to the combined classify+enrich pass.

    uv run python scripts/sample_kaggle.py
"""

import csv
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from normalize import clean_title, https, sane_mrp  # noqa: E402

from app.services.taxonomy import ALL_PATHS  # noqa: E402

ARCHIVE_DIR = Path(__file__).resolve().parents[2] / "archive"
PRODUCTS_CSV = ARCHIVE_DIR / "amazon_products.csv"
CATEGORIES_CSV = ARCHIVE_DIR / "amazon_categories.csv"

BASE = Path(__file__).resolve().parent.parent
OUT_FILE = BASE / "data" / "raw" / "kaggle_fashion_sample.json"

# Documented, fixed -- not a live FX rate. Only needs to be roughly right for
# catalogue realism; nothing downstream computes anything financial from it.
INR_PER_USD = 83.0

# Fashion + travel-adjacent category ids from amazon_categories.csv. Restricted
# to what this script can actually classify into the existing taxonomy -- see
# the module docstring for why electronics/home/beauty are excluded even though
# they are also empty.
FASHION_CATEGORY_IDS = {
    "43", "84", "87", "88", "89", "90", "91", "94", "95", "96", "97", "98",
    "110", "112", "113", "114", "116", "118", "120", "121", "122", "123",
    "264", "265",
}
ADJACENT_CATEGORY_IDS = {
    "99", "100", "101", "102", "103", "104", "105", "106", "107", "108", "109",
}

MIN_TITLE_LEN = 15

# Keyword -> canonical path. Order matters: the first pattern that matches
# wins, so more specific garment words are listed before generic ones.
PATH_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\b(dress shirt|formal shirt|button.?down)\b", re.I),
     "Men's Apparel/Formal Shirts"),
    (re.compile(r"\bt.?shirt\b", re.I), "Men's Apparel/T-Shirts"),
    (re.compile(r"\b(chino|dress pant)\b", re.I), "Men's Apparel/Trousers & Chinos"),
    (re.compile(r"\bjean(s)?\b", re.I), "Men's Apparel/Jeans"),
    (re.compile(r"\b(blazer|suit jacket|sport coat|suit set)\b", re.I),
     "Men's Apparel/Suits & Blazers"),
    (re.compile(r"\bshort(s)?\b", re.I), "Men's Apparel/Shorts"),
    (re.compile(r"\bdress\b", re.I), "Women's Apparel/Dresses"),
    (re.compile(r"\b(trouser|wide leg pant|work pant|dress pant)\b", re.I),
     "Women's Apparel/Trousers"),
    (re.compile(r"\bskirt\b", re.I), "Women's Apparel/Skirts"),
    (re.compile(r"\bblazer\b", re.I), "Women's Apparel/Blazers"),
    (re.compile(r"\b(jacket|coat|parka|windbreaker|raincoat)\b", re.I),
     "Women's Apparel/Jackets & Coats"),
    (re.compile(r"\b(sweater|fleece|cardigan|pullover)\b", re.I),
     "Women's Apparel/Sweaters & Fleece"),
    (re.compile(r"\b(oxford|derby|loafer|dress shoe)\b", re.I), "Footwear/Formal Shoes"),
    (re.compile(r"\bsneaker\b", re.I), "Footwear/Casual Sneakers"),
    (re.compile(r"\bsandal|flip.?flop|slide\b", re.I), "Footwear/Sandals & Floaters"),
    (re.compile(r"\b(heel|pump|stiletto)\b", re.I), "Footwear/Heels"),
    (re.compile(r"\b(ballet flat|flats)\b", re.I), "Footwear/Flats"),
    (re.compile(r"\bwallet\b", re.I), "Bags & Luggage/Wallets"),
    (re.compile(r"\bduffel|duffle\b", re.I), "Bags & Luggage/Duffels"),
    (re.compile(r"\b(suitcase|spinner|carry.on|luggage)\b", re.I),
     "Bags & Luggage/Luggage & Trolleys"),
    (re.compile(r"\b(necklace|bracelet|earring|pendant|ring)\b", re.I),
     "Watches & Jewellery/Jewellery"),
]

# Everything this path list can target, so a corrupt category id or an
# unmatched title fails loudly instead of silently mapping to something wrong.
assert all(path in ALL_PATHS for _, path in PATH_PATTERNS)

MONSOON_PATTERN = re.compile(
    r"\b(rain|waterproof|water.resistant|quick.dry|umbrella|poncho|"
    r"breathable|moisture.wicking)\b",
    re.I,
)

# Per empty path. Kept modest: enrichment cost and time scale with this, and a
# few dozen genuine examples per gap does more for retrieval than a thousand
# near-duplicates would.
PER_PATH = 70
# Monsoon items are sampled on top of (not instead of) their path quota, since
# the goal is specifically to give existing outerwear/footwear/bag buckets
# rain-appropriate options to rank alongside what is already there.
MONSOON_TARGET = 200


def classify(title: str) -> str | None:
    for pattern, path in PATH_PATTERNS:
        if pattern.search(title):
            return path
    return None


def to_record(row: dict, path: str, monsoon: bool) -> dict:
    """Shape matches products_normalized.json, so enrich_kaggle.py reads it
    the same way enrich.py reads the Apify-derived catalogue."""
    price_usd = float(row["price"])
    title = clean_title(row["title"])
    return {
        "id": f"kag-{row['asin'].lower()}",
        "title": title,
        "raw_title": row["title"],
        "brand": "Generic",  # recovered by enrich_kaggle.py, same as Amazon.in
        "category": path.split("/")[0],
        "subcategory": path.split("/")[1],
        "candidate_path": path,  # a hint for the LLM pass, not authoritative
        "monsoon_candidate": monsoon,
        "price_inr": int(round(price_usd * INR_PER_USD)),
        "mrp_inr": sane_mrp(
            price_usd * INR_PER_USD,
            (float(row["listPrice"]) * INR_PER_USD) if row.get("listPrice") else None,
        ),
        "rating": round(float(row["stars"]), 1) if row.get("stars") else None,
        "review_count": int(row["reviews"]) if row.get("reviews") else None,
        "retailer": "Amazon.com (2024 archive)",
        "product_url": row["productURL"],
        "image_url": https(row["imgUrl"]),
        "source": "kaggle-archive",
    }


def main() -> int:
    if not PRODUCTS_CSV.exists():
        print(f"No archive at {PRODUCTS_CSV}.")
        return 1

    keep_ids = FASHION_CATEGORY_IDS | ADJACENT_CATEGORY_IDS
    per_path: dict[str, list[dict]] = {path: [] for _, path in PATH_PATTERNS}
    monsoon: list[dict] = []
    seen_asins: set[str] = set()

    read = kept_quality = 0
    with open(PRODUCTS_CSV, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            read += 1
            if row["category_id"] not in keep_ids:
                continue
            if row["asin"] in seen_asins:
                continue
            try:
                price = float(row["price"] or 0)
                stars = float(row["stars"] or 0)
            except ValueError:
                continue
            if price <= 0 or stars <= 0:
                continue
            if not row["productURL"] or not row["imgUrl"]:
                continue
            title = (row["title"] or "").strip()
            if len(title) < MIN_TITLE_LEN:
                continue
            kept_quality += 1

            path = classify(title)
            is_monsoon = bool(MONSOON_PATTERN.search(title))

            take_for_path = path is not None and len(per_path[path]) < PER_PATH
            take_for_monsoon = is_monsoon and len(monsoon) < MONSOON_TARGET

            if not (take_for_path or take_for_monsoon):
                continue

            seen_asins.add(row["asin"])
            record = to_record(row, path or "Men's Apparel/T-Shirts", is_monsoon)
            if take_for_path:
                per_path[path].append(record) # type: ignore
            elif take_for_monsoon:
                # Monsoon-only samples still need a placeholder path; the
                # enrichment pass reassigns it from the real category words
                # in the title (jacket, boot, backpack, ...).
                monsoon.append(record)

    products = [r for bucket in per_path.values() for r in bucket] + monsoon

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(json.dumps(products, indent=2, ensure_ascii=False))

    print(f"read={read:,} passed_quality_filter={kept_quality:,}")
    print(f"sampled={len(products):,} -> {OUT_FILE}")
    print()
    for path, records in sorted(per_path.items()):
        if records:
            print(f"  {len(records):4d}  {path}")
    print(f"  {len(monsoon):4d}  (monsoon-relevant, cross-category)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
