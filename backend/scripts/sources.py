"""Catalogue source definitions.

The catalogue is designed from a **category taxonomy downward**, not from the
assignment's three example prompts. That distinction matters:

A catalogue built from the example queries would make the demo self-fulfilling.
If the only things in stock are trekking gear, ethnic wear and gift hampers,
then "interpreting intent" collapses into a three-way classifier, retrieval has
no way to be wrong, and the three prompts cannot function as a test of a system
that was derived from them.

So the taxonomy below spans a general Indian marketplace, and the example
prompts are held out as evaluation scenarios (see tests/eval_queries.py). Trek
gear, ethnic wear and gifting appear here because they are genuine parts of that
marketplace -- alongside categories that deliberately compete with them:

    running shoes      vs  trekking boots
    casual winter coat vs  insulated down jacket
    everyday kurta     vs  wedding sherwani
    laptop backpack    vs  50L trekking rucksack

Those near-misses are the point. They are what make ranking quality measurable
rather than automatic.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SourceQuery:
    """One search against one retailer, tagged with the taxonomy it feeds."""

    retailer: str
    query: str
    category: str
    subcategory: str


# --- The twelve top-level categories -------------------------------------
CATEGORIES = [
    "Men's Apparel",
    "Women's Apparel",
    "Ethnic Wear",
    "Footwear",
    "Outdoor & Trekking",
    "Sports & Fitness",
    "Electronics & Accessories",
    "Home & Kitchen",
    "Beauty & Personal Care",
    "Bags & Luggage",
    "Watches & Jewellery",
    "Gifting",
]

# Myntra: apparel, footwear, bags and watches, where its own taxonomy fields
# (masterCategory / category / subCategory / gender / season) are richer than
# anything the general marketplaces return.
MYNTRA_QUERIES = [
    # Ethnic Wear
    SourceQuery("Myntra", "men sherwani wedding", "Ethnic Wear", "Sherwanis"),
    SourceQuery("Myntra", "men kurta pyjama set festive", "Ethnic Wear", "Kurta Sets"),
    SourceQuery("Myntra", "women lehenga choli wedding", "Ethnic Wear", "Lehengas"),
    SourceQuery("Myntra", "women saree party wear", "Ethnic Wear", "Sarees"),
    SourceQuery("Myntra", "women anarkali kurta set festive", "Ethnic Wear", "Kurta Sets"),
    SourceQuery("Myntra", "men nehru jacket ethnic", "Ethnic Wear", "Nehru Jackets"),
    SourceQuery("Myntra", "men mojari juttis ethnic footwear", "Ethnic Wear", "Ethnic Footwear"),
    SourceQuery(
        "Myntra", "women juttis mojari ethnic footwear", "Ethnic Wear", "Ethnic Footwear"
    ),
    SourceQuery("Myntra", "women potli clutch bag ethnic", "Ethnic Wear", "Ethnic Accessories"),
    # Men's Apparel -- everyday clothing, and a casual winter jacket that
    # competes directly with the technical down jackets in Outdoor & Trekking.
    SourceQuery("Myntra", "men casual shirt cotton", "Men's Apparel", "Shirts"),
    SourceQuery("Myntra", "men formal trousers", "Men's Apparel", "Trousers"),
    SourceQuery("Myntra", "men casual jacket winter", "Men's Apparel", "Jackets"),
    # Women's Apparel
    SourceQuery("Myntra", "women casual top tshirt", "Women's Apparel", "Tops"),
    SourceQuery("Myntra", "women jeans denim", "Women's Apparel", "Jeans"),
    # Footwear -- running shoes are the deliberate near-miss for trekking boots.
    SourceQuery("Myntra", "men running sports shoes", "Footwear", "Sports Shoes"),
    # Bags & Luggage, Watches
    SourceQuery("Myntra", "women handbag shoulder bag", "Bags & Luggage", "Handbags"),
    SourceQuery("Myntra", "men analog watch", "Watches & Jewellery", "Watches"),
]

# Amazon.in: gear, equipment, electronics, home and gifting -- categories where
# the fashion retailers have little or no stock.
AMAZON_QUERIES = [
    # Outdoor & Trekking
    SourceQuery("Amazon.in", "trekking shoes men waterproof", "Outdoor & Trekking", "Footwear"),
    SourceQuery("Amazon.in", "down jacket winter men", "Outdoor & Trekking", "Insulation"),
    SourceQuery("Amazon.in", "thermal innerwear men winter", "Outdoor & Trekking", "Base Layers"),
    SourceQuery("Amazon.in", "fleece jacket men", "Outdoor & Trekking", "Mid Layers"),
    SourceQuery("Amazon.in", "winter gloves touchscreen", "Outdoor & Trekking", "Accessories"),
    SourceQuery("Amazon.in", "rucksack backpack 50L trekking", "Outdoor & Trekking", "Backpacks"),
    SourceQuery(
        "Amazon.in", "headlamp rechargeable camping", "Outdoor & Trekking", "Navigation & Safety"
    ),
    SourceQuery("Amazon.in", "sleeping bag winter camping", "Outdoor & Trekking", "Camp & Sleep"),
    SourceQuery("Amazon.in", "woolen socks winter thermal", "Outdoor & Trekking", "Accessories"),
    # Gifting
    SourceQuery("Amazon.in", "premium gift hamper", "Gifting", "Hampers"),
    SourceQuery("Amazon.in", "dry fruits gift box premium", "Gifting", "Gourmet & Dry Fruits"),
    SourceQuery("Amazon.in", "scented candle gift set", "Gifting", "Home Fragrance"),
    SourceQuery("Amazon.in", "silver plated gift anniversary", "Gifting", "Keepsakes"),
    SourceQuery("Amazon.in", "perfume gift set couple", "Gifting", "Fragrance"),
    # Electronics & Accessories
    SourceQuery(
        "Amazon.in", "bluetooth wireless headphones", "Electronics & Accessories", "Audio"
    ),
    SourceQuery(
        "Amazon.in", "smartwatch fitness tracker", "Electronics & Accessories", "Wearables"
    ),
    # Home & Kitchen
    SourceQuery("Amazon.in", "nonstick cookware set kitchen", "Home & Kitchen", "Cookware"),
    SourceQuery("Amazon.in", "cotton bedsheet double bed", "Home & Kitchen", "Bedding"),
    SourceQuery("Amazon.in", "mixer grinder juicer", "Home & Kitchen", "Appliances"),
    SourceQuery("Amazon.in", "insulated steel water bottle", "Home & Kitchen", "Drinkware"),
    # Sports & Fitness
    SourceQuery("Amazon.in", "yoga mat exercise", "Sports & Fitness", "Yoga"),
    SourceQuery("Amazon.in", "dumbbells home gym set", "Sports & Fitness", "Strength Training"),
    # Beauty & Personal Care
    SourceQuery("Amazon.in", "face wash men skincare", "Beauty & Personal Care", "Skincare"),
    # Bags & Luggage -- a laptop backpack competes with the trekking rucksacks.
    SourceQuery("Amazon.in", "laptop backpack office", "Bags & Luggage", "Backpacks"),
]

ALL_QUERIES = MYNTRA_QUERIES + AMAZON_QUERIES

ACTORS = {
    "Amazon.in": "fascinating_lentil/amazon-scraper",
    "Myntra": "codingfrontend/myntra-product-search-scraper",
}

# Ajio was evaluated and dropped. Its WAF returns 403 to both HEAD and GET from
# any datacentre IP -- the actor itself defaults to a residential proxy for that
# reason -- so an Ajio link could never be independently confirmed live, only
# trusted on provenance. Its queries were reassigned to Myntra, where liveness
# is checkable.

# Dataset IDs from the Apify runs that built the committed catalogue.
#
# The build replays these finished runs by default instead of re-running the
# actors. Scraping is billed per result, so a rebuild that re-scraped would cost
# real money and -- because retailer listings shift daily -- would also return a
# different catalogue, making the committed data irreproducible.
CAPTURED_DATASETS = {
    # Amazon bundles 5 search terms per run, in contiguous ~50-item blocks.
    "uoaYU8fZu3wGfQBEF": "Amazon.in",  # trekking apparel
    "vgyZkySGx4SrbAuOg": "Amazon.in",  # trekking equipment
    "aTYjjaFgkiUh54wre": "Amazon.in",  # gifting
    "UnIdJc31E7ssX4UOJ": "Amazon.in",  # electronics + home
    "Kk4tpji93hEFEQo6t": "Amazon.in",  # fitness, beauty, bags
    # Myntra runs one search term per dataset.
    "5HVApqXs2fYN3vBoZ": "Myntra",
    "7TJv7wlWKIUCA5OEG": "Myntra",
    "ccYQ6NqeV1kdiBcgL": "Myntra",
    "tIaw9GEDIBIVowKvt": "Myntra",
    "NIboKTzVWY1Sb6dDX": "Myntra",
    "Hw4cmPU2X8dIYcWDD": "Myntra",
    "V1w8xkG1GkMPGhdmt": "Myntra",
    "MWfwS8tTmFxcg0CEV": "Myntra",
    "TbNqhFBDA3zjCPRXl": "Myntra",
    "Ry6XOkOUZcCTTsoRk": "Myntra",
}

# Per search term, how many listings to keep.
#
# The two retailers bill differently, which changes the economics: Myntra honours
# its limit (ask 12, get 12, pay for 12), while Amazon ignores it and returns
# ~50 per keyword regardless. Since Amazon's 50 are paid for either way, keeping
# only 10 of them would waste two thirds of the spend -- hence the higher cap.
KEEP_PER_QUERY = {"Amazon.in": 25, "Myntra": 12}
DEFAULT_KEEP = 12
