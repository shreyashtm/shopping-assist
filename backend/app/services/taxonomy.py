"""The catalogue's canonical vocabulary.

Two axes, deliberately separated:

**Axis 1 - product type** (this module's `PRODUCT_TAXONOMY`). What a thing *is*.
**Axis 2 - use case / occasion** (a product's `attributes`). What it is *for*.

The previous taxonomy mixed them, and the cost was concrete: "Outdoor &
Trekking" is a use case, "Footwear" is a product type, so footwear ended up
split across three categories -- trekking boots under Outdoor, running shoes
under Footwear, mojaris under Ethnic Wear. Each product got exactly one home,
and any query crossing the axis failed. A request for formal shoes searched
`Footwear`, found only running shoes, and returned those.

Separating the axes means one product answers many intents:

    trekking boots        = Footwear/Boots            + use_case: trekking
    formal wedding shoes  = Footwear/Formal Shoes     + occasion: wedding
    a 50L rucksack        = Bags & Luggage/Backpacks  + use_case: trekking

The vocabulary is closed. Products are mapped onto it rather than labelled
freely, because free-text labels drift: the catalogue previously held `Shirts`
and `Tops` where the planner looked for `Casual Shirts` and `Tops & T-Shirts`,
and the mismatch was invisible until a slot silently found nothing.
"""

# Canonical product types. Categories are things, never occasions.
PRODUCT_TAXONOMY: dict[str, list[str]] = {
    "Men's Apparel": [
        "Casual Shirts", "Formal Shirts", "T-Shirts", "Trousers & Chinos",
        "Jeans", "Suits & Blazers", "Jackets & Coats", "Sweaters & Fleece",
        "Thermals & Base Layers", "Shorts",
    ],
    "Women's Apparel": [
        "Tops & T-Shirts", "Dresses", "Jeans", "Trousers", "Skirts",
        "Blazers", "Jackets & Coats", "Sweaters & Fleece",
    ],
    "Ethnic Wear": [
        "Kurta Sets", "Sherwanis", "Nehru Jackets", "Lehengas", "Sarees",
        "Ethnic Accessories",
    ],
    "Footwear": [
        "Formal Shoes", "Casual Sneakers", "Sports Shoes", "Boots",
        "Sandals & Floaters", "Heels", "Flats", "Ethnic Footwear",
    ],
    "Bags & Luggage": [
        "Backpacks", "Handbags & Clutches", "Luggage & Trolleys",
        "Duffels", "Wallets",
    ],
    "Watches & Jewellery": ["Watches", "Jewellery"],
    "Outdoor & Camping Gear": [
        "Camp & Sleep", "Navigation & Safety", "Trekking Equipment",
        "Outdoor Accessories",
    ],
    "Sports & Fitness": [
        "Yoga", "Strength Training", "Cardio", "Activewear", "Cycling",
    ],
    "Electronics & Accessories": [
        "Audio", "Wearables", "Charging & Power", "Mobile Accessories",
    ],
    "Home & Kitchen": ["Cookware", "Appliances", "Bedding", "Decor", "Storage"],
    "Beauty & Personal Care": [
        "Skincare", "Haircare", "Grooming", "Fragrance", "Makeup",
    ],
    "Gifting": [
        "Hampers", "Gourmet & Dry Fruits", "Keepsakes", "Home Fragrance",
    ],
}

# Axis 2. Closed so that retrieval can match on them reliably.
USE_CASES = [
    "trekking", "camping", "running", "gym", "yoga", "cycling", "travel",
    "office", "formal", "daily-wear", "party", "festive", "wedding",
    "layering", "gifting", "home", "grooming",
]

OCCASIONS = [
    "wedding", "anniversary", "birthday", "festive", "office", "interview",
    "party", "travel", "everyday",
]

# Below this a subcategory cannot offer a real choice -- one or two products is
# a listing, not a recommendation.
MIN_VIABLE_PER_SUBCATEGORY = 4

ALL_PATHS = [
    f"{category}/{sub}"
    for category, subs in PRODUCT_TAXONOMY.items()
    for sub in subs
]


def is_valid_path(path: str) -> bool:
    return path in ALL_PATHS


def split_path(path: str) -> tuple[str, str] | None:
    category, _, sub = path.partition("/")
    return (category, sub) if is_valid_path(path) else None
