"""Rule-based query interpretation for when no LLM is reachable.

This is the honest floor of the product, not a second implementation of it. It
maps keywords to categories and can ask fixed questions, but it cannot do the
thing the assistant is actually for: knowing that a Himalayan pass in late
October means sub-zero nights. Anything built on it is flagged `degraded_mode`
so weaker results are never passed off as reasoning.
"""

from app.schemas.query import (
    Bucket,
    ClarifyingQuestion,
    QueryFilters,
    QuestionOption,
    ResolvedContext,
    StructuredQuery,
)

# Keyword -> (bucket name, canonical catalogue paths, catalogue-language phrase).
#
# Paths are canonical rather than free text for the same reason the LLM path
# uses them: retrieval gates on exact Category/Subcategory, so a fallback that
# named categories loosely would gate everything out and return nothing --
# which would break the one guarantee this module exists to uphold.
_ROUTES: list[tuple[tuple[str, ...], str, list[str], str]] = [
    (
        ("trek", "hike", "hiking", "mountain", "camp", "trekking"),
        "Trekking Essentials",
        [
            "Men's Apparel/Jackets & Coats",
            "Men's Apparel/Thermals & Base Layers",
            "Men's Apparel/Sweaters & Fleece",
            "Footwear/Boots",
            "Bags & Luggage/Backpacks",
            "Outdoor & Camping Gear/Camp & Sleep",
            "Outdoor & Camping Gear/Navigation & Safety",
            "Outdoor & Camping Gear/Outdoor Accessories",
        ],
        "trekking gear for cold weather",
    ),
    (
        ("wedding", "sherwani", "saree", "lehenga", "kurta", "ethnic", "festive"),
        "Traditional Wear",
        [
            "Ethnic Wear/Sherwanis", "Ethnic Wear/Kurta Sets",
            "Ethnic Wear/Lehengas", "Ethnic Wear/Sarees",
            "Ethnic Wear/Nehru Jackets", "Footwear/Ethnic Footwear",
        ],
        "traditional Indian occasion wear",
    ),
    (
        ("gift", "hamper", "anniversary", "present", "gifting"),
        "Gift Ideas",
        [
            "Gifting/Hampers", "Gifting/Gourmet & Dry Fruits",
            "Gifting/Keepsakes", "Gifting/Home Fragrance",
            "Beauty & Personal Care/Fragrance",
        ],
        "premium gift set",
    ),
    (
        ("shirt", "tshirt", "t-shirt", "top", "jacket", "jeans", "trouser"),
        "Clothing",
        [
            "Men's Apparel/Casual Shirts", "Men's Apparel/T-Shirts",
            "Men's Apparel/Jackets & Coats", "Men's Apparel/Trousers & Chinos",
            "Women's Apparel/Tops & T-Shirts", "Women's Apparel/Jeans",
        ],
        "casual everyday clothing",
    ),
    (
        ("shoe", "sneaker", "footwear", "boot", "sandal"),
        "Footwear",
        [
            "Footwear/Sports Shoes", "Footwear/Casual Sneakers",
            "Footwear/Formal Shoes", "Footwear/Boots",
        ],
        "shoes",
    ),
    (("watch",), "Watches", ["Watches & Jewellery/Watches"], "wrist watch"),
    (
        ("gym", "workout", "fitness", "yoga", "dumbbell", "exercise"),
        "Fitness Gear",
        ["Sports & Fitness/Yoga", "Sports & Fitness/Strength Training"],
        "home workout equipment",
    ),
    (
        ("kitchen", "cookware", "bedsheet", "home", "mixer"),
        "Home & Kitchen",
        [
            "Home & Kitchen/Cookware", "Home & Kitchen/Appliances",
            "Home & Kitchen/Bedding",
        ],
        "kitchen and home essentials",
    ),
    (
        ("skincare", "face wash", "beauty", "grooming"),
        "Personal Care",
        ["Beauty & Personal Care/Skincare"],
        "skincare products",
    ),
    (
        ("bag", "backpack", "luggage", "handbag"),
        "Bags",
        ["Bags & Luggage/Backpacks", "Bags & Luggage/Handbags & Clutches"],
        "backpack",
    ),
    (
        ("headphone", "smartwatch", "earbud", "electronic", "fitness band"),
        "Electronics",
        [
            "Electronics & Accessories/Wearables",
            "Electronics & Accessories/Audio",
        ],
        "audio and wearable electronics",
    ),
]

# Fallback when nothing matched: search broadly rather than return an empty
# screen, which is this module's entire purpose.
_CATCH_ALL_PATHS = [
    "Men's Apparel/Casual Shirts", "Men's Apparel/Jackets & Coats",
    "Women's Apparel/Tops & T-Shirts", "Ethnic Wear/Kurta Sets",
    "Footwear/Sports Shoes", "Gifting/Hampers",
    "Home & Kitchen/Appliances", "Bags & Luggage/Backpacks",
]

_GENERIC_QUESTIONS = [
    ClarifyingQuestion(
        slot="budget",
        question="Roughly what budget?",
        options=[
            QuestionOption(label="Under Rs.500", value="price_max:500"),
            QuestionOption(label="Rs.500 - 1,500", value="price_min:500,price_max:1500"),
            QuestionOption(label="Rs.1,500 - 3,000", value="price_min:1500,price_max:3000"),
            QuestionOption(label="Premium", value="price_min:3000"),
        ],
    ),
    ClarifyingQuestion(
        slot="gender",
        question="Who's it for?",
        options=[
            QuestionOption(label="Men", value="gender:men"),
            QuestionOption(label="Women", value="gender:women"),
            QuestionOption(label="Doesn't matter", value="gender:unisex"),
        ],
    ),
]


def build_offline_query(query: str, answers: list[str]) -> StructuredQuery:
    text = query.lower()
    buckets: list[Bucket] = []
    categories: list[str] = []

    for keywords, bucket_name, paths, phrase in _ROUTES:
        if any(word in text for word in keywords):
            if bucket_name in {b.name for b in buckets}:
                continue
            categories.extend(p.split("/")[0] for p in paths)
            buckets.append(
                Bucket(
                    name=bucket_name,
                    search_phrases=[phrase, query],
                    why_needed=f"Matched on your mention of {bucket_name.lower()}.",
                    role="recommended",
                    catalogue_paths=paths,
                )
            )

    if not buckets:
        buckets = [
            Bucket(
                name="Suggestions",
                search_phrases=[query],
                why_needed="Closest matches to what you described.",
                role="recommended",
                catalogue_paths=_CATCH_ALL_PATHS,
            )
        ]

    # Ask only when the request is thin and nothing has been answered yet.
    thin = len(text.split()) < 10 and not answers
    return StructuredQuery(
        intent_summary=f"Looking for: {query.strip()}",
        buckets=buckets,
        # Categories are left unset: the per-slot paths already constrain
        # retrieval, and a global category filter would only narrow it further.
        filters=QueryFilters(),
        context=ResolvedContext(),
        assumptions=["Interpreted without AI reasoning, so this is a keyword match."],
        needs_clarification=thin,
        questions=_GENERIC_QUESTIONS if thin else [],
        confidence=0.3,
    )
