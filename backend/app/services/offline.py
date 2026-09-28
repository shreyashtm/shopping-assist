"""Rule-based query interpretation for when no LLM is reachable.

This is the honest floor of the product, not a second implementation of it. It
maps keywords to categories and can ask fixed questions, but it cannot do the
thing the assistant is actually for: knowing that a Himalayan pass in late
October means sub-zero nights. Anything built on it is flagged `degraded_mode`
so weaker results are never passed off as reasoning.
"""

import re

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
        ("trek", "hike", "hiking", "mountain", "camp", "trekking", "sleeping bag"),
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
    # Clothing is split by garment. One "Clothing" route searching every shelf
    # with "casual everyday clothing" answered "a warm jacket" with casual
    # shirts, because the phrase, not the request, decided the ranking.
    (
        ("jacket", "coat", "parka", "puffer"),
        "Jackets",
        ["Men's Apparel/Jackets & Coats", "Women's Apparel/Jackets & Coats"],
        "jacket",
    ),
    (
        ("t-shirt", "tshirt", "tee", "top"),
        "T-Shirts & Tops",
        ["Men's Apparel/T-Shirts", "Women's Apparel/Tops & T-Shirts"],
        "t-shirt",
    ),
    (
        ("shirt",),
        "Shirts",
        ["Men's Apparel/Casual Shirts", "Men's Apparel/Formal Shirts"],
        "shirt",
    ),
    (
        ("jeans", "denim"),
        "Jeans",
        ["Men's Apparel/Jeans", "Women's Apparel/Jeans"],
        "jeans",
    ),
    (
        ("trouser", "pant", "chino"),
        "Trousers",
        ["Men's Apparel/Trousers & Chinos", "Women's Apparel/Trousers"],
        "trousers",
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
    # Its own route rather than part of Bags: Bags searches with the phrase
    # "backpack", which ranked backpacks above every wallet for "a wallet".
    (("wallet",), "Wallets", ["Bags & Luggage/Wallets"], "leather wallet"),
    (
        ("gym", "workout", "fitness", "yoga", "dumbbell", "exercise"),
        "Fitness Gear",
        ["Sports & Fitness/Yoga", "Sports & Fitness/Strength Training"],
        "home workout equipment",
    ),
    (
        # No bare "home": it is a modifier far more often than a request
        # ("home office", "home workout"), and routed those to mixer grinders.
        ("kitchen", "cookware", "bedsheet", "mixer", "appliance", "home decor",
         "home essentials"),
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


_MODIFIER_PHRASES = (
    "top rated", "top-rated", "top quality", "top-quality", "top brand", "top notch",
    "top-notch", "top selling", "top-selling",
)


_CLAUSE_SPLIT = re.compile(r",|;|\band\b|\bwith\b|\bplus\b|\balso\b")
_LEADING_FILLER = re.compile(
    r"^\s*(i\s+(really\s+)?(need|want|would like|am looking for)|looking for|need|want"
    r"|get me|find me|show me|suggest)\s+"
)
_EDGE_STOPWORDS = {"for", "my", "to", "the", "in", "of", "on", "some", "me", "i", "your"}
_ARTICLES = {"a", "an", "the", "some"}


def _claim(text: str, pattern: str) -> str:
    """Blank out matches with same-length spaces, so offsets stay aligned."""
    return re.sub(pattern, lambda m: " " * len(m.group()), text)


def _what_was_asked(text: str, start: int) -> str:
    """The shopper's own words around a keyword match: "a warm jacket".

    Taken from the clause holding the match, a few words either side, so a
    group is searched and titled by what was asked for it -- not by the whole
    request, which pulled every group toward every item mentioned.
    """
    clause_start = 0
    for sep in _CLAUSE_SPLIT.finditer(text):
        if sep.start() >= start:
            clause_end = sep.start()
            break
        clause_start = sep.end()
    else:
        clause_end = len(text)
    before = _LEADING_FILLER.sub("", text[clause_start:start]).split()[-3:]
    after = text[start:clause_end].split()[:3]
    words = [w.strip(".!?'\"()") for w in before + after]
    while words and words[-1] in _EDGE_STOPWORDS | _ARTICLES:
        words.pop()
    while words and words[0] in _EDGE_STOPWORDS:
        words.pop(0)
    return " ".join(w for w in words if w)


def build_offline_query(query: str, answers: list[str]) -> StructuredQuery:
    text = query.lower()
    buckets: list[Bucket] = []
    categories: list[str] = []

    # A keyword containing a space or hyphen claims its words first, so
    # "fitness band" routes to Electronics without its "fitness" also opening
    # a dumbbell group, "sleeping bag" to trekking without "bag" opening a
    # backpack group, and "t-shirt" without its "shirt" opening Shirts.
    # Modifier phrases are claimed by no route at all: "top rated" and
    # "top quality" describe a product, they do not ask for a top.
    single_word_text = text
    for phrase in _MODIFIER_PHRASES:
        single_word_text = _claim(single_word_text, rf"\b{re.escape(phrase)}\b")
    for keywords, *_ in _ROUTES:
        for word in keywords:
            if " " in word or "-" in word:
                single_word_text = _claim(single_word_text, rf"\b{re.escape(word)}")

    matched = []
    for keywords, bucket_name, paths, phrase in _ROUTES:
        # Word-start match, not substring: "top" must not fire inside
        # "laptop", while "bag" still matches "bags" and "camp" "camping".
        starts = [
            m.start()
            for word in keywords
            for m in [re.search(
                rf"\b{re.escape(word)}",
                text if (" " in word or "-" in word) else single_word_text,
            )]
            if m
        ]
        if starts:
            matched.append((min(starts), bucket_name, paths, phrase))

    # A broad route gives up shelves a narrower matched route covers: asked for
    # "trekking gear" and "a warm jacket", jackets belong in Jackets, and
    # trekking gear should show the thermals and camp kit nobody named.
    narrowed = []
    for start, bucket_name, paths, phrase in matched:
        claimed = {
            p for _, other, other_paths, _ in matched
            if other != bucket_name and len(other_paths) < len(paths)
            for p in other_paths
        }
        kept = [p for p in paths if p not in claimed]
        narrowed.append((start, bucket_name, kept or paths, phrase))

    # In the order the shopper asked, not the order of this table.
    for start, bucket_name, paths, phrase in sorted(narrowed, key=lambda m: m[0]):
        asked = _what_was_asked(text, start)
        search = " ".join(w for w in asked.split() if w not in _ARTICLES) or phrase
        categories.extend(p.split("/")[0] for p in paths)
        buckets.append(
            Bucket(
                name=bucket_name,
                search_phrases=list(dict.fromkeys([phrase, search])),
                why_needed=f"You asked for {asked or phrase}.",
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
