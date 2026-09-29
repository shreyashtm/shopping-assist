"""Relevance acceptance tests: do the cards match what was asked for?

The other suites check that the pipeline is *consistent* -- every card comes
from a shelf the plan allowed, scores are sorted, nothing crashes. That is
circular when the plan itself is wrong: a keyword route that sent "a warm
jacket" to a Clothing bucket including casual shirts passed every one of them,
and the live app showed casual shirts for a jacket.

These tests judge the actual cards against what a shopper means, with the
expectations written by hand here -- not derived from the route table or any
other code under test. They run the full pipeline on the real catalogue in
keyword mode (no LLM), which is what shoppers get whenever the model is
unavailable.
"""

from datetime import date

import pytest

from app.adapters.llm.base import LLMUnavailable
from app.core.cache import response_cache
from app.schemas.recommend import RecommendRequest
from app.services.catalogue import Catalogue
from app.services.recommend import recommend


class _NoModel:
    name = "none"
    is_real = True

    def structured(self, **_):
        raise LLMUnavailable("relevance tests run without a model")


@pytest.fixture(scope="module")
def catalogue():
    return Catalogue.load()


def _cards(catalogue, query, answers=()):
    response_cache.clear()
    response = recommend(
        RecommendRequest(query=query, answers=list(answers), skip_clarification=True),
        catalogue, _NoModel(), today=date(2026, 9, 29),
    )
    return {g.name: g.items for g in response.groups}


JACKETS = {"Jackets & Coats"}
SHOES = {"Sports Shoes", "Casual Sneakers", "Formal Shoes", "Boots"}
WATCHES = {"Watches"}
BAGS = {"Backpacks", "Handbags & Clutches"}
GIFTS = {"Hampers", "Gourmet & Dry Fruits", "Keepsakes", "Home Fragrance", "Fragrance"}
TREK_GEAR = {
    "Jackets & Coats", "Thermals & Base Layers", "Sweaters & Fleece", "Boots",
    "Backpacks", "Camp & Sleep", "Navigation & Safety", "Outdoor Accessories",
}

KIT = ("I need trekking gear, a warm jacket, running shoes, a watch, "
       "a backpack and a gift hamper for my trip")


def _only(items, allowed, label):
    wrong = [(i.product.subcategory, i.product.title[:50]) for i in items
             if i.product.subcategory not in allowed]
    assert not wrong, f"{label}: cards outside {sorted(allowed)}: {wrong}"


def test_every_named_item_in_a_kit_gets_the_right_products(catalogue):
    """The request from the live report, item by item."""
    groups = _cards(catalogue, KIT)
    by_ask = {
        "a warm jacket": ("Jackets", JACKETS),
        "running shoes": ("Footwear", SHOES),
        "a watch": ("Watches", WATCHES),
        "a backpack": ("Bags", BAGS),
        "a gift hamper": ("Gift Ideas", GIFTS),
    }
    for asked, (group, allowed) in by_ask.items():
        assert group in groups, f"nothing shown for {asked!r}; groups: {list(groups)}"
        _only(groups[group], allowed, asked)
    _only(groups["Trekking Essentials"], TREK_GEAR, "trekking gear")


def test_running_shoes_are_running_shoes(catalogue):
    groups = _cards(catalogue, KIT)
    titles = [i.product.title.lower() for i in groups["Footwear"]]
    assert all("run" in t or "sport" in t or "trainer" in t for t in titles), titles


def test_trekking_gear_does_not_repeat_the_jacket_group(catalogue):
    groups = _cards(catalogue, KIT)
    assert not any(i.product.subcategory == "Jackets & Coats" for i in groups["Trekking Essentials"])


def test_no_card_is_explained_by_a_different_item_in_the_request(catalogue):
    """Watches read "Made for gifting" when every group searched the whole
    sentence and the sentence mentioned a gift hamper."""
    groups = _cards(catalogue, KIT)
    for name, items in groups.items():
        if name == "Gift Ideas":
            continue
        for item in items:
            assert "gifting" not in item.reason.lower(), (name, item.product.title, item.reason)


@pytest.mark.parametrize("query, allowed", [
    ("a warm jacket for winter", JACKETS),
    ("I need a bag or backpack for carrying my laptop and books every day to office", BAGS),
    ("top rated laptop backpack for office", BAGS),
    ("a sleeping bag for camping", {"Camp & Sleep", "Navigation & Safety", "Outdoor Accessories",
                                    "Thermals & Base Layers", "Sweaters & Fleece", "Jackets & Coats",
                                    "Boots", "Backpacks"}),
    ("home workout dumbbells and a yoga mat", {"Strength Training", "Yoga", "Cardio"}),
    ("wireless headphones and a fitness band", {"Wearables", "Audio"}),
    ("a watch and a wallet for my dad", WATCHES | {"Wallets"}),
    ("premium gift hamper for my parents anniversary", GIFTS),
    ("formal shirt for office", {"Formal Shirts", "Casual Shirts"}),
    ("slim fit jeans", {"Jeans"}),
])
def test_single_requests_only_show_what_was_asked(catalogue, query, allowed):
    groups = _cards(catalogue, query, ["gender:men"])
    assert groups, f"no results for {query!r}"
    for name, items in groups.items():
        _only(items, allowed, f"{query!r} / {name}")


def test_a_sleeping_bag_request_leads_with_sleeping_bags(catalogue):
    groups = _cards(catalogue, "a sleeping bag for camping")
    first = next(iter(groups.values()))[0]
    assert "sleeping bag" in first.product.title.lower(), first.product.title


def test_an_occasion_in_the_request_applies_to_the_item_asked_for(catalogue):
    """Live: a "Bhaiya Bhabhi" (brother and sister-in-law) hamper, tagged
    festive/wedding/everyday, was the best match for a parents' 25th
    anniversary once groups stopped seeing the rest of the request."""
    groups = _cards(catalogue, "a gift hamper for my parents' 25th anniversary")
    for items in groups.values():
        for item in items:
            occasions = item.product.attributes.occasion
            assert not occasions or "anniversary" in occasions, (item.product.title, occasions)


def test_no_reason_cites_a_material_that_says_nothing():
    """Live: "Made for gifting; built with assorted." -- a catalogue value
    that names no material, cited to the shopper as if it did."""
    from app.schemas.product import Product
    from app.schemas.query import Bucket, ResolvedContext
    from app.services.retrieval import score_product

    bucket = Bucket(name="Gifts", search_phrases=["gift box"], why_needed="x",
                    role="required", catalogue_paths=["Gifting/Hampers"])
    for material in ("Assorted", "mixed", "Various", "Leather"):
        product = Product(
            id="p", title="Gift Box", brand="B", category="Gifting", subcategory="Hampers",
            price_inr=1000, description="d", retailer="Amazon.in",
            product_url="https://example.com/p", attributes={"material": material},
        )
        reasons = score_product(product, 0.5, bucket, ResolvedContext()).reasons
        cited = any(r.startswith("built with") for r in reasons)
        assert cited == (material == "Leather"), (material, reasons)


def test_a_purpose_after_for_does_not_open_a_second_group(catalogue):
    """Live: "running shoes for the gym and morning jogs" showed four
    dumbbell sets under "You asked for shoes for the gym"."""
    groups = _cards(catalogue, "running shoes for the gym and morning jogs", ["gender:men"])
    assert list(groups) == ["Footwear"], list(groups)


def test_a_purpose_still_counts_when_it_is_the_request(catalogue):
    """"gym equipment" and "a bag and gym gloves" name gym gear as an item."""
    assert "Fitness Gear" in _cards(catalogue, "home gym equipment and dumbbells")


def test_casual_garments_are_never_tagged_formal(catalogue):
    """Live: a typography-print tee tagged formality=formal was shown as
    "Matches the occasion's formality". A T-shirt, top, pair of shorts or
    activewear piece is never formal wear."""
    casual_types = {"T-Shirts", "Tops & T-Shirts", "Shorts", "Activewear"}
    wrong = [(p.subcategory, p.title[:50]) for p in catalogue.products
             if p.subcategory in casual_types and p.attributes.formality == "formal"]
    assert not wrong, wrong


def test_a_budget_range_shows_in_range_picks_when_good_ones_exist(catalogue):
    """Live: Rs 1,500-3,000 running shoes showed four shoes under Rs 1,100
    while two verified in-range running shoes were near-ties."""
    groups = _cards(catalogue, "running shoes for the gym and morning jogs",
                    ["gender:men", "price_min:1500,price_max:3000"])
    shoes = groups["Footwear"]
    in_range = [i for i in shoes if 1500 <= i.product.price_inr <= 3000]
    assert in_range, [(i.product.title[:40], i.product.price_inr) for i in shoes]
    assert all(i.product.subcategory == "Sports Shoes" for i in in_range), "boots must not be promoted"


# Live: "wool socks for winter" showed jackets and running shoes, because
# "socks" and ~25 other everyday product words had no keyword route and fell
# through to generic suggestions.
@pytest.mark.parametrize("query, allowed", [
    ("wool socks for winter", {"Socks & Hosiery"}),
    ("thermals for a cold trip", {"Thermals & Base Layers"}),
    ("a warm sweater or hoodie", {"Sweaters & Fleece"}),
    ("a summer dress", {"Dresses"}),
    ("a pleated skirt", {"Skirts"}),
    ("ballerina flats for office", {"Flats"}),
    ("comfortable slippers for home", {"Sandals & Floaters"}),
    ("a perfume for him", {"Fragrance"}),
    ("winter gloves", {"Outdoor Accessories"}),
    ("cotton shorts for summer", {"Shorts"}),
    ("a navy blazer for work", {"Suits & Blazers", "Blazers"}),
    ("a tracksuit for the gym", {"Activewear"}),
    ("a cabin suitcase for travel", {"Luggage & Trolleys", "Duffels"}),
    ("a suitcase for my trip", {"Luggage & Trolleys", "Duffels"}),
    ("I need a suit for my interview", {"Suits & Blazers", "Blazers"}),
    ("formal dress shirt", {"Formal Shirts", "Casual Shirts"}),
])
def test_everyday_product_words_route_to_their_shelf(catalogue, query, allowed):
    groups = _cards(catalogue, query)
    assert "Suggestions" not in groups, f"{query!r} fell through to generic suggestions"
    for name, items in groups.items():
        _only(items, allowed, f"{query!r} / {name}")


def test_new_routes_do_not_fire_on_lookalike_words():
    from app.services.offline import build_offline_query
    for query, unwanted in (("what is the capital of France", "Accessories"),
                            ("home decor for my new flat", "Flats"),
                            ("a fleece lined winter jacket", "Sweaters & Fleece"),
                            ("thermal socks for trekking", "Thermals"),
                            ("something dressy for a party", "Dresses"),
                            ("a suitcase for my trip", "Suits & Blazers")):
        names = [b.name for b in build_offline_query(query, []).buckets]
        assert unwanted not in names, (query, names)


def test_an_anniversary_hamper_request_shows_real_hampers(catalogue):
    """Live: no hamper was tagged for anniversaries, so a request for an
    anniversary hamper showed only perfumes and a keepsake."""
    groups = _cards(catalogue, "a gift hamper for my parents' 25th anniversary")
    shelves = {i.product.subcategory for items in groups.values() for i in items}
    assert shelves & {"Hampers", "Gourmet & Dry Fruits"}, shelves
    titles = " ".join(i.product.title.lower() for items in groups.values() for i in items)
    assert "bhaiya" not in titles and "pureheart" not in titles


# Live, with the model path: "I'm a man going on a trek near Leh from 20 to
# 27 December, need thermals and warm socks, budget 3000" returned "Nothing in
# the catalogue fits". One cause (fixed in 85b98d9) was shelf spelling. The
# other: the model's global filters.categories listed only the thermals shelf,
# and that entry was kept for the socks group too (both sit under "Men's
# Apparel"), so socks came back empty while dozens were in budget.


def _plan_for(paths_by_bucket, categories, price_max=None, gender=None):
    class _Plan:
        name = "plan"
        is_real = True

        def structured(self, **_):
            return {
                "intent_summary": "Planned.", "is_shopping_request": True,
                "buckets": [{"name": name, "search_phrases": [name.lower()], "why_needed": "x",
                             "role": "required", "catalogue_paths": paths}
                            for name, paths in paths_by_bucket.items()],
                "filters": {"price_max": price_max, "gender": gender, "categories": categories},
                "context": {}, "assumptions": [],
            }
    return _Plan()


def test_a_category_filter_naming_one_shelf_does_not_empty_another_group(catalogue):
    response_cache.clear()
    response = recommend(
        RecommendRequest(query="thermals and warm socks for a trek", skip_clarification=True),
        catalogue,
        _plan_for({"Thermals": ["Men's Apparel/Thermals & Base Layers"],
                   "Warm Socks": ["Men's Apparel/Socks & Hosiery"]},
                  ["Men's Apparel/Thermals & Base Layers"], price_max=3000, gender="men"),
        today=date(2026, 9, 29),
    )
    assert {g.name for g in response.groups} == {"Thermals", "Warm Socks"}


def test_every_stocked_shelf_returns_results_whatever_else_the_filter_names(catalogue):
    """The general guard against false "Nothing in the catalogue fits": a
    group planned on a shelf that has products must return some, however the
    model filled the global category filter."""
    from app.services.taxonomy import ALL_PATHS

    stocked = sorted({f"{p.category}/{p.subcategory}" for p in catalogue.products} & set(ALL_PATHS))
    empty = []
    for path in stocked:
        other = next(p for p in stocked if p.split("/")[0] == path.split("/")[0] and p != path) \
            if sum(p.split("/")[0] == path.split("/")[0] for p in stocked) > 1 else "Footwear/Boots"
        response_cache.clear()
        response = recommend(
            RecommendRequest(query=f"something from {path}", skip_clarification=True),
            catalogue, _plan_for({"Need": [path]}, [other]), today=date(2026, 9, 29),
        )
        if not response.groups:
            empty.append((path, other))
    assert not empty, empty


# The two requests from the live frontend check, in keyword mode.


def test_leh_request_respects_what_it_states(catalogue):
    groups = _cards(catalogue, "I'm a man going on a trek near Leh from 20 to 27 December, "
                               "need thermals and warm socks, budget 3000")
    assert list(groups) == ["Thermals & Base Layers", "Socks"]
    for items in groups.values():
        for item in items:
            assert item.product.attributes.gender in ("men", "unisex"), item.product.title
            assert item.product.price_inr <= 3000, (item.product.title, item.product.price_inr)


def test_hampta_request_shows_only_what_was_named(catalogue):
    groups = _cards(catalogue, "I'm a man trekking Hampta Pass the last week of October for "
                               "a week, need a warm jacket and trekking shoes")
    assert list(groups) == ["Jackets", "Footwear"]
    _only(groups["Jackets"], JACKETS, "warm jacket")
    for item in groups["Jackets"] + groups["Footwear"]:
        assert item.product.attributes.gender in ("men", "unisex"), item.product.title


def test_a_group_offers_a_real_choice_even_when_one_item_is_planned(catalogue):
    """Live, with the model path: "women's jeans and a top for college" was
    planned with max_items=1 per group ("a top" read as a quantity), and the
    shopper got a single jeans and a single top to choose from."""
    plan = _plan_for({"Jeans": ["Women's Apparel/Jeans"], "Top": ["Women's Apparel/Tops & T-Shirts"]},
                     [], price_max=1500, gender="women")
    original = plan.structured

    def one_each(**kw):
        payload = original(**kw)
        for bucket in payload["buckets"]:
            bucket["max_items"] = 1
        return payload

    plan.structured = one_each
    response_cache.clear()
    response = recommend(RecommendRequest(query="women's jeans and a top for college",
                                          skip_clarification=True),
                         catalogue, plan, today=date(2026, 9, 29))
    assert all(len(g.items) >= 3 for g in response.groups), [(g.name, len(g.items)) for g in response.groups]


def test_the_requests_occasion_applies_to_every_group_on_the_model_path(catalogue):
    """Live, with the model path: "a birthday gift for my wife, she loves
    perfumes and skincare" was planned with a "Beauty Gift Hampers" group whose
    own text never said "birthday", so the occasion check never ran for it and
    a "Bhaiya Bhabhi" (brother and sister-in-law) hamper was shown. The
    perfume group came back empty: every unisex perfume gift set was tagged
    only for anniversaries and weddings. This mirrors the logged plan."""
    class _LoggedPlan:
        name = "plan"
        is_real = True

        def structured(self, **_):
            def b(name, path, why):
                return {"name": name, "search_phrases": [name.lower()], "why_needed": why,
                        "role": "recommended", "catalogue_paths": [path], "max_items": 3}
            return {
                "intent_summary": "A gift for a wife who loves perfumes and skincare.",
                "is_shopping_request": True,
                "buckets": [b("Fragrance & Perfumes", "Beauty & Personal Care/Fragrance", "She loves perfumes."),
                            b("Beauty Gift Hampers", "Gifting/Hampers", "A pampering set for her.")],
                "filters": {"price_max": 10000, "gender": "women",
                            "categories": ["Beauty & Personal Care", "Gifting"]},
                "context": {}, "assumptions": [],
            }

    response_cache.clear()
    response = recommend(
        RecommendRequest(query="a birthday gift for my wife, she loves perfumes and skincare",
                         skip_clarification=True),
        catalogue, _LoggedPlan(), today=date(2026, 9, 29))
    groups = {g.name: g.items for g in response.groups}
    titles = [i.product.title for items in groups.values() for i in items]
    assert not any("Bhaiya" in t for t in titles), titles
    assert groups.get("Fragrance & Perfumes"), list(groups)


# The three issues from the frontend scenario run.

MANALI_KIT = ("I'm a man going to Manali next week, need a jacket, thermals, gloves, "
              "socks, trekking shoes and a backpack")


def _response(catalogue, query, provider=None, **kw):
    response_cache.clear()
    return recommend(RecommendRequest(query=query, **kw), catalogue, provider or _NoModel(),
                     today=date(2026, 9, 29))


def test_named_items_show_products_even_when_a_question_is_asked(catalogue):
    """Live: a request naming six items got only "Roughly what budget?"."""
    for query in (MANALI_KIT, "a birthday gift for my wife, she loves perfumes and skincare"):
        response = _response(catalogue, query)
        assert response.groups, query


def test_a_decline_is_marked_so_the_page_does_not_suggest_widening_a_budget(catalogue):
    class _Decline:
        name = "d"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "Geography.", "is_shopping_request": False, "buckets": [],
                    "filters": {}, "context": {}, "assumptions": []}

    response = _response(catalogue, "what is the capital of France", _Decline(), skip_clarification=True)
    assert response.declined is True
    assert _response(catalogue, "wool socks for winter", skip_clarification=True).declined is False


def test_dates_the_shopper_never_gave_are_not_shown_as_theirs(catalogue):
    """Live: "women's high heels and a blazer for an office party" showed
    "Dates · 2026-09-29 (you)" -- the model had set a one-day trip on today."""
    class _InventsADate:
        name = "i"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "Heels and a blazer for an office party.", "is_shopping_request": True,
                    "buckets": [{"name": "Heels", "search_phrases": ["heels"], "why_needed": "x",
                                 "role": "required", "catalogue_paths": ["Footwear/Flats"]}],
                    "filters": {"gender": "women"},
                    "context": {"start_date": "2026-09-29", "end_date": "2026-09-29", "duration_days": 1},
                    "assumptions": []}

    response = _response(catalogue, "I need women's high heels and a blazer for an office party, budget 3000",
                         _InventsADate())
    assert not any(v.name == "dates" for v in response.context_variables)


def _invents_budget(price_min, price_max):
    class _Model:
        name = "i"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "Warm socks.", "is_shopping_request": True,
                    "buckets": [{"name": "Socks", "search_phrases": ["wool socks"], "why_needed": "x",
                                 "role": "required", "catalogue_paths": ["Men's Apparel/Socks & Hosiery"]}],
                    "filters": {"price_min": price_min, "price_max": price_max},
                    "assumptions": []}
    return _Model()


def _budget(response):
    return next((v.value, v.status) for v in response.context_variables if v.name == "budget")


def test_a_budget_the_shopper_never_gave_is_not_applied(catalogue):
    """Live: qwen3:8b gave a Hampta Pass trek that named no budget a
    ₹2,000-40,000 window, and "a hamper under 5000" a ₹2,500 floor."""
    model = _invents_budget(2000, 40000)
    unstated = _response(catalogue, "wool socks for a trek in October", model, skip_clarification=True)
    assert _budget(unstated) == (None, "needed")
    ceiling = _response(catalogue, "wool socks under 40000", model, skip_clarification=True)
    assert _budget(ceiling) == ("Under ₹40,000", "known")


def test_the_shoppers_amount_wins_over_the_models(catalogue):
    """Live, qwen3:8b: "suit and formal shoes ... budget 8000" came back as
    ₹2,320-12,487, the suit shelf's price range copied from the prompt."""
    response = _response(catalogue, "wool socks, budget 8000", _invents_budget(2320, 12487),
                         skip_clarification=True)
    assert _budget(response) == ("Under ₹8,000", "known")


def test_a_budget_the_shopper_did_give_is_kept(catalogue):
    response = _response(catalogue, "wool socks, budget 2000 to 40000", _invents_budget(2000, 40000),
                         skip_clarification=True)
    assert _budget(response) == ("₹2,000 – ₹40,000", "known")


def test_a_stated_occasion_place_and_recipient_are_not_marked_needed(catalogue):
    cases = {
        "women's high heels and a blazer for an office party": "occasion",
        "women's jeans and a top for college": "occasion",
        MANALI_KIT: "location",
        "a birthday gift for my wife, she loves perfumes": "recipient",
    }
    for query, slot in cases.items():
        needed = {v.name for v in _response(catalogue, query).context_variables if v.status == "needed"}
        assert slot not in needed, (query, needed)


def _fills_in(location, gender):
    class _Model:
        name = "i"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "Warm clothes.", "is_shopping_request": True,
                    "buckets": [{"name": "Thermals", "search_phrases": ["thermal"], "why_needed": "x",
                                 "role": "required", "catalogue_paths": ["Men's Apparel/Thermals & Base Layers"]}],
                    "filters": {"gender": gender},
                    "context": {"location": location,
                                "climate_note": "Winter in India typically ranges from 5C to 15C."},
                    "assumptions": []}
    return _Model()


def test_a_place_and_wearer_the_shopper_never_gave_are_not_shown(catalogue):
    """Live, qwen3:8b: "Place · India (inferred)" with "add a date for India"
    for "something warm for winter", "N/A (no location provided)" as a page
    heading, and "For · unisex (you)" though nobody said so."""
    for location, gender in (("India", "unisex"), ("N/A (no location provided)", "unspecified")):
        response = _response(catalogue, "something warm for winter", _fills_in(location, gender),
                             skip_clarification=True)
        shown = {v.name: (v.value, v.status) for v in response.context_variables}
        assert response.context.location is None, location
        assert response.context.climate_note is None, "a model-written climate with no place behind it"
        assert shown.get("gender", (None, None))[0] != gender, gender


def test_a_place_and_wearer_the_shopper_did_give_are_kept(catalogue):
    response = _response(catalogue, "thermals for men for my Manali trip",
                         _fills_in("Manali, Himachal Pradesh", "men"), skip_clarification=True)
    assert response.context.location == "Manali, Himachal Pradesh"


def test_a_trip_month_already_past_is_next_years(catalogue):
    """Live, qwen3:8b on 29 Sep 2026: "Leh for 10 days in January" became
    1-10 January 2026, and the page showed 3C nights for Leh in winter."""
    class _PastDate:
        name = "i"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "Leh in January.", "is_shopping_request": True,
                    "buckets": [{"name": "Thermals", "search_phrases": ["thermal"], "why_needed": "x",
                                 "role": "required", "catalogue_paths": ["Men's Apparel/Thermals & Base Layers"]}],
                    "context": {"location": "Leh, Ladakh", "start_date": "2026-01-01",
                                "end_date": "2026-01-10", "duration_days": 10},
                    "assumptions": []}

    response = _response(catalogue, "Going to Leh for 10 days in January", _PastDate(), skip_clarification=True)
    assert str(response.context.start_date) == "2027-01-01"
    assert str(response.context.end_date) == "2027-01-10"


def test_a_trip_length_is_not_a_budget(catalogue):
    """Live: "10 days" counted as a stated price, so the model's invented
    ₹1,000-20,000 was shown as the shopper's own budget."""
    response = _response(catalogue, "wool socks for 10 days in January", _invents_budget(1000, 20000),
                         skip_clarification=True)
    assert _budget(response) == (None, "needed")


def test_a_gendered_recipient_is_not_asked_their_gender(catalogue):
    """Live: "a birthday gift for my 10 year old nephew" was asked "What
    gender is your nephew?"."""
    class _AsksGender:
        name = "i"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "A birthday gift for a nephew.", "is_shopping_request": True,
                    "buckets": [{"name": "Gifts", "search_phrases": ["gift"], "why_needed": "x",
                                 "role": "required", "catalogue_paths": ["Gifting/Hampers"]}],
                    "context": {"recipient": "nephew"},
                    "needs_clarification": True,
                    "questions": [{"slot": "gender", "question": "What gender is your nephew?", "options": [
                        {"label": "Boy", "value": "gender:men"}, {"label": "Girl", "value": "gender:women"}]}],
                    "assumptions": []}

    response = _response(catalogue, "birthday gift for my 10 year old nephew", _AsksGender())
    asked = [q.question for q in (response.questions or [])]
    assert not any("gender" in q.lower() for q in asked), asked
    shown = {v.name: v.value for v in response.context_variables}
    assert shown.get("gender") != "unspecified"


def test_a_wearer_is_kept_only_when_the_words_imply_one(catalogue):
    """Live: "traditional wear for my friend's wedding" showed "For · men
    (you)" while the same plan offered lehengas and sarees."""
    cases = {
        "traditional wear for my friend's wedding in March": None,
        "a saree for my friend's wedding": "women",
        "I'm a man trekking Hampta Pass": "men",
        "thermals for my dad": "men",
    }
    for query, expected in cases.items():
        response = _response(catalogue, query, _fills_in("Hampta Pass", "women" if expected == "women" else "men"),
                             skip_clarification=True)
        shown = {v.name: v.value for v in response.context_variables}
        assert shown.get("gender") == expected, (query, shown.get("gender"))


def test_premium_is_a_preference_not_a_price(catalogue):
    """Live: "a premium gifting hamper" showed "Budget · ₹1,299 – ₹5,299
    (you)", the hamper shelf's range copied from the prompt."""
    response = _response(catalogue, "a premium gifting hamper for my parents' 25th anniversary",
                         _invents_budget(1299, 5299), skip_clarification=True)
    assert _budget(response) == (None, "needed")


def test_a_couple_is_not_asked_for_one_gender(catalogue):
    """Live: "my parents' 25th anniversary" was asked "Who is this for? Men /
    Women"."""
    class _Couple:
        name = "i"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "An anniversary hamper.", "is_shopping_request": True,
                    "buckets": [{"name": "Hampers", "search_phrases": ["hamper"], "why_needed": "x",
                                 "role": "required", "catalogue_paths": ["Gifting/Hampers"]}],
                    "context": {"recipient": "parents"},
                    "assumptions": []}

    response = _response(catalogue, "a hamper for my parents' anniversary", _Couple())
    assert not any(v.name == "gender" and v.status == "needed" for v in response.context_variables)
    assert not any("who is this for" in q.question.lower() for q in (response.questions or []))


def _plans(paths_by_group, elevation=None):
    class _Model:
        name = "i"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "A trip.", "is_shopping_request": True,
                    "buckets": [{"name": name, "search_phrases": [name.lower()], "why_needed": "x",
                                 "role": "recommended", "catalogue_paths": paths}
                                for name, paths in paths_by_group.items()],
                    "context": {"elevation_estimate_m": elevation},
                    "assumptions": []}
    return _Model()


TRIP_PLAN = {
    "Warm Coat": ["Men's Apparel/Jackets & Coats"],
    "Navigation & Safety": ["Outdoor & Camping Gear/Navigation & Safety"],
    "Evening Wear": ["Ethnic Wear/Kurta Sets", "Men's Apparel/Formal Shirts"],
}


def test_trek_and_ethnic_shelves_only_when_the_request_calls_for_them(catalogue):
    """Live, qwen3:8b: headlamps for "Europe in December" and kurta sets for
    a Goa beach holiday -- a trek-shaped plan copied onto every trip."""
    def plan(query, elevation=None, answers=()):
        response = _response(catalogue, query, _plans(TRIP_PLAN, elevation),
                             skip_clarification=True, answers=list(answers))
        names = {g.name for g in response.groups}
        shelves = {f"{i.product.category}/{i.product.subcategory}" for g in response.groups for i in g.items}
        return names, shelves

    names, shelves = plan("going to Europe in December for two weeks")
    assert "Navigation & Safety" not in names
    assert not any(s.startswith("Ethnic Wear/") for s in shelves)

    names, _ = plan("trekking Hampta Pass in October")
    assert "Navigation & Safety" in names
    names, _ = plan("Going to Leh for 10 days in January", elevation=3500)
    assert "Navigation & Safety" in names, "a high destination calls for it without the word trek"
    _, shelves = plan("clothes for my friend's wedding")
    assert any(s.startswith("Ethnic Wear/") for s in shelves)
    _, shelves = plan("clothes for an event next month", answers=["occasion:wedding"])
    assert any(s.startswith("Ethnic Wear/") for s in shelves)


def test_hinglish_budget_and_relations_are_understood(catalogue):
    """Live: "bhai ki shaadi ke liye kurta chahiye, 3000 tak" showed "Budget ·
    ₹3,000 – ₹3,000" and asked "Who is this for?" about a brother."""
    class _Kurta:
        name = "i"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "A kurta for a brother's wedding.", "is_shopping_request": True,
                    "buckets": [{"name": "Kurta Set", "search_phrases": ["kurta set"], "why_needed": "x",
                                 "role": "required", "catalogue_paths": ["Ethnic Wear/Kurta Sets"]}],
                    "filters": {"price_min": 3000, "price_max": 3000},
                    "context": {"recipient": "bhai"},
                    "assumptions": []}

    response = _response(catalogue, "bhai ki shaadi ke liye kurta chahiye, 3000 tak", _Kurta())
    assert _budget(response) == ("Under ₹3,000", "known")
    assert not any(v.name == "gender" and v.status == "needed" for v in response.context_variables)


def test_the_named_month_wins_over_the_models_dates(catalogue):
    """Live, qwen3:8b on 29 Sep 2026: "Shimla in January" became 29 Sep - 31
    Dec 2026, a 90-day trip starting today, shown as the shopper's dates."""
    class _WrongMonth:
        name = "i"
        is_real = True

        def structured(self, **_):
            return {"intent_summary": "A jacket for Shimla.", "is_shopping_request": True,
                    "buckets": [{"name": "Jacket", "search_phrases": ["winter jacket"], "why_needed": "x",
                                 "role": "required", "catalogue_paths": ["Men's Apparel/Jackets & Coats"]}],
                    "context": {"location": "Shimla, Himachal Pradesh", "start_date": "2026-09-29",
                                "end_date": "2026-12-31", "duration_days": 90},
                    "assumptions": []}

    response = _response(catalogue, "women's winter jacket for Shimla in January", _WrongMonth(),
                         skip_clarification=True)
    assert (str(response.context.start_date), str(response.context.end_date)) == ("2027-01-01", "2027-01-31")


def test_may_the_verb_is_not_a_month():
    from app.services.context_slots import stated_month

    assert stated_month("I may need a jacket") is None
    assert stated_month("a trip in May") == 5
    assert stated_month("Leh for 10 days in January") == 1
