"""Keyword-routing tests for the no-LLM fallback interpreter.

Found in a live offline session: "I need a bag or backpack for carrying my
laptop and books every day to office" returned a Clothing group of shirts and
trousers above the backpacks, because the Clothing route's keyword "top" was
matched as a substring of "laptop".
"""

from app.services.offline import build_offline_query


def _bucket_names(query: str) -> list[str]:
    return [b.name for b in build_offline_query(query, answers=[]).buckets]


def test_keyword_inside_another_word_does_not_route():
    assert _bucket_names(
        "I need a bag or backpack for carrying my laptop and books every day to office"
    ) == ["Bags"]


def test_keyword_still_matches_its_plural_and_longer_forms():
    assert "T-Shirts & Tops" in _bucket_names("some nice tops for summer")
    assert "Bags" in _bucket_names("looking at bags")
    assert "Trekking Essentials" in _bucket_names("going camping next week")


def test_multi_word_keyword_still_matches():
    assert "Personal Care" in _bucket_names("a gentle face wash")


# Found by a 26-input keyword-mode battery: whole-word collisions and a
# missing keyword, not substring matches.


def test_home_as_a_modifier_does_not_route_to_kitchen():
    assert "Home & Kitchen" not in _bucket_names("desktop stand for my home office")
    assert _bucket_names("home workout dumbbells and a yoga mat") == ["Fitness Gear"]


def test_kitchen_requests_still_route_to_home_and_kitchen():
    assert _bucket_names("mixer grinder and cookware for my new kitchen") == ["Home & Kitchen"]


def test_a_multi_word_keyword_claims_its_words():
    """"fitness band" is a wearable; its "fitness" must not also open a
    dumbbell group."""
    assert _bucket_names("wireless headphones and a fitness band") == ["Electronics"]


def test_wallet_gets_its_own_route():
    names = _bucket_names("a watch and a wallet and a belt for my dad")
    assert names == ["Watches", "Wallets"]


def test_home_appliances_still_route_without_the_bare_home_keyword():
    assert _bucket_names("some home appliances") == ["Home & Kitchen"]


# From the code review of today's routing fixes.


def test_top_as_a_modifier_does_not_route_to_clothing():
    assert _bucket_names("top rated laptop backpack for office") == ["Bags"]
    assert _bucket_names("top quality running shoes") == ["Footwear"]
    assert _bucket_names("a top and jeans for college") == ["T-Shirts & Tops", "Jeans"]


def test_sleeping_bag_is_camping_gear_not_a_bag():
    assert _bucket_names("a sleeping bag for camping") == ["Trekking Essentials"]


def test_home_decor_still_routes_to_home_and_kitchen():
    assert _bucket_names("home decor for my new flat") == ["Home & Kitchen"]
    assert _bucket_names("home essentials for a new flat") == ["Home & Kitchen"]



# Reported from the live app: "I need trekking gear, a warm jacket, running
# shoes, a watch, a backpack and a gift hamper for my trip" showed casual
# shirts for "a warm jacket", Gift Ideas second, and every group searching
# with the whole sentence (so watches read "Made for gifting").

KIT = ("I need trekking gear, a warm jacket, running shoes, a watch, "
       "a backpack and a gift hamper for my trip")


def test_a_jacket_request_searches_jackets_not_casual_shirts():
    jackets = next(b for b in build_offline_query(KIT, []).buckets if b.name == "Jackets")
    assert all("Jackets & Coats" in p for p in jackets.catalogue_paths)


def test_groups_follow_the_order_they_were_asked_for():
    assert _bucket_names(KIT) == [
        "Trekking Essentials", "Jackets", "Footwear", "Watches", "Bags", "Gift Ideas",
    ]


def test_each_group_searches_with_its_own_words_not_the_whole_request():
    for bucket in build_offline_query(KIT, []).buckets:
        assert KIT.lower() not in [p.lower() for p in bucket.search_phrases]
        assert "hamper" not in " ".join(bucket.search_phrases) or bucket.name == "Gift Ideas"


def test_headings_quote_what_the_shopper_said():
    jackets = next(b for b in build_offline_query(KIT, []).buckets if b.name == "Jackets")
    assert jackets.why_needed == "You asked for “a warm jacket”."


def test_t_shirt_does_not_also_open_a_shirts_group():
    assert _bucket_names("a plain t-shirt for college") == ["T-Shirts & Tops"]


def test_a_broad_route_gives_up_shelves_a_named_item_covers():
    trek = next(b for b in build_offline_query(KIT, []).buckets if b.name == "Trekking Essentials")
    assert "Men's Apparel/Jackets & Coats" not in trek.catalogue_paths
    assert "Bags & Luggage/Backpacks" not in trek.catalogue_paths
    assert "Men's Apparel/Thermals & Base Layers" in trek.catalogue_paths


def test_a_broad_route_alone_keeps_all_its_shelves():
    trek = build_offline_query("trekking gear for a cold trek", []).buckets[0]
    assert "Men's Apparel/Jackets & Coats" in trek.catalogue_paths


def test_occasion_heading_reads_naturally():
    b = build_offline_query("a gift hamper for my parents' 25th anniversary", []).buckets[0]
    assert b.why_needed == "You asked for “a gift hamper”, for an anniversary."


def test_fallback_chips_match_the_main_path():
    """Keyword mode showed "Under Rs.500" and "Doesn't matter" while the
    rest of the app says "Under ₹500" and "Anyone / unisex"."""
    from app.services.context_slots import BUDGET_QUESTION, GENDER_QUESTION

    questions = build_offline_query("shirt", []).questions
    assert questions == [BUDGET_QUESTION, GENDER_QUESTION]
    assert not any("Rs." in o.label for q in questions for o in q.options)


def test_a_heading_without_an_article_still_reads_correctly():
    b = build_offline_query("wedding sherwani for my brother", []).buckets[0]
    assert b.why_needed == "You asked for “wedding sherwani”."
