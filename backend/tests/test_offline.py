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
    assert "Clothing" in _bucket_names("some nice tops for summer")
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
    assert "Clothing" in _bucket_names("a top and jeans for college")


def test_sleeping_bag_is_camping_gear_not_a_bag():
    assert _bucket_names("a sleeping bag for camping") == ["Trekking Essentials"]


def test_home_decor_still_routes_to_home_and_kitchen():
    assert _bucket_names("home decor for my new flat") == ["Home & Kitchen"]
    assert _bucket_names("home essentials for a new flat") == ["Home & Kitchen"]
