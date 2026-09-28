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
