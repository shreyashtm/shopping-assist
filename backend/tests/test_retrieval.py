"""Retrieval tests.

These pin the judgements that make recommendations correct rather than merely
plausible -- particularly that concrete attributes can outrank raw text
similarity when the request implies real-world constraints.
"""

from datetime import date

import numpy as np
import pytest

from app.schemas.product import Product
from app.schemas.query import Bucket, ClimateContext, ContextConstraints, QueryFilters, ResolvedContext
from app.services.catalogue import Catalogue, embedding_text
from app.services.retrieval import (
    dedupe_across_buckets,
    implied_seasons,
    passes_occasion_context,
    passes_filters,
    sanitize_categories,
    score_product,
    search_bucket,
    temperature_fit,
)


def make_product(pid: str, **overrides) -> Product:
    base = dict(
        id=pid,
        title=overrides.pop("title", f"Product {pid}"),
        brand="TestBrand",
        category=overrides.pop("category", "Men's Apparel"),
        subcategory=overrides.pop("subcategory", "Jackets & Coats"),
        price_inr=overrides.pop("price_inr", 2000),
        description="A test product.",
        retailer="Amazon.in",
        product_url=f"https://example.com/{pid}",
        link_status=overrides.pop("link_status", "verified"),
    )
    base.update(overrides)
    return Product(**base)


COLD = ResolvedContext(climate_note="Late October at 4,200m: nights below freezing, snow likely.")
MILD = ResolvedContext(climate_note=None)
TREK_BUCKET = Bucket(
    name="Layering & Insulation",
    search_phrases=["insulated down jacket for trekking"],
    why_needed="Sub-zero nights at altitude.",
    role="required",
    catalogue_paths=["Men's Apparel/Jackets & Coats"],
)


# --- hard filters ---------------------------------------------------------

def test_only_the_price_ceiling_excludes():
    """Deliberate change of contract, not a weakened test.

    price_min used to exclude, and that emptied five required buckets on a
    real trek request: the shopper picked "Rs 10,000 - Rs 25,000" and every
    one of the catalogue's 43 thermals, 111 socks and 6 navigation items
    costs under Rs 10,000, so the floor deleted those categories entirely.

    A budget range says what someone is willing to spend, not what they
    insist on spending. The ceiling stays a hard constraint; the floor moved
    to `score_product` as a ranking preference, so cheaper products are
    ordered lower but still shown.
    """
    product = make_product("p", price_inr=5000)
    assert not passes_filters(product, QueryFilters(price_max=3000))
    assert passes_filters(product, QueryFilters(price_min=6000))
    assert passes_filters(product, QueryFilters(price_min=1000, price_max=6000))


def test_gender_filter_keeps_unisex():
    womens = make_product("w", attributes={"gender": "women"})
    unisex = make_product("u", attributes={"gender": "unisex"})
    filters = QueryFilters(gender="women")
    assert passes_filters(womens, filters)
    assert passes_filters(unisex, filters), "unisex must not be filtered out by a gender request"
    assert not passes_filters(make_product("m", attributes={"gender": "men"}), filters)


def test_requested_unisex_means_no_preference_between_men_and_women():
    """Fuzzed against a local model: filters.gender="unisex" collapsed a
    77-product category to the 3 products literally tagged unisex, emptying
    a bucket whose catalogue_paths were otherwise entirely correct. The
    GENDER_QUESTION chip is labelled "Anyone / unisex" -- requesting unisex
    means no preference between men's and women's, not "only the literally
    unisex-tagged slice"."""
    filters = QueryFilters(gender="unisex")
    assert passes_filters(make_product("m", attributes={"gender": "men"}), filters)
    assert passes_filters(make_product("w", attributes={"gender": "women"}), filters)
    assert passes_filters(make_product("u", attributes={"gender": "unisex"}), filters)


def test_requested_unisex_still_excludes_kids():
    """Kids is a distinct, deliberate category (age-appropriateness, not
    cut) -- expressing no preference between men's and women's is not a
    request for children's items too."""
    filters = QueryFilters(gender="unisex")
    assert not passes_filters(make_product("k", attributes={"gender": "kids"}), filters)


def test_sanitize_categories_drops_unrecognized_noise():
    """Fuzzed against a local model: nothing in the schema tells the model
    what filters.categories should contain, and a weak model filled it with
    plain topic words ('trekking', 'outdoor') instead of real taxonomy
    entries. Matched literally that zeroes every candidate in every bucket --
    the exact failure this exists to prevent."""
    filters = QueryFilters(categories=["trekking", "outdoor"])
    sanitized = sanitize_categories(filters)
    assert sanitized.categories == [], "fully-noisy input must leave no category constraint"


def test_sanitize_categories_keeps_recognized_values_and_drops_only_noise():
    filters = QueryFilters(categories=["Men's Apparel", "not-a-real-category"])
    sanitized = sanitize_categories(filters)
    assert sanitized.categories == ["Men's Apparel"]


def test_sanitize_categories_is_a_no_op_when_everything_is_recognized():
    filters = QueryFilters(categories=["Men's Apparel", "Footwear/Boots"])
    sanitized = sanitize_categories(filters)
    assert sanitized.categories == ["Men's Apparel", "Footwear/Boots"]


def test_sanitize_categories_leaves_an_empty_list_alone():
    filters = QueryFilters()
    assert sanitize_categories(filters) is filters


def test_category_filter_accepts_bare_category_name():
    product = make_product("p", category="Men's Apparel", subcategory="Casual Shirts")
    assert passes_filters(product, QueryFilters(categories=["Men's Apparel"]))
    assert not passes_filters(product, QueryFilters(categories=["Footwear"]))


def test_category_filter_also_accepts_a_full_taxonomy_path():
    """The interpreter's own schema and prompt describe catalogue paths as
    "Category/Subcategory" throughout -- `catalogue_paths` always uses that
    form -- but never says `filters.categories` must be bare. When the model
    reasonably (and, live, reproducibly) puts a full path there instead of a
    bare category name, every real product must not silently fail the filter:
    a live capture of "Men's Apparel/Casual Shirts" in `filters.categories`
    zeroed out an entire turn's results for a category that holds 74 products,
    with no error -- just an empty screen."""
    product = make_product("p", category="Men's Apparel", subcategory="Casual Shirts")
    assert passes_filters(product, QueryFilters(categories=["Men's Apparel/Casual Shirts"]))
    assert not passes_filters(
        product, QueryFilters(categories=["Men's Apparel/Formal Shirts"])
    )


def test_wedding_bucket_rejects_explicitly_non_wedding_occasion():
    bucket = Bucket(
        name="Accessories & Jewellery",
        search_phrases=["wedding accessories"],
        why_needed="Accessories for a wedding outfit.",
        catalogue_paths=["Watches & Jewellery/Jewellery"],
    )
    halloween = make_product(
        "cosplay",
        category="Watches & Jewellery",
        subcategory="Jewellery",
        attributes={"occasion": ["party"]},
    )
    festive = make_product(
        "festive",
        category="Watches & Jewellery",
        subcategory="Jewellery",
        attributes={"occasion": ["festive"]},
    )

    assert not passes_occasion_context(halloween, bucket)
    assert passes_occasion_context(festive, bucket)


def test_anniversary_bucket_rejects_a_birthday_tagged_product():
    """Live capture: a "Genshin Merch Box" (occasion=[birthday, festive])
    surfaced as a top pick for a parents' 25th anniversary hamper. The
    wedding-only check above never covered anniversary at all."""
    bucket = Bucket(
        name="Anniversary Gift",
        search_phrases=["premium gift hamper"],
        why_needed="A 25th anniversary gift for your parents.",
        catalogue_paths=["Gifting/Keepsakes"],
    )
    genshin_merch = make_product(
        "genshin", category="Gifting", subcategory="Keepsakes",
        attributes={"occasion": ["birthday", "festive"]},
    )
    assert not passes_occasion_context(genshin_merch, bucket)


def test_anniversary_bucket_accepts_a_matching_or_untagged_product():
    bucket = Bucket(
        name="Anniversary Gift",
        search_phrases=["premium gift hamper"],
        why_needed="A 25th anniversary gift for your parents.",
        catalogue_paths=["Gifting/Keepsakes"],
    )
    tagged = make_product(
        "matching", category="Gifting", subcategory="Keepsakes",
        attributes={"occasion": ["anniversary"]},
    )
    untagged = make_product("plain", category="Gifting", subcategory="Keepsakes")
    assert passes_occasion_context(tagged, bucket)
    assert passes_occasion_context(untagged, bucket), (
        "missing evidence must not be treated as a conflict"
    )


def test_generic_gifting_bucket_is_not_gated_by_occasion():
    """Only the curated, specific occasions gate at all -- a bucket that
    doesn't name one of them (an ordinary gift bucket with no stated
    occasion) must not reject products over occasion mismatches."""
    bucket = Bucket(
        name="Gift Ideas",
        search_phrases=["gift set"],
        why_needed="A nice gift.",
        catalogue_paths=["Gifting/Keepsakes"],
    )
    birthday_only = make_product(
        "b", category="Gifting", subcategory="Keepsakes",
        attributes={"occasion": ["birthday"]},
    )
    assert passes_occasion_context(birthday_only, bucket)


# --- temperature reasoning ------------------------------------------------

def test_warmer_jacket_wins_for_sub_zero_conditions():
    warm = make_product("warm", attributes={"temp_rating_c": -10})
    mild = make_product("mild", attributes={"temp_rating_c": 5})
    assert temperature_fit(warm, COLD)[0] > temperature_fit(mild, COLD)[0]


def test_measured_cold_nights_rank_colder_jacket_higher():
    """Regression for the Hampta Pass defect: -20C beats -5C at -14.7C nights."""
    hampta = ResolvedContext(
        climate=ClimateContext(
            source="climatological",
            temp_min_c=-14.7,
            temp_max_c=-2.1,
            elevation_m=4393,
        )
    )
    warm = make_product("warm", attributes={"temp_rating_c": -20})
    inadequate = make_product("bad", attributes={"temp_rating_c": -5})
    assert temperature_fit(warm, hampta)[0] > temperature_fit(inadequate, hampta)[0]
    assert temperature_fit(inadequate, hampta)[0] < 1.0


def test_temperature_ignored_when_request_implies_no_cold():
    warm = make_product("warm", attributes={"temp_rating_c": -10})
    assert temperature_fit(warm, MILD) == (0.0, None)


def test_uninsulated_product_has_no_temperature_opinion():
    boots = make_product("boots", attributes={"temp_rating_c": None})
    assert temperature_fit(boots, COLD) == (0.0, None)


def test_attributes_can_outrank_text_similarity():
    """The core claim: a better-suited product wins even when it *reads* less
    like the query. The mild jacket has higher cosine similarity here."""
    warm = make_product("warm", attributes={"temp_rating_c": -10, "use_case": ["trekking"]})
    mild = make_product("mild", attributes={"temp_rating_c": 8})

    warm_scored = score_product(warm, semantic=0.60, bucket=TREK_BUCKET, context=COLD)
    mild_scored = score_product(mild, semantic=0.72, bucket=TREK_BUCKET, context=COLD)

    assert warm_scored.score > mild_scored.score
    assert any("sub-zero" in r for r in warm_scored.reasons)


def test_scoring_records_grounded_reasons():
    product = make_product("p", attributes={"use_case": ["trekking"], "temp_rating_c": -8})
    scored = score_product(product, semantic=0.5, bucket=TREK_BUCKET, context=COLD)
    assert scored.reasons, "scoring must explain itself for the ranker to stay grounded"


def test_out_of_stock_is_heavily_penalised():
    stocked = make_product("a", in_stock=True)
    gone = make_product("b", in_stock=False)
    a = score_product(stocked, 0.5, TREK_BUCKET, COLD)
    b = score_product(gone, 0.5, TREK_BUCKET, COLD)
    assert a.score > b.score


def test_unverified_link_is_mildly_penalised():
    verified = score_product(make_product("a", link_status="verified"), 0.5, TREK_BUCKET, COLD)
    blocked = score_product(make_product("b", link_status="blocked"), 0.5, TREK_BUCKET, COLD)
    assert verified.score > blocked.score
    # Mild, not disqualifying -- a blocked link still points at a real product.
    assert verified.score - blocked.score < 0.1


# --- bucket search --------------------------------------------------------

def test_search_bucket_respects_filters_and_limit():
    products = [make_product(f"p{i}", price_inr=1000 * i) for i in range(1, 8)]
    vectors = np.tile(np.eye(1, 384, dtype=np.float32), (len(products), 1))
    catalogue = Catalogue(products, vectors)

    results = search_bucket(
        catalogue,
        query_vectors=vectors[0],
        bucket=TREK_BUCKET,
        filters=QueryFilters(price_max=4000),
        context=COLD,
        limit=2,
    )
    assert len(results) == 2
    assert all(r.product.price_inr <= 4000 for r in results)


def test_global_category_filter_does_not_zero_out_a_bucket_it_never_described():
    """Reproduces the fuzzed failure directly: the interpreter set
    filters.categories=["Outdoor & Camping Gear"] for the whole request while
    separately (and correctly) planning a Footwear bucket. A single
    request-global category value cannot describe every bucket in a
    multi-category request; the bucket's own catalogue_paths is the
    authoritative, per-bucket type gate and must win when the two disagree
    entirely."""
    boots = [make_product(f"boot{i}", category="Footwear", subcategory="Boots") for i in range(3)]
    vectors = np.tile(np.eye(1, 384, dtype=np.float32), (len(boots), 1))
    catalogue = Catalogue(boots, vectors)
    footwear_bucket = Bucket(
        name="Footwear",
        search_phrases=["trekking boots"],
        why_needed="Footwear for the trek.",
        catalogue_paths=["Footwear/Boots"],
    )

    results = search_bucket(
        catalogue,
        query_vectors=vectors[0],
        bucket=footwear_bucket,
        filters=QueryFilters(categories=["Outdoor & Camping Gear"]),
        context=MILD,
        limit=8,
    )

    assert results, "a bucket's own catalogue_paths must not be overridden by an unrelated global category filter"


def test_global_category_filter_still_applies_when_it_overlaps_the_bucket():
    """The fix is scoped narrowly: when filters.categories genuinely does
    describe this bucket, it must keep working as a real constraint, not be
    disabled wholesale."""
    mens_jacket = make_product("mj", category="Men's Apparel", subcategory="Jackets & Coats")
    vectors = np.tile(np.eye(1, 384, dtype=np.float32), (1, 384))
    catalogue = Catalogue([mens_jacket], vectors)
    bucket = Bucket(
        name="Layering",
        search_phrases=["jacket"],
        why_needed="x",
        catalogue_paths=["Men's Apparel/Jackets & Coats", "Women's Apparel/Jackets & Coats"],
    )

    results = search_bucket(
        catalogue, query_vectors=vectors[0], bucket=bucket,
        filters=QueryFilters(categories=["Women's Apparel"]), context=MILD, limit=8,
    )
    assert results == [], "an overlapping-but-narrower global category filter must still exclude"


def test_catalogue_rejects_misaligned_embedding_matrix():
    with pytest.raises(ValueError):
        Catalogue([make_product("a")], np.zeros((5, 384), dtype=np.float32))


def test_dedupe_keeps_product_in_its_best_bucket():
    product = make_product("shared")
    strong = score_product(product, 0.9, TREK_BUCKET, COLD)
    weak = score_product(product, 0.2, TREK_BUCKET, COLD)
    deduped = dedupe_across_buckets({"Layering": [strong], "Footwear": [weak]})
    assert [i.product.id for i in deduped["Layering"]] == ["shared"]
    assert deduped["Footwear"] == []


# --- indexed text ---------------------------------------------------------

def test_embedding_text_includes_inferred_attributes():
    """Attributes must be searchable, not just filterable: nobody types a
    product title, they describe conditions."""
    product = make_product(
        "p",
        title="Mens Arctic Crest",
        attributes={"temp_rating_c": -15, "use_case": ["trekking"], "material": "down"},
    )
    text = embedding_text(product)
    assert "trekking" in text
    assert "-15" in text
    assert "down" in text


def test_boosts_cannot_rescue_an_irrelevant_product():
    """Regression: attribute boosts must not override semantic relevance.

    With additive boosts, a jacket at cosine 0.42 collected enough trekking and
    temperature bonuses to outrank a trekking boot at cosine 0.63 inside a
    bucket that explicitly asked for footwear. Multiplicative boosts scale with
    relevance, so a poor semantic match can no longer be promoted by tags.
    """
    footwear_bucket = Bucket(
        name="Footwear",
        search_phrases=["insulated waterproof trekking boots with ankle support"],
        why_needed="Snow and steep scree.",
        role="required",
        # Both products below sit in this path, so the test isolates scoring
        # rather than gating.
        catalogue_paths=["Footwear/Boots"],
    )
    jacket_kwargs = {"category": "Footwear", "subcategory": "Boots"}
    # The jacket is tagged perfectly for the trip but is not footwear.
    jacket = make_product(
        "jacket",
        attributes={"temp_rating_c": -10, "use_case": ["trekking"], "season": ["winter"]},
        **jacket_kwargs,
    )
    # The boot matches the bucket semantically but carries no temperature rating.
    boot = make_product("boot", attributes={"use_case": ["trekking"]}, **jacket_kwargs)

    jacket_scored = score_product(jacket, semantic=0.42, bucket=footwear_bucket, context=COLD)
    boot_scored = score_product(boot, semantic=0.63, bucket=footwear_bucket, context=COLD)

    assert boot_scored.score > jacket_scored.score, (
        "a better semantic match must win; boosts are tie-breakers, not overrides"
    )


def test_boosts_still_reorder_comparable_products():
    """The other half of the trade: among products that *are* relevant,
    attributes must still decide. Otherwise the fix would have flattened the
    reasoning that makes recommendations correct."""
    bucket = TREK_BUCKET
    warm = make_product("warm", attributes={"temp_rating_c": -10, "use_case": ["trekking"]})
    mild = make_product("mild", attributes={"temp_rating_c": 8})

    warm_scored = score_product(warm, semantic=0.60, bucket=bucket, context=COLD)
    mild_scored = score_product(mild, semantic=0.66, bucket=bucket, context=COLD)
    assert warm_scored.score > mild_scored.score


def test_weak_matches_are_filtered_out():
    """A bucket should return nothing rather than pad itself with products that
    merely share a tag."""
    products = [make_product(f"p{i}") for i in range(5)]
    vectors = np.tile(np.array([[1.0] + [0.0] * 383], dtype=np.float32), (len(products), 1))
    catalogue = Catalogue(products, vectors)
    # Query vector orthogonal to every product: nothing is relevant.
    orthogonal = np.array([0.0, 1.0] + [0.0] * 382, dtype=np.float32)

    results = search_bucket(
        catalogue, orthogonal, TREK_BUCKET, QueryFilters(), COLD, limit=5
    )
    assert results == []


def test_group_visibility_follows_type_correctness_not_similarity():
    """Policy change, recorded deliberately.

    Before category gating, a similarity floor was the only defence against
    irrelevant products, so a 0.31 match was suppressed. Now a candidate only
    reaches scoring if its product type is one the slot accepts, which makes it
    relevant by construction -- suppressing it on similarity discarded genuine
    trekking boots and emptied whole groups. Only a near-zero match is dropped.
    """
    from app.services.retrieval import is_group_worth_showing

    modest = score_product(make_product("modest"), semantic=0.31, bucket=TREK_BUCKET, context=COLD)
    strong = score_product(make_product("ok"), semantic=0.55, bucket=TREK_BUCKET, context=COLD)
    noise = score_product(make_product("noise"), semantic=0.01, bucket=TREK_BUCKET, context=COLD)

    assert is_group_worth_showing([strong])
    assert is_group_worth_showing([modest]), "type-correct products are relevant"
    assert not is_group_worth_showing([noise])
    assert not is_group_worth_showing([])


def test_each_phrase_is_searched_separately():
    """Regression: averaging phrase vectors produced a centroid that matched
    none of them. A bucket asking for both a headlamp and a jacket must find
    both, not their meaningless midpoint."""
    headlamp = make_product("headlamp")
    jacket = make_product("jacket")
    catalogue = Catalogue(
        [headlamp, jacket],
        np.array(
            [[1.0, 0.0] + [0.0] * 382, [0.0, 1.0] + [0.0] * 382], dtype=np.float32
        ),
    )
    # Two orthogonal phrase vectors, one per product. Their mean would score
    # ~0.707 against each -- below neither, but a poorer signal than either.
    phrases = np.array(
        [[1.0, 0.0] + [0.0] * 382, [0.0, 1.0] + [0.0] * 382], dtype=np.float32
    )
    results = search_bucket(catalogue, phrases, TREK_BUCKET, QueryFilters(), COLD, limit=2)

    assert len(results) == 2
    # Each product matched one phrase exactly, so both should score ~1.0.
    assert all(r.semantic > 0.99 for r in results)


def test_slot_with_no_catalogue_path_returns_nothing():
    """A slot the planner could not map is a declared gap. Retrieval must not
    go looking for the nearest thing -- that is how women's jeans once answered
    a request for wedding-suit trousers."""
    products = [make_product(f"p{i}") for i in range(4)]
    vectors = np.tile(np.eye(1, 384, dtype=np.float32), (len(products), 1))
    catalogue = Catalogue(products, vectors)
    unmapped = Bucket(
        name="Formal Trousers",
        search_phrases=["formal trousers", "dress pants"],
        why_needed="A suit needs matching trousers.",
        role="required",
        catalogue_paths=[],
    )
    assert search_bucket(catalogue, vectors[0], unmapped, QueryFilters(), COLD, 5) == []


def test_category_gate_beats_high_similarity():
    """The original failure, as a test: a product of the wrong type must be
    excluded no matter how well it scores."""
    jeans = make_product("jeans", category="Women's Apparel", subcategory="Jeans")
    trousers = make_product(
        "trousers", category="Men's Apparel", subcategory="Trousers & Chinos"
    )
    catalogue = Catalogue(
        [jeans, trousers],
        np.array(
            # Jeans are the *better* semantic match here, deliberately.
            [[1.0] + [0.0] * 383, [0.6, 0.8] + [0.0] * 382], dtype=np.float32
        ),
    )
    slot = Bucket(
        name="Formal Trousers",
        search_phrases=["formal trousers"],
        why_needed="Suit trousers.",
        role="required",
        catalogue_paths=["Men's Apparel/Trousers & Chinos"],
    )
    results = search_bucket(
        catalogue, np.array([1.0] + [0.0] * 383, dtype=np.float32),
        slot, QueryFilters(), COLD, 5,
    )
    assert [r.product.id for r in results] == ["trousers"]


def test_matches_paths_is_exact():
    from app.services.retrieval import matches_paths

    boot = make_product("b", category="Footwear", subcategory="Boots")
    assert matches_paths(boot, ["Footwear/Boots"])
    assert not matches_paths(boot, ["Footwear/Formal Shoes"])
    assert not matches_paths(boot, []), "an empty path list accepts nothing"


def test_matches_paths_tolerates_a_leading_or_trailing_slash():
    """Fuzzed against a stronger local model: it returned catalogue_paths
    with a leading slash ("/Men's Apparel/Jackets & Coats"), which fails an
    exact-match test as completely as a wrong path -- zero candidates, no
    error. The formatting is noise; the path underneath was correct."""
    from app.services.retrieval import matches_paths

    boot = make_product("b", category="Footwear", subcategory="Boots")
    assert matches_paths(boot, ["/Footwear/Boots"])
    assert matches_paths(boot, ["Footwear/Boots/"])
    assert not matches_paths(boot, ["/Footwear/Formal Shoes"]), (
        "normalizing formatting must not make a genuinely wrong path match"
    )


# --- contextual suitability: the monsoon / winter-jacket defect ------------
#
# The failure these pin: for Mumbai in monsoon (26-29C) a snow jacket rated to
# -5C ranked first in a "lightweight rain jacket" bucket. Nothing rejected it
# and nothing penalised it -- the temperature rule only ever fired when it was
# cold, and season inference produced an empty set for anything between 5C and
# 30C, so there was no signal in either direction.

MUMBAI_MONSOON = ResolvedContext(
    location="Mumbai",
    climate=ClimateContext(
        source="measured",
        temp_min_c=25.8,
        temp_max_c=28.9,
        precipitation_mm=4.3,
        window_start=date(2026, 8, 24),
        window_end=date(2026, 8, 24),
    ),
)


def test_overwarm_insulation_is_penalised_in_heat():
    """The regression. A -5C jacket must object to a 29C day, not shrug."""
    snow_jacket = make_product("snow", attributes={"temp_rating_c": -5})
    fit, reason = temperature_fit(snow_jacket, MUMBAI_MONSOON)
    assert fit < 0, "insulation far beyond the conditions must score negative"
    assert reason and "too warm" in reason


def test_thermal_mismatch_scales_with_how_wrong_it_is():
    severe = make_product("severe", attributes={"temp_rating_c": -5})
    mild = make_product("mild", attributes={"temp_rating_c": 5})
    fine = make_product("fine", attributes={"temp_rating_c": 15})

    assert temperature_fit(severe, MUMBAI_MONSOON)[0] < temperature_fit(mild, MUMBAI_MONSOON)[0]
    assert temperature_fit(mild, MUMBAI_MONSOON)[0] < 0
    assert temperature_fit(fine, MUMBAI_MONSOON)[0] == 0.0, "appropriate kit draws no objection"


def test_cold_nights_outrank_warm_days():
    """A high desert is +20C by day and below freezing at night.

    The night is what the kit has to survive, so a warm daytime high must not
    become an argument against carrying insulation.
    """
    desert = ResolvedContext(
        climate=ClimateContext(source="measured", temp_min_c=-8.0, temp_max_c=20.0)
    )
    parka = make_product("parka", attributes={"temp_rating_c": -15})
    fit, _ = temperature_fit(parka, desert)
    assert fit > 0, "cold-side suitability must win outright when nights are cold"


def test_thermal_conflict_sinks_a_strong_text_match():
    """Ranking must not be able to overpower contextual suitability."""
    bucket = Bucket(
        name="Lightweight Waterproof Jacket",
        search_phrases=["lightweight waterproof rain jacket"],
        why_needed="Heavy rain in Mumbai.",
        catalogue_paths=["Men's Apparel/Jackets & Coats"],
    )
    snow_jacket = make_product("snow", attributes={"temp_rating_c": -5})
    rain_shell = make_product("shell", attributes={"temp_rating_c": 15})

    # The snow jacket reads *better* against the query text than the shell.
    wrong = score_product(snow_jacket, 0.90, bucket, MUMBAI_MONSOON)
    right = score_product(rain_shell, 0.60, bucket, MUMBAI_MONSOON)

    assert right.score > wrong.score, (
        "a product wrong for the conditions must lose even with higher similarity"
    )


def test_season_inference_has_no_dead_zone():
    """5C-30C previously produced no season at all -- most of India, most of the year."""
    seasons = implied_seasons(MUMBAI_MONSOON)
    assert "monsoon" in seasons, "sustained daily rainfall must read as monsoon"
    assert "winter" not in seasons, "26C nights are not winter"


def test_precipitation_is_read_as_a_rate_not_a_window_total():
    """4.3mm is a wet day; as a trip total it failed a threshold meant for a week."""
    one_day = MUMBAI_MONSOON.climate
    assert one_day.precipitation_mm_per_day == pytest.approx(4.3)

    spread = one_day.model_copy(update={"window_end": date(2026, 9, 6)})
    assert spread.precipitation_mm_per_day < 1.0, "the same total over 2 weeks is not monsoon"
    assert "monsoon" not in implied_seasons(ResolvedContext(climate=spread))


# --- generalized suitability layer: constraints wired into real retrieval --
#
# test_suitability.py and test_constraints.py pin the two modules in
# isolation; these confirm the wiring in score_product/search_bucket itself --
# that a hard mismatch actually disappears from a bucket's results rather
# than merely scoring poorly, and that the module docstring's severity
# ordering (hard > strong > soft) holds when real candidates compete.

RAIN_BUCKET = Bucket(
    name="Rain Gear",
    search_phrases=["rain jacket"],
    why_needed="Heavy rain expected.",
    catalogue_paths=["Men's Apparel/Jackets & Coats"],
)
HEAVY_RAIN = ContextConstraints(min_water_resistance="waterproof")


def test_hard_suitability_mismatch_is_excluded_from_search_bucket_results():
    unprotected = make_product("none", attributes={"water_resistance": "none"})
    waterproof = make_product("proof", attributes={"water_resistance": "waterproof"})
    vectors = np.tile(np.eye(1, 384, dtype=np.float32), (2, 1))
    catalogue = Catalogue([unprotected, waterproof], vectors)

    results = search_bucket(
        catalogue,
        query_vectors=vectors[0],
        bucket=RAIN_BUCKET,
        filters=QueryFilters(),
        context=ResolvedContext(),
        limit=10,
        constraints=HEAVY_RAIN,
    )

    ids = {r.product.id for r in results}
    assert "proof" in ids
    assert "none" not in ids, "a hard suitability mismatch must not merely score low, it must be gone"


def test_score_product_without_constraints_is_unaffected_by_suitability():
    """Default (no constraints passed) must behave exactly as before this
    layer existed -- every pre-existing retrieval test relies on this."""
    product = make_product("p", attributes={"water_resistance": "none", "formality": "casual"})
    with_none = score_product(product, 0.5, TREK_BUCKET, COLD)
    with_empty = score_product(product, 0.5, TREK_BUCKET, COLD, ContextConstraints())
    assert with_none.score == with_empty.score


def test_strong_suitability_penalty_sinks_a_better_text_match():
    """Mirrors test_thermal_conflict_sinks_a_strong_text_match for the new
    axis: a formal-occasion bucket must not let a casual item win on
    similarity alone once formality is known."""
    bucket = Bucket(
        name="Wedding Outfit",
        search_phrases=["wedding outfit jacket"],
        why_needed="Friend's wedding.",
        catalogue_paths=["Men's Apparel/Jackets & Coats"],
    )
    constraints = ContextConstraints(required_formality="formal")
    casual_but_close_text = make_product("casual", attributes={"formality": "casual"})
    formal_but_weaker_text = make_product("formal", attributes={"formality": "formal"})

    weak_text_match = score_product(formal_but_weaker_text, 0.55, bucket, MILD, constraints)
    strong_text_match = score_product(casual_but_close_text, 0.70, bucket, MILD, constraints)

    assert weak_text_match.score > strong_text_match.score, (
        "the right formality must be able to outrank higher raw similarity"
    )


# --- Use-case conflict: axis 2 must gate, not merely boost ------------------
#
# Real defect, seen on the flagship trek query. "Trekking Trousers" returned
# "Women's Shiny Pleated Wide Leg Pants Party Night" -- a product literally
# carrying use_case=['party']. The two-axis taxonomy (see services/taxonomy.py)
# separates *what a thing is* from *what it is for*, but only axis 1 was
# enforced: catalogue_paths is a hard gate, while use_case was additive-only
# (BOOST_USE_CASE) with no penalty for an opposing value. A party pant
# therefore lost only a boost it never earned, and with no trekking trousers
# in stock for that gender and budget it ranked first.
#
# The honest outcome is an unfilled slot -- which this system already produces
# for Trekking Footwear in the very same response.

PARTY_BUCKET = Bucket(
    name="Party Outfit",
    search_phrases=["party wear trousers"],
    why_needed="A night out.",
    role="required",
    catalogue_paths=["Women's Apparel/Trousers"],
)
TREK_TROUSERS_BUCKET = Bucket(
    name="Trekking Trousers",
    search_phrases=["trekking trousers for hiking"],
    why_needed="Durable trousers for a multi-day trek at altitude.",
    role="required",
    catalogue_paths=["Women's Apparel/Trousers"],
)


def _trouser(pid: str, use_case: list[str]) -> Product:
    return make_product(
        pid,
        category="Women's Apparel",
        subcategory="Trousers",
        attributes={"gender": "women", "use_case": use_case},
    )


def test_conflicting_use_case_sinks_below_a_neutral_product():
    """A party pant must not outrank an unlabelled trouser for a trek, even
    when it is the better semantic match."""
    party = _trouser("party", ["party"])
    neutral = _trouser("neutral", [])

    party_scored = score_product(
        party, semantic=0.72, bucket=TREK_TROUSERS_BUCKET, context=MILD
    )
    neutral_scored = score_product(
        neutral, semantic=0.60, bucket=TREK_TROUSERS_BUCKET, context=MILD
    )

    assert neutral_scored.score > party_scored.score


def test_missing_use_case_is_never_treated_as_a_conflict():
    """Same rule temperature_fit() and suitability.evaluate() already follow:
    absent evidence is not opposing evidence."""
    neutral = _trouser("neutral", [])
    baseline = score_product(neutral, semantic=0.60, bucket=TREK_TROUSERS_BUCKET, context=MILD)

    assert baseline.score >= 0.60 * 0.99


def test_matching_use_case_is_still_rewarded():
    trekking = _trouser("trek", ["trekking"])
    neutral = _trouser("neutral", [])

    trek_scored = score_product(
        trekking, semantic=0.60, bucket=TREK_TROUSERS_BUCKET, context=MILD
    )
    neutral_scored = score_product(
        neutral, semantic=0.60, bucket=TREK_TROUSERS_BUCKET, context=MILD
    )

    assert trek_scored.score > neutral_scored.score


def test_no_penalty_when_the_bucket_expresses_no_use_case():
    """Only an explicit conflict counts. A bucket with no strong use-case
    signal must not penalise anything."""
    party = _trouser("party", ["party"])

    in_party_bucket = score_product(party, semantic=0.60, bucket=PARTY_BUCKET, context=MILD)

    assert in_party_bucket.score >= 0.60 * 0.99


# --- price_min orders, it does not exclude ---------------------------------
#
# Reported from the live app. A shopper picked "Rs 10,000 - Rs 25,000" for a
# Hampta Pass trek and five required buckets came back empty: the catalogue
# holds 43 thermals, 111 socks and 6 navigation items, and *every one* of them
# costs under Rs 10,000. Nothing in those categories costs Rs 10,000, so the
# floor deleted the entire category.
#
# A budget range states what someone is willing to spend, not what they insist
# on spending. price_max is a real constraint -- "I cannot afford more than
# this". price_min is a preference, and a shopper is never harmed by being
# shown a cheaper item that fits. It belongs on the ranking side, with the
# boosts, not on the gate side with catalogue_paths.


def test_price_min_does_not_exclude_cheaper_products():
    cheap = make_product("cheap", price_inr=500)

    assert passes_filters(cheap, QueryFilters(price_min=10000, price_max=25000))


def test_price_max_still_excludes():
    """The genuine constraint is untouched: too expensive is still too
    expensive."""
    dear = make_product("dear", price_inr=30000)

    assert not passes_filters(dear, QueryFilters(price_min=10000, price_max=25000))


def test_a_product_inside_the_stated_range_outranks_one_far_below():
    """The preference survives as ordering: when both exist, the in-budget
    product comes first."""
    in_range = make_product("in_range", price_inr=12000)
    far_below = make_product("far_below", price_inr=400)
    filters = QueryFilters(price_min=10000, price_max=25000)

    scored_in = score_product(in_range, 0.60, TREK_BUCKET, MILD, filters=filters)
    scored_below = score_product(far_below, 0.60, TREK_BUCKET, MILD, filters=filters)

    assert scored_in.score > scored_below.score


def test_no_price_preference_leaves_scores_alone():
    product = make_product("p", price_inr=400)

    with_filter = score_product(product, 0.60, TREK_BUCKET, MILD, filters=QueryFilters())
    assert with_filter.score >= 0.60 * 0.99


def test_below_budget_products_are_ordered_by_closeness_to_the_floor():
    """Being under budget is a matter of degree, not a yes/no.

    If a shopper asks for Rs 10,000-25,000 and nothing is in range, a
    Rs 9,000 jacket is very nearly what they asked for while a Rs 500 one is
    not. A flat in-range/out-of-range boost scores those two identically,
    which is why the preference is scaled by how close the price gets to the
    stated floor.
    """
    filters = QueryFilters(price_min=10000, price_max=25000)
    near = make_product("near", price_inr=9000)
    far = make_product("far", price_inr=500)

    scored_near = score_product(near, 0.60, TREK_BUCKET, MILD, filters=filters)
    scored_far = score_product(far, 0.60, TREK_BUCKET, MILD, filters=filters)

    assert scored_near.score > scored_far.score


def test_in_budget_still_beats_the_closest_below_budget_product():
    filters = QueryFilters(price_min=10000, price_max=25000)
    inside = make_product("inside", price_inr=10500)
    just_under = make_product("just_under", price_inr=9800)

    scored_in = score_product(inside, 0.60, TREK_BUCKET, MILD, filters=filters)
    scored_under = score_product(just_under, 0.60, TREK_BUCKET, MILD, filters=filters)

    assert scored_in.score > scored_under.score
