"""Unit tests for pure logic in services/recommend.py.

No LLM calls, no network, no catalogue load -- these test local functions in
isolation. Kept separate from test_api_contract.py, which exercises the real
pipeline end to end and therefore needs a live provider.
"""

from app.schemas.query import Bucket
from app.services.recommend import MAX_BUCKETS_FOR_PREVIEW, _select_preview_buckets


def _bucket(name: str, role: str = "recommended", priority: int = 2) -> Bucket:
    return Bucket(name=name, why_needed="test", role=role, priority=priority)


def test_preview_caps_at_the_limit_regardless_of_total_bucket_count():
    """The regression this exists for: a broad request must still get SOME
    preview, capped at the limit, rather than none at all once it crosses it.

    At least one required bucket, so this exercises the cap itself rather than
    the separate no-required-bucket suppression covered elsewhere."""
    buckets = [_bucket("core", role="required")] + [_bucket(f"bucket-{i}") for i in range(7)]
    assert len(buckets) > MAX_BUCKETS_FOR_PREVIEW

    selected = _select_preview_buckets(buckets, MAX_BUCKETS_FOR_PREVIEW)

    assert len(selected) == MAX_BUCKETS_FOR_PREVIEW


def test_required_buckets_are_previewed_before_recommended_or_optional():
    buckets = [
        _bucket("Nice extra", role="optional", priority=1),
        _bucket("Expected layer", role="recommended", priority=1),
        _bucket("Core need", role="required", priority=2),
    ]

    selected = _select_preview_buckets(buckets, 1)

    assert [b.name for b in selected] == ["Core need"], (
        "role must outrank a lower priority number within a weaker role"
    )


def test_priority_breaks_ties_within_the_same_role():
    buckets = [
        _bucket("Third", role="required", priority=3),
        _bucket("First", role="required", priority=1),
        _bucket("Second", role="required", priority=2),
    ]

    selected = _select_preview_buckets(buckets, 2)

    assert [b.name for b in selected] == ["First", "Second"]


def test_stable_sort_preserves_interpreter_order_among_true_ties():
    buckets = [
        _bucket("Alpha", role="required"),
        _bucket("Beta", role="required"),
        _bucket("Gamma", role="required"),
    ]

    selected = _select_preview_buckets(buckets, 2)

    assert [b.name for b in selected] == ["Alpha", "Beta"], (
        "equal role and priority must keep the interpreter's original order"
    )


def test_a_focused_request_is_unaffected():
    """A request already at or under the limit previews everything, exactly
    as the original all-or-nothing gate did for the case it did handle."""
    buckets = [_bucket("Layering"), _bucket("Footwear")]

    selected = _select_preview_buckets(buckets, MAX_BUCKETS_FOR_PREVIEW)

    assert len(selected) == 2


def test_scattered_buckets_with_no_required_role_get_no_preview():
    """The other failure mode this guards against: a request with no real
    signal produces buckets that are all optional/undifferentiated -- "top N
    by priority" is meaningless there, so it must not preview anything."""
    buckets = [_bucket(f"guess-{i}", role="optional", priority=2) for i in range(6)]

    selected = _select_preview_buckets(buckets, MAX_BUCKETS_FOR_PREVIEW)

    assert selected == []


def test_a_single_required_bucket_is_enough_to_justify_a_preview():
    buckets = [
        _bucket("Core need", role="required", priority=1),
        *[_bucket(f"guess-{i}", role="optional", priority=2) for i in range(6)],
    ]

    selected = _select_preview_buckets(buckets, MAX_BUCKETS_FOR_PREVIEW)

    assert "Core need" in [b.name for b in selected]


# --- shelf repair, named items, nearest-shelf gap note ----------------------

def test_shelf_filed_under_wrong_category_is_moved():
    from app.services.interpreter import _canonical_path

    assert _canonical_path("Women's Apparel/Sarees") == "Ethnic Wear/Sarees"
    # Ambiguous names (held by two categories) are left alone.
    assert _canonical_path("Nowhere/Sweaters & Fleece") == "Nowhere/Sweaters & Fleece"


def test_named_item_missing_from_plan_gets_a_group():
    from app.schemas.query import Bucket
    from app.services.recommend import _add_named_items

    saree = Bucket(name="Saree", why_needed="x", catalogue_paths=["Ethnic Wear/Sarees"])
    plan = _add_named_items([saree], "a saree and jewellery for my sister's wedding")
    assert [b.name for b in plan] == ["Saree", "Jewellery"]
    assert plan[1].role == "required"
    # Already covered, or not named: nothing is added.
    assert _add_named_items(plan, "a saree and jewellery") == plan
    assert _add_named_items([saree], "a saree for a wedding") == [saree]


def test_heels_gap_names_nearest_shelves_without_filling_it():
    from app.schemas.query import Bucket
    from app.services.recommend import _no_stock_reason

    assert "Flats" in _no_stock_reason(Bucket(name="Heels", why_needed="x"))
    assert _no_stock_reason(Bucket(name="Gizmo", why_needed="x")).endswith("yet")


def test_implied_gender_from_relation_and_item():
    from app.services.context_slots import implied_gender

    assert implied_gender("a saree and jewellery for my sister's wedding, budget 8000") == "women"
    assert implied_gender("something nice for my girlfriend") == "women"
    assert implied_gender("a watch for my dad") == "men"
    assert implied_gender("gift for my mom and dad") is None
    assert implied_gender("a trekking jacket") is None
