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
