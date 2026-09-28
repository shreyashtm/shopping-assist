from app.core.formatting import indian_grouping


def test_indian_grouping_matches_en_in():
    assert indian_grouping(0) == "0"
    assert indian_grouping(999) == "999"
    assert indian_grouping(1000) == "1,000"
    assert indian_grouping(15000) == "15,000"
    assert indian_grouping(150000) == "1,50,000"
    assert indian_grouping(223676) == "2,23,676"
    assert indian_grouping(12345678) == "1,23,45,678"
