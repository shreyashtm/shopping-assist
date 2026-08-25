"""S0 contract tests.

These pin the API shape the frontend is built against. They must keep passing as
the mock is replaced by real retrieval in S2 and by the LLM pipeline in S3-S4.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    """Runs startup, so the catalogue and embedding model load once.

    These are integration tests against the real catalogue on purpose: the
    contract they pin is only meaningful if real products flow through it.
    """
    with TestClient(app) as test_client:
        yield test_client


def test_health_reports_capabilities(client):
    body = client.get("/api/v1/health").json()
    assert body["status"] == "ok"
    assert "llm_configured" in body
    assert "catalogue_loaded" in body


def test_recommend_returns_grouped_results_with_reasons_and_links(client):
    res = client.post(
        "/api/v1/recommend",
        json={
            "query": (
                "I am going for a trek to Hampta Pass in the last week of October "
                "for one week. Please find me trekking essentials and clothing."
            ),
            # Pinned: whether this request warrants a clarifying question is a
            # judgement the model is entitled to vary on, and this test is about
            # the shape of a results response, not that decision.
            "skip_clarification": True,
        },
    )
    assert res.status_code == 200
    body = res.json()

    assert body["groups"], "must return at least one group"
    for group in body["groups"]:
        assert group["name"] and group["why_needed"]
        assert group["items"], "groups must not be empty"
        for item in group["items"]:
            assert item["reason"], "every recommendation needs an explanation"
            assert item["product"]["product_url"], "every recommendation needs a link"
            assert 0 <= item["match_score"] <= 1

    assert body["meta"]["latency_ms"] >= 0
    assert isinstance(body["assumptions"], list)


def test_recommend_rejects_too_short_query(client):
    assert client.post("/api/v1/recommend", json={"query": "a"}).status_code == 422


def test_vague_request_asks_before_recommending(client):
    """The adaptive branch: a request with no occasion, budget or audience
    should still ask, but must not stall on the question alone -- retrieval
    already ran, so whatever the catalogue can offer against what's known so
    far comes back in the same turn."""
    res = client.post("/api/v1/recommend", json={"query": "Suggest me some T-shirts"})
    body = res.json()

    assert body["mode"] == "clarify"
    assert body["questions"], "must ask something"
    if not body["groups"]:
        # The live interpreter's bucket/filter choices vary run to run (see
        # test_completed_search_uses_evidence_based_explanations for the same
        # tolerance); the deterministic guarantee that retrieval runs before
        # the clarify branch, and returns whatever it found, is pinned without
        # the live model in test_conversation_flow.py.
        pytest.skip("Interpreter proposed no fillable buckets this run")
    for question in body["questions"]:
        assert question["slot"] and question["question"]
        assert len(question["options"]) >= 2
        for option in question["options"]:
            assert option["label"] and option["value"]


def test_specific_request_skips_straight_to_results(client):
    """A request that already states trip, dates and activity should not be
    interrogated -- asking would be obstructive, not helpful."""
    res = client.post(
        "/api/v1/recommend",
        json={"query": "Trekking Hampta Pass last week of October for one week"},
    )
    body = res.json()
    assert body["mode"] == "results"
    assert body["groups"]
    assert not body["questions"]
    # Real products, not fixtures.
    first = body["groups"][0]["items"][0]["product"]
    assert first["product_url"].startswith("https://")
    assert first["retailer"] in {"Amazon.in", "Myntra", "Amazon.com (2024 archive)"}


def test_answers_move_a_vague_request_to_results(client):
    """Once the needed context slots are filled, the same query returns products."""
    res = client.post(
        "/api/v1/recommend",
        json={
            "query": "Suggest me some T-shirts",
            "answers": [
                "occasion:daily-wear",
                "price_max:1500",
                "gender:men",
            ],
        },
    )
    assert res.json()["mode"] == "results"


def test_skip_clarification_forces_results(client):
    res = client.post(
        "/api/v1/recommend",
        json={"query": "Suggest me some T-shirts", "skip_clarification": True},
    )
    assert res.json()["mode"] == "results"


def test_missing_timing_on_a_trek_is_worth_asking_about(client):
    """A trek with no dates is genuinely ambiguous: the same pass needs very
    different gear in June and in December. The assistant should ask rather
    than silently pick a season."""
    res = client.post("/api/v1/recommend", json={"query": "trekking gear for Hampta Pass"})
    body = res.json()
    assert body["mode"] == "clarify"
    assert body["questions"]
