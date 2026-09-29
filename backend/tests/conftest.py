"""Shared test configuration.

Tests marked `live` send real requests to whichever LLM provider backend/.env
configures. They used to run on every `pytest` invocation whenever a key was
present, silently spending the provider's quota -- on a free tier, 50 requests
a day -- and slowing the suite from seconds to minutes. They now run only when
asked for explicitly:

    RUN_LIVE_TESTS=1 uv run pytest -m live
"""

import os

import pytest


def pytest_collection_modifyitems(config, items):
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="calls the real LLM provider; set RUN_LIVE_TESTS=1 to run")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(autouse=True)
def _fresh_plan_cache():
    """The first turn's plan is reused by later turns of the same request;
    one test's fake plan must never answer another test's follow-up."""
    from app.services.recommend import plan_cache

    plan_cache.clear()
    yield
    plan_cache.clear()
