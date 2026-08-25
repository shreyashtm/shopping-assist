"""Tests for settings parsing.

cors_origin_list has one real-world failure mode: a trailing slash pasted
into the env var. Browser Origin headers never carry one (Origin is
scheme+host+port only, never a path), and CORSMiddleware matches exactly --
so an unstripped trailing slash silently rejects every real browser request.
This was invisible until it broke a live deployment.
"""

from app.core.config import Settings


def test_cors_origin_list_strips_trailing_slash():
    settings = Settings(_env_file=None, cors_origins="https://example.vercel.app/")
    assert settings.cors_origin_list == ["https://example.vercel.app"]


def test_cors_origin_list_handles_multiple_origins_with_whitespace():
    settings = Settings(
        _env_file=None,
        cors_origins="https://example.vercel.app/, http://localhost:3000",
    )
    assert settings.cors_origin_list == [
        "https://example.vercel.app",
        "http://localhost:3000",
    ]


def test_cors_origin_list_default_has_no_trailing_slash():
    settings = Settings(_env_file=None)
    assert all(not o.endswith("/") for o in settings.cors_origin_list)

