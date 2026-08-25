"""FastAPI application entrypoint."""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.v1.router import api_router
from app.core.config import get_settings
from app.core.deps import load_catalogue, load_embedder, load_provider, load_taxonomy
from app.core.errors import register_error_handlers

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

settings = get_settings()

@asynccontextmanager
async def lifespan(_: FastAPI):
    """Load the catalogue and embedding model once, at startup.

    Doing this lazily on first request would put a multi-second model load in
    front of a real user; doing it here surfaces a missing catalogue in the logs
    at boot instead of as a failed search.

    The embedder is warmed last, and on purpose: it is the slowest of the four
    (~7s) and the only one a request can survive without, so the cheap checks
    log their state before the expensive one starts.
    """
    load_catalogue()
    load_taxonomy()
    load_provider()
    load_embedder()
    yield


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description="Natural-language shopping assistant API.",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

register_error_handlers(app)
app.include_router(api_router, prefix=settings.api_v1_prefix)


@app.get("/")
async def root() -> dict:
    return {"service": settings.app_name, "docs": "/docs", "api": settings.api_v1_prefix}
