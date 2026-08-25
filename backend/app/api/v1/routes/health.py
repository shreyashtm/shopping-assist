from fastapi import APIRouter

from app.core import deps
from app.core.cache import response_cache
from app.core.config import get_settings

router = APIRouter(tags=["system"])


@router.get("/health")
async def health() -> dict:
    """Readiness probe.

    Reports which capabilities are actually live, so a degraded deployment --
    missing catalogue, missing API key, missing embedding model -- is visible
    without having to run a search and infer it from bad results.
    """
    settings = get_settings()
    catalogue = deps.peek_catalogue()
    return {
        "status": "ok",
        "app": settings.app_name,
        "llm_configured": deps.get_provider() is not None,
        "interpret_model": settings.interpret_model,
        "catalogue_loaded": catalogue is not None,
        "catalogue_size": len(catalogue) if catalogue else 0,
        "embeddings_ready": bool(catalogue and catalogue.has_vectors),
        "cache": response_cache.stats,
    }
