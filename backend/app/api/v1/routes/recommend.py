import json
import logging
from collections.abc import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from app.core.deps import get_catalogue, get_provider
from app.schemas.recommend import RecommendRequest, RecommendResponse
from app.services.recommend import recommend as run_recommendation
from app.services.recommend import recommend_events

router = APIRouter(tags=["recommendations"])
logger = logging.getLogger(__name__)


@router.post("/recommend", response_model=RecommendResponse)
async def recommend(payload: RecommendRequest) -> RecommendResponse:
    """Turn a natural-language shopping request into grouped recommendations.

    Returns one of two modes. When the request leaves a gap that would change
    which products are right, and the user has not already answered or skipped,
    the response carries questions instead of results. Answering re-posts the
    same query with `answers` filled in, which costs no extra interpretation.
    """
    return run_recommendation(payload, get_catalogue(), get_provider())


def _sse(event: str, data: object) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.post("/recommend/stream")
async def recommend_stream(payload: RecommendRequest) -> StreamingResponse:
    """Same pipeline, streamed as server-sent events.

    A completed search spends most of its time in the interpretation call and
    condition lookup. This endpoint reports the real stage boundaries as they
    happen so the wait is legible; the plain POST above remains for clients
    that would rather just block.
    """

    def generate() -> Iterator[str]:
        try:
            for event, data in recommend_events(payload, get_catalogue(), get_provider()):
                if event == "stage":
                    yield _sse("stage", {"stage": data})
                else:
                    yield _sse("result", data.model_dump(mode="json"))
        except Exception:  # noqa: BLE001 - the stream must close cleanly
            logger.exception("Streaming recommendation failed")
            yield _sse("error", {"detail": "Something went wrong finding products."})

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",  # stops nginx buffering the stream flat
        },
    )
