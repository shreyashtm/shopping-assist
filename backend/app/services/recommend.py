"""Orchestration: request in, recommendations (or questions) out.

The pipeline is deliberately short and mostly free:

    interpret (1 LLM call)  ->  resolve climate (HTTP)  ->  retrieve (local)

One LLM call per completed search when explanations are composed from retrieval
evidence rather than a second model call. When the interpreter decides it needs
to ask something, the request stops after one call and returns questions; the
answers come back as structured chip values, so resuming costs no extra
interpretation either.

Failure policy: never show an empty screen. If the LLM is unreachable the
request falls through to a keyword interpretation and the response says so via
`degraded_mode`, rather than returning nothing or pretending the weaker result.
"""

import logging
import time
import uuid
from collections.abc import Iterator
from datetime import date
from typing import Any

from app.adapters.embeddings.local import get_embedder
from app.adapters.llm.base import LLMProvider, LLMUnavailable
from app.adapters.weather.open_meteo import OpenMeteoClient
from app.core.cache import cache_key, response_cache
from app.core.config import get_settings
from app.core.deps import get_taxonomy
from app.schemas.query import QueryFilters, StructuredQuery
from app.schemas.recommend import (
    Recommendation,
    RecommendationGroup,
    RecommendRequest,
    RecommendResponse,
    ResponseMeta,
    UnfilledSlot,
)
from app.services.catalogue import Catalogue
from app.services.context import (
    apply_climate_from_answers,
    build_climate_question,
    has_climate_answers,
    needs_place_climate,
    resolve_climate,
)
from app.services.constraints import derive_constraints
from app.services.context_slots import apply_context_audit, is_specific_trip
from app.services.explain import explain_pick
from app.services.interpreter import interpret, merge_answers, offline_interpret
from app.services.retrieval import (
    ScoredProduct,
    dedupe_across_buckets,
    is_group_worth_showing,
    sanitize_categories,
    search_bucket,
)

logger = logging.getLogger(__name__)

# A conversation stops asking after this many answered clarifying rounds and
# shows the best available recommendation instead, however imperfect. Without
# an explicit ceiling, a gap that keeps resurfacing -- a genuine new one each
# turn, or the same one mislabelled -- can keep `needs_clarification` true
# forever and the recommendation stage is never reached. Matches the "never
# ask more than 4" ceiling already used for questions within a single turn
# (see `interpreter.SYSTEM` and `StructuredQuery._cap_questions`).
MAX_CLARIFY_ANSWERS = 4

# A clarify response previews products alongside its questions -- except when
# the request is too undirected for a preview to mean anything. "Trekking
# gear, no dates yet" still gets 3-4 cohesive buckets (layering, footwear,
# navigation) worth showing; "a gift for my sister" with nothing else stated
# gets the interpreter improvising across unrelated life categories (apparel,
# jewellery, bags, beauty, home) because it has no real signal to focus on.
# Past this many buckets, showing the preview stopped being "a useful initial
# outcome" and started being 20+ weak "closest match" picks the shopper has
# to wade through before reaching the two questions that would have actually
# focused the search. The system prompt's own bucket-count guidance ("split
# into 2-5 buckets") is the source for where "focused" ends.
MAX_BUCKETS_FOR_PREVIEW = 3


def _with_overrides(inferred: QueryFilters, override: QueryFilters | None) -> QueryFilters:
    """Apply client-supplied filters on top of the interpreted ones.

    Precedence is inferred -> tapped chips -> explicit client filters, because
    an explicit filter is the only one the caller stated themselves rather than
    having derived on their behalf.

    Unset fields are skipped rather than copied: a null `price_max` in the
    request means "no opinion", not "remove the ceiling the model inferred".
    """
    if override is None:
        return inferred
    merged = inferred.model_dump()
    for field, value in override.model_dump().items():
        if value is None or value == [] or value == "":
            continue
        merged[field] = value
    return QueryFilters(**merged)


def _attach_climate(
    structured: StructuredQuery,
    answers: list[str],
    today: date,
    skip_clarification: bool,
) -> tuple[StructuredQuery, list[str]]:
    """Resolve measured conditions, or decide to ask the shopper.

    Returns the updated query and any notes to append to the response meta.
    """
    notes: list[str] = []
    ctx = structured.context

    if has_climate_answers(answers):
        structured = structured.model_copy(
            update={"context": apply_climate_from_answers(ctx, answers)}
        )
        return structured, notes

    if not needs_place_climate(ctx):
        return structured, notes

    client = OpenMeteoClient()
    try:
        climate = resolve_climate(
            ctx,
            client,
            today,
            proposed_lat=ctx.location_lat,
            proposed_lon=ctx.location_lon,
            proposed_elevation_m=float(ctx.elevation_estimate_m)
            if ctx.elevation_estimate_m is not None
            else None,
        )
    finally:
        client.close()

    if climate is None:
        return structured, notes

    structured = structured.model_copy(
        update={
            "context": ctx.model_copy(
                update={"climate": climate, "climate_note": climate.summary or ctx.climate_note}
            )
        }
    )

    if climate.source != "unobtainable" or skip_clarification:
        if climate.source == "unobtainable":
            notes.append(
                "Conditions could not be verified; ranking without temperature evidence."
            )
        return structured, notes

    # A thin request should be asked; a fully specified trek should not stall on
    # weather lookup failing for an obscure place name.
    if is_specific_trip(structured):
        notes.append(
            "Conditions could not be verified; ranking without temperature evidence."
        )
        return structured, notes

    # Named place, dates stated, but lookup failed -- ask rather than invent.
    climate_q = build_climate_question(structured.context)
    existing = [q for q in structured.questions if q.slot != "climate"]
    structured = structured.model_copy(
        update={
            "needs_clarification": True,
            "questions": ([climate_q] + existing)[:4],
        }
    )
    return structured, notes


def _to_group(
    name: str,
    why: str,
    items: list[ScoredProduct],
    limit: int,
    context: StructuredQuery,
) -> RecommendationGroup:
    top = items[:limit]
    best = max((i.score for i in top), default=1.0) or 1.0
    return RecommendationGroup(
        name=name,
        why_needed=why,
        items=[
            Recommendation(
                product=i.product,
                reason=explain_pick(i, context.context),
                match_score=round(min(1.0, max(0.0, i.score / best)), 3),
            )
            for i in top
        ],
    )


def _fresh_response(cached: RecommendResponse, *, latency_ms: int) -> RecommendResponse:
    """Return a cache hit with a new id and latency.

    Older cached payloads may predate `context_variables` / `unfilled_slots`;
    normalise so clients never see undefined optional arrays.
    """
    return cached.model_copy(
        update={
            "query_id": str(uuid.uuid4()),
            "context_variables": cached.context_variables or [],
            "unfilled_slots": cached.unfilled_slots or [],
            "groups": cached.groups or [],
            "meta": cached.meta.model_copy(
                update={"cached": True, "latency_ms": latency_ms}
            ),
        }
    )


def _cache_and_return(key: str, response: RecommendResponse) -> RecommendResponse:
    """Store a completed response, unless it was produced without full reasoning.

    Degraded answers are deliberately not cached: they are the product of a
    transient failure, and caching one would keep serving the weaker result for
    half an hour after the cause cleared.
    """
    if not response.meta.degraded_mode:
        response_cache.set(key, response)
    return response


def recommend(
    payload: RecommendRequest,
    catalogue: Catalogue,
    provider: LLMProvider | None,
    today: date | None = None,
) -> RecommendResponse:
    """Blocking variant. Drains the event stream and returns the final response."""
    final: RecommendResponse | None = None
    for event, data in recommend_events(payload, catalogue, provider, today):
        if event == "result":
            final = data
    assert final is not None, "the event stream always ends with a result"
    return final


def recommend_events(
    payload: RecommendRequest,
    catalogue: Catalogue,
    provider: LLMProvider | None,
    today: date | None = None,
) -> Iterator[tuple[str, Any]]:
    """Run the pipeline, yielding ("stage", label) as it progresses.

    A completed search takes 20-30 seconds, nearly all of it in one structured
    interpretation call plus external condition lookup. Reporting the real stage
    boundaries turns that into legible progress instead of a blank spinner --
    and because the stages are emitted where they actually happen, the progress
    cannot drift out of sync with the work.

    Always ends with ("result", RecommendResponse).
    """
    started = time.perf_counter()
    settings = get_settings()
    today = today or date.today()
    notes: list[str] = []
    llm_calls = 0
    degraded = False

    key = cache_key(payload.query, payload.answers, payload.skip_clarification)
    cached = response_cache.get(key)
    if cached is not None:
        yield "stage", "cached"
        # Returned with a fresh id and latency so the response still describes
        # *this* request, but flagged cached so the saving is visible.
        yield "result", _fresh_response(
            cached, latency_ms=int((time.perf_counter() - started) * 1000)
        )
        return

    # --- 1. Interpret -----------------------------------------------------
    yield "stage", "interpreting"
    structured: StructuredQuery
    if provider is None:
        structured = offline_interpret(payload.query, payload.answers)
        degraded = True
        notes.append("No LLM configured; used keyword interpretation.")
    else:
        try:
            structured = interpret(
                provider,
                settings.interpret_model,
                payload.query,
                today,
                payload.answers,
                timeout_s=settings.interpret_timeout_s,
                effort=settings.interpret_effort,
                taxonomy=get_taxonomy(),
            )
            llm_calls += 1
        except LLMUnavailable as exc:
            logger.warning("Interpretation failed, degrading: %s", exc)
            structured = offline_interpret(payload.query, payload.answers)
            degraded = True
            notes.append("AI interpretation unavailable; fell back to keyword matching.")

    if payload.answers:
        structured = merge_answers(structured, payload.answers)

    if payload.filters is not None:
        structured = structured.model_copy(
            update={"filters": _with_overrides(structured.filters, payload.filters)}
        )

    # Sanitized once here rather than inside passes_filters(): filters are
    # shared across every bucket's search, so this only needs doing once per
    # request, not once per candidate product.
    structured = structured.model_copy(
        update={"filters": sanitize_categories(structured.filters)}
    )

    # --- 1b. Resolve conditions (Open-Meteo, no LLM) -----------------------
    yield "stage", "checking conditions"
    structured, climate_notes = _attach_climate(
        structured, payload.answers, today, payload.skip_clarification
    )
    notes.extend(climate_notes)

    structured, context_variables = apply_context_audit(structured, payload.answers, today)

    def elapsed() -> int:
        return int((time.perf_counter() - started) * 1000)

    # --- 2. Decline politely rather than inventing results ----------------
    if not structured.is_shopping_request:
        yield "result", RecommendResponse(
            query_id=str(uuid.uuid4()),
            mode="results",
            intent_summary=(
                "That doesn't look like a shopping request — tell me what you're "
                "looking for and I'll find it."
            ),
            meta=ResponseMeta(
                latency_ms=elapsed(), llm_calls=llm_calls,
                degraded_mode=degraded, catalogue_size=len(catalogue), notes=notes,
            ),
        )
        return

    # --- 3. Explicit stop condition for clarification ----------------------
    # However many rounds have already been answered, don't ask forever:
    # past the ceiling, proceed with whatever is known.
    if structured.needs_clarification and len(payload.answers) >= MAX_CLARIFY_ANSWERS:
        notes.append(
            "Reached the clarification limit; showing the best matches from "
            "what's known so far."
        )
        structured = structured.model_copy(
            update={"needs_clarification": False, "questions": []}
        )

    # --- 4. Retrieve, once per bucket ---------------------------------------
    # Runs whether or not we are also about to ask a follow-up: a request
    # should never end a turn with only a question and no products when the
    # catalogue can already offer something against what's known so far.
    yield "stage", "searching"
    embedder = get_embedder()
    if not embedder.is_semantic:
        degraded = True
        notes.append("Semantic model unavailable; matching on literal wording only.")

    constraints = derive_constraints(structured)

    per_bucket: dict[str, list[ScoredProduct]] = {}
    for bucket in structured.buckets:
        # The bucket name is appended as a fallback phrase so a bucket whose
        # phrases all miss still has something to match on.
        phrases = [*bucket.search_phrases, bucket.name]
        per_bucket[bucket.name] = search_bucket(
            catalogue,
            embedder.embed(phrases),
            bucket,
            structured.filters,
            structured.context,
            limit=settings.candidates_per_bucket,
            constraints=constraints,
        )

    per_bucket = dedupe_across_buckets(per_bucket)

    shown = {
        b.name: per_bucket[b.name]
        for b in structured.buckets
        if is_group_worth_showing(per_bucket.get(b.name, []))
    }

    # Anything the planner asked for that the catalogue could not cover is
    # recorded explicitly. The two causes read differently to a user: a slot
    # with no catalogue path at all means we stock nothing of that type, while
    # an empty result means we stock the type but nothing matched the request.
    unfilled: list[UnfilledSlot] = []
    for bucket in structured.buckets:
        if bucket.name in shown:
            continue
        if not bucket.catalogue_paths:
            reason = "this catalogue doesn't stock that type of product yet"
        else:
            reason = "nothing in stock matched closely enough"
        unfilled.append(
            UnfilledSlot(name=bucket.name, role=bucket.role, reason=reason)
        )

    missing_required = [u for u in unfilled if u.role == "required"]
    if missing_required:
        notes.append(
            "Could not cover: "
            + ", ".join(u.name for u in missing_required)
            + " — so this is not a complete answer to the request."
        )

    groups = []
    for bucket in structured.buckets:
        candidates = shown.get(bucket.name)
        if not candidates:
            continue
        groups.append(
            _to_group(bucket.name, bucket.why_needed, candidates, bucket.max_items, structured)
        )

    if not groups:
        notes.append("No products matched the filters; try relaxing budget or category.")

    # --- 5. Ask, if asking would still change the answer --------------------
    # Retrieval already ran above, so a clarify response carries whatever
    # products are already good matches alongside the follow-up questions --
    # a turn never ends with only a question and no recommendation when the
    # request was focused enough for that preview to be useful. A request
    # broad enough to spread across many buckets gets questions only; the
    # preview would be scattered "closest match" filler, not a real answer.
    if structured.needs_clarification and not payload.skip_clarification:
        focused = len(structured.buckets) <= MAX_BUCKETS_FOR_PREVIEW
        yield "result", _cache_and_return(key, RecommendResponse(
            query_id=str(uuid.uuid4()),
            mode="clarify",
            intent_summary=structured.intent_summary,
            context=structured.context,
            assumptions=structured.assumptions,
            context_variables=context_variables,
            questions=structured.questions,
            groups=groups if focused else [],
            unfilled_slots=unfilled if focused else [],
            meta=ResponseMeta(
                latency_ms=elapsed(), llm_calls=llm_calls,
                degraded_mode=degraded, catalogue_size=len(catalogue), notes=notes,
            ),
        ))
        return

    yield "result", _cache_and_return(
        key,
        RecommendResponse(
            query_id=str(uuid.uuid4()),
            mode="results",
            intent_summary=structured.intent_summary,
            context=structured.context,
            assumptions=structured.assumptions,
            context_variables=context_variables,
            groups=groups,
            unfilled_slots=unfilled,
            meta=ResponseMeta(
                latency_ms=elapsed(),
                llm_calls=llm_calls,
                degraded_mode=degraded,
                catalogue_size=len(catalogue),
                notes=notes,
            ),
        ),
    )
