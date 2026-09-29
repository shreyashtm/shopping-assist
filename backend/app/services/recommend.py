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
from app.adapters.weather.nominatim import NominatimClient
from app.adapters.weather.open_meteo import OpenMeteoClient
from app.core.cache import cache_key, response_cache
from app.core.config import get_settings
from app.core.deps import get_taxonomy
from app.schemas.query import Bucket, ClarifyingQuestion, QueryFilters, StructuredQuery
from app.schemas.recommend import (
    Recommendation,
    RecommendationGroup,
    RecommendRequest,
    RecommendResponse,
    ResponseMeta,
    UnfilledSlot,
)
from app.services.catalogue import Catalogue
from app.services.constraints import derive_constraints
from app.services.context import (
    apply_climate_from_answers,
    build_climate_question,
    has_climate_answers,
    needs_place_climate,
    resolve_climate,
)
from app.services.context_slots import (
    apply_context_audit,
    has_budget_answers,
    has_date_answers,
    is_specific_trip,
    stated_dates,
    stated_money,
    stated_place,
    stated_wearer,
    states_only_ceiling,
)
from app.services.explain import explain_pick
from app.services.interpreter import (
    drop_unsatisfiable_budget_options,
    interpret,
    merge_answers,
    offline_interpret,
)
from app.services.offline import stated_filters
from app.services.retrieval import (
    ScoredProduct,
    dedupe_across_buckets,
    is_group_worth_showing,
    sanitize_categories,
    search_bucket,
)
from app.services.taxonomy import ALL_PATHS

logger = logging.getLogger(__name__)

# A conversation stops asking after this many answered clarifying rounds and
# shows the best available recommendation instead, however imperfect. Without
# an explicit ceiling, a gap that keeps resurfacing -- a genuine new one each
# turn, or the same one mislabelled -- can keep `needs_clarification` true
# forever and the recommendation stage is never reached. Matches the "never
# ask more than 4" ceiling already used for questions within a single turn
# (see `interpreter.SYSTEM` and `StructuredQuery._cap_questions`).
MAX_CLARIFY_ANSWERS = 4
# The model plans how many of a thing the shopper needs ("a top" -> 1), but
# a group is a set of options to choose between. Live, max_items=1 left
# "women's jeans and a top for college" with one pair of jeans and one top.
MIN_OPTIONS_PER_GROUP = 3

# A clarify response previews products alongside its questions. The risk this
# guards against is real: "a gift for my sister" with nothing else stated gets
# the interpreter improvising across unrelated life categories (apparel,
# jewellery, bags, beauty, home) because it has no real signal to focus on,
# and showing all of that would be 20+ weak "closest match" picks the shopper
# has to wade through before reaching the questions that would have actually
# focused the search.
#
# This used to gate on the request's *total* bucket count -- past this many
# buckets, no preview at all. That excluded exactly the requests this app is
# for: "trekking Hampta Pass" or "monsoon wear for Mumbai" routinely plan 5-8
# cohesive buckets (layering, footwear, navigation, rain protection, ...),
# so the broadest, most effort-justifying requests got the worst possible
# result -- a completely empty first turn. `_select_preview_buckets` now
# caps *how many buckets are shown*, not *whether any are*: it always picks
# the MAX_BUCKETS_FOR_PREVIEW most important ones (required role first, then
# declared priority) regardless of how many total buckets exist, so a request
# never gets excluded from a preview for being thorough.
MAX_BUCKETS_FOR_PREVIEW = 3


def _select_preview_buckets(buckets: list[Bucket], limit: int) -> list[Bucket]:
    """The buckets worth previewing before questions are answered, or none.

    A request under `limit` buckets previews all of them, unchanged from
    before. Past that, previewing is worth doing only when the request has
    genuine structure to prioritise by -- at least one `role="required"`
    bucket. Without that signal, "top N by priority" is not actually
    meaningful: "a gift for my sister" with no other detail produces buckets
    across apparel, jewellery, bags and beauty that are all `role="optional"`
    at the same priority, because the interpreter has no real basis to rank
    them -- it is guessing across unrelated categories, not planning a kit.
    Showing "the first 3 guesses" there is not a smaller version of a useful
    preview, it is still filler, just less of it. That case gets no preview
    at all and goes straight to the question.
    """
    if len(buckets) <= limit:
        return buckets
    if not any(b.role == "required" for b in buckets):
        return []

    role_rank = {"required": 0, "recommended": 1, "optional": 2}
    ranked = sorted(buckets, key=lambda b: (role_rank.get(b.role, 3), b.priority))
    return ranked[:limit]


def _next_year(day: date, today: date) -> date:
    """`day` moved forward whole years until it is no longer past."""
    while day < today:
        try:
            day = day.replace(year=day.year + 1)
        except ValueError:  # 29 February
            day = day.replace(year=day.year + 1, day=28)
    return day


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
    places = NominatimClient()
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
            places=places,
        )
    finally:
        client.close()
        places.close()

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
    """Store a completed response, unless it looks transient.

    Two kinds of answer are deliberately not cached, for the same reason:
    caching one keeps serving it long after the cause has cleared.

    **Degraded answers** are the product of a provider failure.

    **Results that filled nothing** are the product of an unlucky plan. Seen in
    production on the flagship trek query: the model produced buckets nothing
    could fill, returning 0 groups and 9 unfilled slots -- and rewording the
    same request returned 9 populated groups, so the catalogue was never the
    problem. Because the empty answer was cached, every later run of that exact
    wording served it back, turning one bad roll of the dice into a permanent
    wrong answer.

    A clarify turn is exempt: it legitimately has no groups yet because it is
    asking a question, not failing to answer one.
    """
    filled_nothing = (
        response.mode == "results"
        and not response.groups
        and bool(response.unfilled_slots)
    )
    if response.meta.degraded_mode or filled_nothing:
        return response
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
            # One line per plan: model output varies between calls, and twice an
            # intermittent result (an empty page, a single card) could only be
            # diagnosed by replaying the request to see what had been planned.
            logger.info(
                "Plan: %s | filters=%s",
                "; ".join(
                    f"{b.name} {b.catalogue_paths} x{b.max_items}" for b in structured.buckets
                ),
                structured.filters.model_dump(exclude_defaults=True),
            )
        except LLMUnavailable as exc:
            logger.warning("Interpretation failed, degrading: %s", exc)
            structured = offline_interpret(payload.query, payload.answers)
            degraded = True
            notes.append("AI interpretation unavailable; fell back to keyword matching.")

    # Dates the shopper never gave are the model's invention, not a fact about
    # the trip. Live, "heels and a blazer for an office party" showed "Dates ·
    # 2026-09-29 (you)": a one-day trip on today's date. Dropping them lets the
    # audit ask when a date genuinely matters, instead of guessing.
    ctx = structured.context
    if ctx.start_date and not stated_dates(payload.query) and not has_date_answers(payload.answers):
        structured = structured.model_copy(update={"context": ctx.model_copy(
            update={"start_date": None, "end_date": None, "duration_days": None})})

    # A trip is planned ahead, so a date already past means the model picked
    # the wrong year. Live, qwen3:8b read "Leh ... in January" on 29 Sep 2026
    # as January 2026, and the page showed 3C nights for Leh in winter.
    ctx = structured.context
    if ctx.start_date and ctx.start_date < today:
        start, end = _next_year(ctx.start_date, today), None
        if ctx.end_date:
            end = start + (ctx.end_date - ctx.start_date)
        structured = structured.model_copy(update={"context": ctx.model_copy(
            update={"start_date": start, "end_date": end})})

    # Same for price. Live, qwen3:8b gave a Hampta Pass trek that named no
    # budget a ₹2,000-40,000 window, and "a hamper under 5000" a ₹2,500
    # floor, each silently hiding products the shopper could have bought.
    # And "budget 8000" came back as ₹2,320-12,487, the suit shelf's price
    # range copied from the prompt. When the words give an amount the parser
    # can read, those words win over the model's numbers.
    filters = structured.filters
    if not has_budget_answers(payload.answers):
        stated = stated_filters(payload.query.lower())
        update: dict[str, int | None] = {}
        if stated.price_min or stated.price_max:
            update = {"price_min": stated.price_min, "price_max": stated.price_max}
        elif not stated_money(payload.query):
            update = {"price_min": None, "price_max": None}
        elif filters.price_min and states_only_ceiling(payload.query):
            update = {"price_min": None}
        if update and any(getattr(filters, k) != v for k, v in update.items()):
            structured = structured.model_copy(update={"filters": filters.model_copy(update=update)})

    # And for place and wearer. Live, qwen3:8b set "India" for "something
    # warm for winter" (then warned "add a date for India") and "N/A (no
    # location provided)" for office wear, which became a page heading; and
    # showed "For · unisex (you)" when the shopper had said nothing of the kind.
    ctx = structured.context
    if ctx.location and not stated_place(ctx.location, payload.query):
        structured = structured.model_copy(update={"context": ctx.model_copy(update={
            "location": None, "location_lat": None, "location_lon": None,
            "elevation_estimate_m": None})})
    # With no place there is no weather lookup, so any climate note is the
    # model's own: "No specific climate information provided.", or "Winter in
    # India typically ranges from 5C to 15C" -- a number no source backs.
    ctx = structured.context
    if not ctx.location and ctx.climate_note:
        structured = structured.model_copy(update={"context": ctx.model_copy(update={"climate_note": None})})
    filters = structured.filters
    if (filters.gender or "").lower() in {"unisex", "both"} and not stated_wearer(payload.query):
        structured = structured.model_copy(update={"filters": filters.model_copy(update={"gender": None})})

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

    # Budget chips are checked against real catalogue prices here rather than
    # inside interpret(), because this is where the two sources of questions
    # converge: model-generated ones and the deterministic fallbacks
    # context_slots.py appends for unfilled gaps. The reported defect came from
    # the latter -- a hardcoded "Premium (Rs3,000+)" chip offered against
    # women's ethnic wear, where the dearest product is Rs 1,955 -- so a guard
    # that only inspected model output would have missed it entirely.
    if structured.questions:
        # Checked against the *required* buckets only. Using every bucket's
        # paths made the guard too lenient: for "traditional wear" the model
        # also plans an accessories bucket holding a Rs 9,684 potli clutch, so
        # a "Rs 3,000+" chip looked satisfiable against that union while the
        # outfit itself -- the thing actually being asked for -- topped out at
        # Rs 1,955. Tapping it still returned zero groups. A budget the
        # required need cannot meet is not a budget worth offering.
        required = [b for b in structured.buckets if b.role == "required"]
        proposed_paths = [
            path for bucket in (required or structured.buckets) for path in bucket.catalogue_paths
        ]
        surviving = drop_unsatisfiable_budget_options(
            [q.model_dump() for q in structured.questions], proposed_paths, get_taxonomy()
        )
        structured = structured.model_copy(
            update={
                "questions": [ClarifyingQuestion.model_validate(q) for q in surviving],
                "needs_clarification": bool(surviving) and structured.needs_clarification,
            }
        )

    def elapsed() -> int:
        return int((time.perf_counter() - started) * 1000)

    # --- 2. Decline politely rather than inventing results ----------------
    if not structured.is_shopping_request:
        yield "result", RecommendResponse(
            query_id=str(uuid.uuid4()),
            mode="results",
            declined=True,
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
            request_text=payload.query,
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
        elif not any(path in ALL_PATHS for path in bucket.catalogue_paths):
            # A planning error, not a stock gap: the shelves named do not exist.
            reason = "couldn't match this to a section of the catalogue"
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
            _to_group(
                bucket.name, bucket.why_needed, candidates,
                max(bucket.max_items, MIN_OPTIONS_PER_GROUP), structured,
            )
        )

    if not groups:
        notes.append("No products matched the filters; try relaxing budget or category.")

    # --- 5. Ask, if asking would still change the answer --------------------
    # Retrieval already ran above, so a clarify response carries whatever
    # products are already good matches alongside the follow-up questions --
    # a turn never ends with only a question and no recommendation.
    #
    # Previously this was all-or-nothing: a request spanning more than
    # MAX_BUCKETS_FOR_PREVIEW buckets got questions only, on the theory that a
    # scattered preview across many buckets would read as filler rather than a
    # real answer. In practice that excluded exactly the requests this app
    # exists for -- a trekking kit or a monsoon wardrobe routinely plans 5-8
    # buckets, so the broadest, highest-effort requests were the ones that got
    # *zero* preview, the worst possible outcome for the queries most worth
    # answering well.
    #
    # The fix keeps the same instinct -- don't show scattered filler -- but
    # applies it per bucket instead of per request: preview only the
    # highest-priority buckets (required role first, then declared priority),
    # capped at MAX_BUCKETS_FOR_PREVIEW, rather than refusing to preview any
    # of them once the total crosses that count. `questions` still covers the
    # whole request regardless of how much of it gets previewed.
    if structured.needs_clarification and not payload.skip_clarification:
        preview_buckets = _select_preview_buckets(structured.buckets, MAX_BUCKETS_FOR_PREVIEW)
        preview_names = {b.name for b in preview_buckets}
        preview_groups = [g for g in groups if g.name in preview_names]
        preview_unfilled = [u for u in unfilled if u.name in preview_names]
        yield "result", _cache_and_return(key, RecommendResponse(
            query_id=str(uuid.uuid4()),
            mode="clarify",
            intent_summary=structured.intent_summary,
            context=structured.context,
            assumptions=structured.assumptions,
            context_variables=context_variables,
            questions=structured.questions,
            groups=preview_groups,
            unfilled_slots=preview_unfilled,
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
