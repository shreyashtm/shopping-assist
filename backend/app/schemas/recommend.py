"""Request and response contract for POST /api/v1/recommend."""

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.product import Product
from app.schemas.query import ClarifyingQuestion, ContextVariable, QueryFilters, ResolvedContext


class RecommendRequest(BaseModel):
    query: str = Field(min_length=3, max_length=1000)
    filters: QueryFilters | None = Field(
        default=None,
        description="Explicit constraints, applied on top of the interpreted ones. "
        "Precedence is inferred -> tapped answers -> these. Unset fields mean "
        "'no opinion' and leave the inferred value in place.",
    )
    answers: list[str] = Field(
        default_factory=list,
        description="Machine values from clarifying-question chips the user tapped, e.g. "
        "['occasion:daily-wear', 'price_max:500']. These merge into the interpreted query "
        "deterministically -- no second interpretation call is needed to read them.",
    )
    skip_clarification: bool = Field(
        default=False,
        description="Set by the 'just show me now' escape hatch. Forces results using "
        "assumed defaults rather than asking anything.",
    )


class Recommendation(BaseModel):
    product: Product
    reason: str = Field(description="Why this item fits *this* request, citing request details.")
    match_score: float = Field(ge=0, le=1)


class RecommendationGroup(BaseModel):
    name: str
    why_needed: str
    items: list[Recommendation]


class UnfilledSlot(BaseModel):
    """A need the catalogue could not cover.

    Reported rather than hidden. The alternative -- filling the slot with the
    nearest available product -- is how a request for a wedding suit came back
    with hiking boots and a utility jacket.
    """

    name: str
    role: str
    reason: str = Field(description="Plain-language explanation, shown to the user.")


class ResponseMeta(BaseModel):
    """Operational truth about how the answer was produced.

    `degraded_mode` is deliberately part of the public contract: when the LLM
    path fails the app still answers from vector search, and the UI says so
    rather than passing off weaker results as full-quality ones.
    """

    latency_ms: int
    llm_calls: int = 0
    cached: bool = False
    degraded_mode: bool = False
    catalogue_size: int = 0
    notes: list[str] = Field(default_factory=list)


class RecommendResponse(BaseModel):
    """Either a set of recommendations, or the questions needed to produce them.

    The two modes share one endpoint because they are two states of the same
    request, not two different operations: `results` carries groups and no
    questions. `clarify` carries questions, and also carries `groups` when
    retrieval already found something worth showing against what's known so
    far -- a turn should not end with only a question when the catalogue can
    already offer a real recommendation. The client re-posts the original
    query plus the tapped answers to move from one to the other.
    """

    query_id: str
    mode: Literal["results", "clarify"] = "results"

    intent_summary: str
    context: ResolvedContext = Field(default_factory=ResolvedContext)
    assumptions: list[str] = Field(default_factory=list)

    context_variables: list[ContextVariable] = Field(
        default_factory=list,
        description="What we know vs still need about the trip or shopper.",
    )

    questions: list[ClarifyingQuestion] = Field(
        default_factory=list, description="Populated only when mode is 'clarify'."
    )
    groups: list[RecommendationGroup] = Field(
        default_factory=list, description="Populated only when mode is 'results'."
    )
    unfilled_slots: list[UnfilledSlot] = Field(
        default_factory=list,
        description="Needs the catalogue could not cover. A non-empty list with a "
        "'required' role means the request was not fully satisfied, and the UI says so.",
    )
    meta: ResponseMeta
