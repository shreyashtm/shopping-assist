"""Deterministic product explanations.

The ranker wrote one sentence per pick. That prose was the visible quality of
the product, but it was also the most expensive call and the one most likely to
invent a specification. Retrieval already produces grounded evidence strings --
"rated to -10C, covering the -15C nights at your dates" -- and this module
composes them into the user-visible reason without a second model call.

Every clause must trace to a matched attribute or a measured number. Fluency is
secondary to traceability.

Only the two strongest clauses are shown, so which two those are is decided
upstream: `retrieval.score_product` sorts its evidence by how much each clause
actually tells a shopper before handing it over. This module trusts that order
and does not re-rank.
"""

from app.schemas.query import ResolvedContext
from app.services.retrieval import ScoredProduct

# Two clauses is the point where a product card still scans in one glance.
MAX_CLAUSES = 2


def _clean(reason: str) -> str:
    return reason.strip().rstrip(".")


def _sentence_case(text: str) -> str:
    """Capitalise the opening letter only.

    Deliberately not `.capitalize()`, which would lowercase the rest and turn
    "rated to -10C" into "Rated to -10c".
    """
    return text[0].upper() + text[1:] if text and text[0].islower() else text


def _fallback(scored: ScoredProduct) -> str:
    """Used when a product carries no citable evidence at all.

    Reached only when nothing was tagged, no material was recorded and the
    listing has too few reviews to quote -- so the honest thing to say is that
    the match came from the description, and nothing more.
    """
    return f"Closest match in {scored.product.subcategory.lower()} for what you described."


def explain_pick(scored: ScoredProduct, context: ResolvedContext) -> str:
    """One explanation assembled from retrieval evidence."""
    clauses = [
        _clean(reason)
        for reason in scored.reasons[:MAX_CLAUSES]
        if reason and reason.strip()
    ]

    if not clauses:
        return _fallback(scored)

    # Only the first clause is capitalised: the rest continue the same sentence
    # after a semicolon, where a capital would read as a new one.
    lead = _sentence_case(clauses[0])
    if len(clauses) == 1:
        return f"{lead}."
    return f"{lead}; {clauses[1]}."
