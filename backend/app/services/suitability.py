"""Check one product against the request's explicit requirement set.

Second half of the generalized suitability mechanism: `constraints.py` says
what the trip needs; this module says whether a candidate meets it, at one of
three severities:

    hard    veto before ranking -- excluded from the bucket entirely
    strong  heavy multiplicative penalty -- can still surface, but sinks
    soft    ordering only -- ties get broken, nothing is excluded or sunk

Water resistance is the only axis that can reach `hard`, and only at its
strictest tier (a trip that genuinely needs waterproof gear, matched against a
product explicitly marked as offering none). Formality tops out at `strong`:
formality is judgement-prone in a way "will this keep me dry" is not, and a
wrong hard veto there is a worse failure than a wrong strong penalty.

Breathability is comfort, not compatibility -- it never appears here at all,
only as a soft evidence string in retrieval's existing scoring.

Both severities key off *explicit* product evidence. `None` on a product
attribute means "not evaluated," and missing evidence is never treated as a
conflict -- the same rule `retrieval.temperature_fit()` and
`passes_occasion_context()` already follow.
"""

from app.schemas.product import Product
from app.schemas.query import ContextConstraints

# Multiplied against PENALTY_SUITABILITY_STRONG in retrieval.py, same shape as
# PENALTY_THERMAL_MISMATCH: a fraction of the max penalty, not the raw score.
# Both strong cases below use the same fraction -- neither is more severe than
# the other, unlike the hard/strong split itself, which is a real ordering.
STRONG_PENALTY = 0.5

_FORMALITY_ORDER = {"casual": 0, "smart_casual": 1, "formal": 2}


class Verdict:
    __slots__ = ("hard_mismatch", "strong_penalty", "soft_boost", "reasons")

    def __init__(
        self,
        hard_mismatch: bool = False,
        strong_penalty: float = 0.0,
        soft_boost: float = 0.0,
        reasons: list[str] | None = None,
    ):
        self.hard_mismatch = hard_mismatch
        self.strong_penalty = strong_penalty
        self.soft_boost = soft_boost
        self.reasons = reasons or []


def _evaluate_water_resistance(product: Product, constraints: ContextConstraints) -> Verdict:
    required = constraints.min_water_resistance
    actual = product.attributes.water_resistance
    if required is None or actual is None:
        return Verdict()

    if actual == "none":
        if required == "waterproof":
            # The strictest requirement, matched against an explicit "offers
            # nothing" -- this is the one case allowed to veto rather than
            # merely sink, because no amount of semantic relevance makes an
            # unprotected garment the right pick for genuinely wet conditions.
            return Verdict(
                hard_mismatch=True,
                reasons=["has no rain protection, but conditions call for it"],
            )
        return Verdict(
            strong_penalty=STRONG_PENALTY,
            reasons=["no rain protection noted for a trip expecting some rain"],
        )

    if actual == "repellent":
        if required == "waterproof":
            return Verdict(
                soft_boost=-0.05,
                reasons=["water-repellent but not fully waterproof for the expected rain"],
            )
        return Verdict(soft_boost=0.1, reasons=["water-repellent, suits light rain"])

    # actual == "waterproof"
    return Verdict(soft_boost=0.15, reasons=["fully waterproof"])


def _evaluate_formality(product: Product, constraints: ContextConstraints) -> Verdict:
    required = constraints.required_formality
    actual = product.attributes.formality
    if required is None or actual is None:
        return Verdict()

    gap = _FORMALITY_ORDER[required] - _FORMALITY_ORDER[actual]
    if gap >= 2:
        return Verdict(strong_penalty=STRONG_PENALTY, reasons=["too casual for the occasion"])
    if gap == 1:
        return Verdict(soft_boost=-0.05, reasons=["a little casual for the occasion"])
    if gap == 0:
        return Verdict(soft_boost=0.08, reasons=["matches the occasion's formality"])
    # gap < 0: the product is dressier than strictly required. Never
    # penalised -- there is nothing wrong with a shopper owning something
    # nicer than the floor they asked for.
    return Verdict()


def evaluate(product: Product, constraints: ContextConstraints) -> Verdict:
    """Combine every suitability axis into one verdict for this product.

    A hard mismatch on any axis wins outright. Otherwise strong penalties
    accumulate and the largest soft adjustment leads the reasons, since the
    caller (retrieval.score_product) only shows the top evidence anyway.
    """
    verdicts = [
        _evaluate_water_resistance(product, constraints),
        _evaluate_formality(product, constraints),
    ]

    hard = any(v.hard_mismatch for v in verdicts)
    strong = sum(v.strong_penalty for v in verdicts)
    soft = sum(v.soft_boost for v in verdicts)
    reasons = [reason for v in verdicts for reason in v.reasons]

    return Verdict(hard_mismatch=hard, strong_penalty=strong, soft_boost=soft, reasons=reasons)
