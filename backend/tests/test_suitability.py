"""Suitability evaluation tests.

Pin the three severities directly: which combinations veto (hard), which
merely sink (strong), and which only reorder (soft) -- plus the rule that
carries the whole module: missing evidence on the product is never treated
as a conflict.
"""

from app.schemas.query import ContextConstraints
from app.services.suitability import evaluate

from tests.test_retrieval import make_product


def test_no_constraints_is_always_neutral():
    product = make_product("p", attributes={"water_resistance": "none", "formality": "casual"})
    verdict = evaluate(product, ContextConstraints())
    assert not verdict.hard_mismatch
    assert verdict.strong_penalty == 0
    assert verdict.soft_boost == 0


def test_unevaluated_product_is_never_penalised_even_under_strict_constraints():
    """None on the product means 'not evaluated', not 'fails'. A pre-enrichment
    or out-of-scope product must not be vetoed or sunk just because the axis
    was never checked -- the same rule temperature_fit() and
    passes_occasion_context() already follow."""
    product = make_product("p")  # attributes defaults: water_resistance=None, formality=None
    constraints = ContextConstraints(min_water_resistance="waterproof", required_formality="formal")
    verdict = evaluate(product, constraints)
    assert not verdict.hard_mismatch
    assert verdict.strong_penalty == 0
    assert verdict.soft_boost == 0


def test_no_water_resistance_against_waterproof_requirement_is_hard():
    product = make_product("p", attributes={"water_resistance": "none"})
    verdict = evaluate(product, ContextConstraints(min_water_resistance="waterproof"))
    assert verdict.hard_mismatch
    assert verdict.reasons


def test_no_water_resistance_against_light_rain_is_only_strong():
    """The strict veto is reserved for the strict requirement -- light rain
    with an unprotected garment sinks the candidate but does not exclude it."""
    product = make_product("p", attributes={"water_resistance": "none"})
    verdict = evaluate(product, ContextConstraints(min_water_resistance="repellent"))
    assert not verdict.hard_mismatch
    assert verdict.strong_penalty > 0


def test_waterproof_product_gets_a_soft_boost_for_wet_conditions():
    product = make_product("p", attributes={"water_resistance": "waterproof"})
    verdict = evaluate(product, ContextConstraints(min_water_resistance="waterproof"))
    assert not verdict.hard_mismatch
    assert verdict.strong_penalty == 0
    assert verdict.soft_boost > 0


def test_repellent_product_is_soft_penalised_against_waterproof_requirement():
    """Repellent is real evidence, just not enough for the strict tier --
    ordering only, never a veto or heavy penalty."""
    product = make_product("p", attributes={"water_resistance": "repellent"})
    verdict = evaluate(product, ContextConstraints(min_water_resistance="waterproof"))
    assert not verdict.hard_mismatch
    assert verdict.strong_penalty == 0
    assert verdict.soft_boost < 0


def test_casual_product_against_formal_requirement_is_strong_never_hard():
    """Formality never reaches hard -- it is judgement-prone in a way rain
    protection is not, so the worst outcome is a heavy penalty, not a veto."""
    product = make_product("p", attributes={"formality": "casual"})
    verdict = evaluate(product, ContextConstraints(required_formality="formal"))
    assert not verdict.hard_mismatch
    assert verdict.strong_penalty > 0


def test_smart_casual_against_formal_requirement_is_only_soft():
    product = make_product("p", attributes={"formality": "smart_casual"})
    verdict = evaluate(product, ContextConstraints(required_formality="formal"))
    assert verdict.strong_penalty == 0
    assert verdict.soft_boost < 0


def test_overqualified_formality_is_never_penalised():
    """A formal item shown for a casual-leaning request is not wrong -- there
    is nothing to object to in owning something nicer than required."""
    product = make_product("p", attributes={"formality": "formal"})
    verdict = evaluate(product, ContextConstraints(required_formality="casual"))
    assert not verdict.hard_mismatch
    assert verdict.strong_penalty == 0
    assert verdict.soft_boost == 0


def test_matching_formality_gets_a_small_soft_boost():
    product = make_product("p", attributes={"formality": "formal"})
    verdict = evaluate(product, ContextConstraints(required_formality="formal"))
    assert verdict.soft_boost > 0


def test_both_axes_can_combine_in_one_verdict():
    product = make_product("p", attributes={"water_resistance": "none", "formality": "casual"})
    constraints = ContextConstraints(min_water_resistance="repellent", required_formality="formal")
    verdict = evaluate(product, constraints)
    assert not verdict.hard_mismatch
    assert verdict.strong_penalty > 0.5  # both axes contributed
    assert len(verdict.reasons) == 2
