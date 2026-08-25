"""Turn free text into a StructuredQuery. This is LLM call 1.

The whole product rests on this step. "Hampta Pass, last week of October, one
week" contains no product vocabulary at all -- no brand, no category, no
keyword that appears in any listing. What it contains is a *situation*, and the
job here is to unfold that situation into the things a shopper actually needs:

    Hampta Pass  -> Himalayan crossing, ~4,200m
    late October -> pre-winter, sub-zero nights, early snow
    one week     -> multi-day pack, odour-resistant layers, spare warmth

Only after that expansion does the request become searchable. Retrieval never
sees the raw sentence; it sees the buckets produced here.

The same call also decides whether to ask anything back. That decision is
adaptive on purpose: a request that already states the trip, the dates and the
duration should go straight to results, while "suggest me some t-shirts" should
not be answered with a guess.
"""

import logging
from datetime import date, timedelta
from typing import Any

from app.adapters.llm.base import LLMProvider
from app.schemas.query import StructuredQuery
from app.services.taxonomy import OCCASIONS

logger = logging.getLogger(__name__)

_GENDER_ALIASES = {
    "male": "men",
    "man": "men",
    "men": "men",
    "female": "women",
    "woman": "women",
    "women": "women",
    "unisex": "unisex",
    "anyone": "unisex",
    "neutral": "unisex",
    "any": "unisex",
    "kids": "kids",
    "children": "kids",
}

# Hand-written rather than derived from the Pydantic model: model_json_schema()
# emits $defs/$ref, which the structured-output validator rejects.
QUERY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent_summary": {"type": "string"},
        "is_shopping_request": {"type": "boolean"},
        "confidence": {"type": "number", "description": "0.0 to 1.0."},
        "needs_clarification": {"type": "boolean"},
        "questions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "slot": {"type": "string"},
                    "question": {"type": "string"},
                    "options": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "label": {"type": "string"},
                                "value": {"type": "string"},
                            },
                            "required": ["label", "value"],
                            "additionalProperties": False,
                        },
                    },
                    "allow_multiple": {"type": "boolean"},
                },
                "required": ["slot", "question", "options", "allow_multiple"],
                "additionalProperties": False,
            },
        },
        "buckets": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "search_phrases": {"type": "array", "items": {"type": "string"}},
                    "why_needed": {"type": "string"},
                    "role": {
                        "type": "string",
                        "enum": ["required", "recommended", "optional"],
                    },
                    "catalogue_paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Category/Subcategory paths from the supplied "
                        "taxonomy. Empty when the catalogue has nothing of this type.",
                    },
                    "priority": {
                        "type": "integer",
                        "description": "1 = essential, 2 = useful, 3 = nice to have.",
                    },
                    "max_items": {"type": "integer", "description": "How many to show, 1 to 8."},
                },
                "required": [
                    "name", "search_phrases", "why_needed", "role",
                    "catalogue_paths", "priority", "max_items",
                ],
                "additionalProperties": False,
            },
        },
        "filters": {
            "type": "object",
            "properties": {
                "price_min": {"type": ["integer", "null"]},
                "price_max": {"type": ["integer", "null"]},
                "gender": {"type": ["string", "null"]},
                "categories": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["price_min", "price_max", "gender", "categories"],
            "additionalProperties": False,
        },
        "context": {
            "type": "object",
            "properties": {
                "location": {"type": ["string", "null"]},
                "location_lat": {"type": ["number", "null"]},
                "location_lon": {"type": ["number", "null"]},
                "elevation_estimate_m": {"type": ["integer", "null"]},
                "start_date": {"type": ["string", "null"]},
                "end_date": {"type": ["string", "null"]},
                "duration_days": {"type": ["integer", "null"]},
                "climate_note": {"type": ["string", "null"]},
                "recipient": {"type": ["string", "null"]},
            },
            "required": [
                "location", "location_lat", "location_lon", "elevation_estimate_m",
                "start_date", "end_date", "duration_days", "climate_note", "recipient",
            ],
            "additionalProperties": False,
        },
        "assumptions": {"type": "array", "items": {"type": "string"}},
    },
    "required": [
        "intent_summary", "is_shopping_request", "confidence", "needs_clarification",
        "questions", "buckets", "filters", "context", "assumptions",
    ],
    "additionalProperties": False,
}

SYSTEM = """You are the interpretation stage of an Indian shopping assistant.

You receive a shopping request in plain English and turn it into a structured
search plan. You never see products and never recommend anything -- a later
stage does that from a real catalogue.

## Expanding intent

Requests describe situations, not products. Unfold the situation into what it
actually demands, using real-world knowledge:

- A named place implies terrain and altitude. Hampta Pass is a ~4,200m Himalayan
  crossing; late October there is pre-winter at altitude.
- A date implies a season. An Indian wedding in March is warm-weather; December
  in Delhi is cold.
- A duration implies quantity and laundry. A week-long trek needs odour-resistant
  layers, not seven cotton t-shirts.
- An occasion implies formality and budget. A 25th anniversary is a silver
  jubilee -- silver gifts carry specific meaning.

Put terrain and activity reasoning in `why_needed`. Do NOT state temperatures in
`climate_note` -- a later stage looks up real weather. For any named place
(mountain pass, trail, city), set `location`, `location_lat`, `location_lon`,
and `elevation_estimate_m` from your geographic knowledge. Leave ALL FOUR
null when the request names no place at all -- do not invent a placeholder
country or region ("India - unspecified city") to fill the field. A generic
"Indian shopping assistant" framing is not a location the user gave you.

## Buckets

Split the need into 2-5 buckets, each a distinct category of thing to buy
(e.g. "Layering & Insulation", "Footwear", "Navigation & Safety"). Search runs
once per bucket, so buckets are what stop a trek request returning ten jackets
and no headlamp.

`search_phrases` must be written in *catalogue* language -- how a product
listing would describe itself -- not in the user's language. The user says
"something for freezing nights"; the phrase should be "insulated down jacket
rated for sub-zero temperatures".

## Composing an outfit or kit

Plan the COMPLETE set the request implies, before considering what is in stock.
A request for "a suit for a wedding" needs a suit jacket, matching trousers, a
shirt and formal shoes -- name every one of those, even if the catalogue has
none of them. Mark each `role`:

  required     - the request is not satisfied without it (trousers for a suit)
  recommended  - expected, but the request stands without it (a tie)
  optional     - a nice addition (a watch)

## Mapping needs onto the catalogue

You are given the catalogue's real taxonomy with live product counts. For each
slot, set `catalogue_paths` to the paths that could genuinely fill it.

This is the most important judgement you make:

- Choose a path only when products of that TYPE would satisfy the need.
- If nothing in the taxonomy is the right type, leave `catalogue_paths` EMPTY.
  An empty list is a correct, useful answer -- it reports a genuine gap. Never
  substitute a different product type to avoid an empty list. Sports shoes do
  not fill a formal-shoes need; jeans do not fill a formal-trousers need; a
  utility jacket does not fill a suit-jacket need.
- Paths with a count of 0 must not be used. Treat them as absent.
- Where the catalogue offers a genuinely different but culturally appropriate
  option, add it as a SEPARATE optional slot with its own honest name -- for an
  Indian wedding with no suits in stock, an "Indian formal alternative" slot
  pointing at Ethnic Wear/Sherwanis is helpful. Do not disguise it as the suit.

## Who this is for

Set `recipient` ONLY when the shopper is buying an item to give to someone
else. A third party mentioned as part of an occasion is not automatically a
recipient:

- "traditional wear for my friend's wedding" -> the shopper is ATTENDING,
  buying for themselves. `recipient` stays null. The wedding is the occasion,
  not evidence of a gift.
- "a gift for my friend's wedding" / "my friend's wedding gift" -> explicit
  gifting language. `recipient` = "friend".
- "something for my mom's anniversary" -> ambiguous the same way as the
  wedding case; default to null unless the phrasing says "gift", "present",
  or names the recipient as who the item is *for* rather than whose event
  it is.

When in doubt, leave `recipient` null. A wrongly-assumed gift search asks
the wrong questions (whose taste, not the shopper's) and can misdirect the
whole plan.

## Asking questions

Set `needs_clarification` true ONLY when a missing detail would genuinely change
which products are right, and the request is too thin to assume your way out of.

- "Suggest me some t-shirts" -> ask. Occasion, budget and who it is for all
  change the answer completely.
- "Trekking Hampta Pass last week of October for a week" -> do NOT ask. The
  trip, timing and duration are all stated; asking would be obstructive.

When you ask, give 2-5 tappable options per question with machine values the
system can merge directly:
  price:   "price_max:500", "price_min:500,price_max:1500", "price_min:3000"
  gender:  "gender:men", "gender:women", "gender:unisex"
  other:   "occasion:office", "use_case:trekking", "category:Footwear"

BUDGET OPTIONS MUST BE SATISFIABLE. The taxonomy gives you each path's real
price range. Before offering a budget option, check it against the ranges of
the paths you actually put in `catalogue_paths` -- every option must contain
at least some real products, and the highest option must not start above the
most expensive product available. Offering a range the catalogue cannot fill
sends the shopper to an empty result they chose themselves, which is worse
than not asking about budget at all. If a category's whole range is narrow,
ask fewer, wider budget options, or skip the budget question entirely and
assume instead.

Never ask about something the user already told you. Never ask more than 4.
Even when asking, still fill in `buckets` for your best current guess -- the
user may skip the questions.

## Assumptions

Every gap you filled without asking goes in `assumptions`, in plain second
person: "Assumed you need a full kit rather than single items."

Set `is_shopping_request` false for anything that is not a shopping request.
Dates today or later; resolve all relative dates against the date given."""


def format_taxonomy(taxonomy: dict) -> str:
    """Render the live catalogue as a compact menu for the planner.

    Zero-count paths are shown rather than hidden: seeing that
    `Footwear/Formal Shoes` exists as a concept but holds 0 products is what
    lets the model report a gap instead of inventing a substitute.

    Each populated path also carries its real price range, because counts
    alone are not enough to generate a *satisfiable* budget question. A
    shopper asking for women's wedding wear was offered a "Rs 2,000-5,000"
    chip and got nothing back: the catalogue holds 24 women's ethnic items,
    but the most expensive is Rs 1,955, so the price filter excluded all of
    them. The model could not have known -- it was shown counts and no prices.
    """
    lines = []
    for category, subs in taxonomy.get("categories", {}).items():
        parts = []
        for sub, entry in subs.items():
            price_range = entry.get("price_range")
            if entry["count"] and price_range:
                low, high = price_range
                parts.append(f"{sub}({entry['count']}, Rs{low}-{high})")
            else:
                parts.append(f"{sub}({entry['count']})")
        lines.append(f"  {category}: {', '.join(parts)}")
    return "\n".join(lines)


def _price_bounds(paths: list[str], taxonomy: dict) -> tuple[int, int] | None:
    """Cheapest and dearest real product across `paths`, or None if unknown."""
    lows: list[int] = []
    highs: list[int] = []
    categories = taxonomy.get("categories", {})
    for path in paths:
        category, _, sub = path.partition("/")
        entry = categories.get(category, {}).get(sub)
        if not entry or not entry.get("count") or not entry.get("price_range"):
            continue
        low, high = entry["price_range"]
        lows.append(low)
        highs.append(high)
    return (min(lows), max(highs)) if lows else None


def _option_is_satisfiable(value: str, low: int, high: int) -> bool:
    """True when the option's price window overlaps real stock.

    Values look like "price_max:1500" or "price_min:1500,price_max:3000".
    Anything unparseable is treated as satisfiable -- refusing to show an
    option we merely failed to read would be worse than showing one.
    """
    want_min, want_max = 0, float("inf")
    for part in value.split(","):
        key, _, raw = part.partition(":")
        if not raw.isdigit():
            continue
        if key.strip() == "price_min":
            want_min = int(raw)
        elif key.strip() == "price_max":
            want_max = int(raw)
    return want_min <= high and want_max >= low


def drop_unsatisfiable_budget_options(
    questions: list[dict], paths: list[str], taxonomy: dict | None
) -> list[dict]:
    """Remove budget chips no product in `paths` can satisfy.

    A shopper tapped a "Rs 2,000-5,000" chip for women's wedding wear and got
    an empty result: the catalogue holds 24 women's ethnic items, but the
    dearest is Rs 1,955. Showing the planner real price ranges made its
    options much better, but cannot guarantee them -- it is asked several
    questions at once and cannot reason about the cross-product (women +
    Rs 2,000-5,000 is empty even when each half is fine), and its output is
    not deterministic. So the same rule is enforced here.

    A budget question whose every option is unsatisfiable is dropped whole:
    asking a question where each answer leads to nothing is worse than
    assuming. Non-budget questions are never touched, and with no paths or no
    taxonomy nothing is dropped -- there would be nothing to check against.
    """
    if not paths or not taxonomy:
        return questions

    bounds = _price_bounds(paths, taxonomy)
    if bounds is None:
        return questions
    low, high = bounds

    kept: list[dict] = []
    for question in questions:
        if question.get("slot") != "budget":
            kept.append(question)
            continue

        options = [
            option
            for option in question.get("options", [])
            if _option_is_satisfiable(option.get("value", ""), low, high)
        ]
        if not options:
            logger.info(
                "Dropped budget question entirely: no option fits the Rs%d-%d "
                "range actually available in %s",
                low,
                high,
                paths,
            )
            continue
        if len(options) != len(question.get("options", [])):
            logger.info(
                "Dropped %d unsatisfiable budget option(s) against Rs%d-%d",
                len(question.get("options", [])) - len(options),
                low,
                high,
            )
        kept.append({**question, "options": options})
    return kept


def build_user_prompt(
    query: str, today: date, answers: list[str], taxonomy: dict | None = None
) -> str:
    parts = [f"Today's date is {today.isoformat()}.", "", f"Request: {query}"]
    if taxonomy:
        parts += [
            "",
            "Catalogue taxonomy with live counts. Paths showing (0) hold no "
            "products -- treat them as absent:",
            format_taxonomy(taxonomy),
        ]
    if answers:
        parts += [
            "",
            "The shopper already answered these clarifying questions "
            "(machine values, apply them as constraints and do NOT ask again):",
            ", ".join(answers),
        ]
    return "\n".join(parts)


def interpret(
    provider: LLMProvider,
    model: str,
    query: str,
    today: date,
    answers: list[str] | None = None,
    timeout_s: float | None = None,
    effort: str | None = None,
    taxonomy: dict | None = None,
) -> StructuredQuery:
    """Run LLM call 1. Raises LLMUnavailable so callers can degrade."""
    payload = provider.structured(
        system=SYSTEM,
        user=build_user_prompt(query, today, answers or [], taxonomy),
        schema=QUERY_SCHEMA,
        model=model,
        max_tokens=3000,
        timeout_s=timeout_s,
        effort=effort,
    )
    structured = StructuredQuery.model_validate(payload)

    # The model can emit a natural-language gender word ("neutral") that
    # ProductAttributes.gender's canonical enum does not recognize -- fuzzing
    # against a local model turned up "neutral" directly, and merge_answers()
    # already normalizes the same class of value from chip answers. Apply the
    # same alias table here so the interpreter's own direct output gets the
    # same protection, not just tapped answers.
    if structured.filters.gender:
        structured.filters.gender = _GENDER_ALIASES.get(
            structured.filters.gender.lower(), structured.filters.gender
        )

    # Answers already supplied settle the matter; a model that asks again would
    # trap the user in a loop.
    if answers:
        structured.needs_clarification = False
        structured.questions = []
    return structured


def canonical_answer_value(value: str) -> str:
    """Relabel a chip value whose prefix promises more than it can deliver.

    `occasion` is a closed vocabulary (services/taxonomy.py::OCCASIONS) that
    retrieval matches against `product.attributes.occasion` to award
    BOOST_OCCASION. The model does not always stay inside it -- a Goa trip
    produced `occasion:beach_casual` and `occasion:dining`, neither of which
    any product can carry, so neither could ever boost anything.

    Such a value is not inert: merge_answers widens the bucket search phrases
    with it, which is genuinely useful. Relabelling it `use_case:` says so
    honestly instead of leaving an occasion chip that cannot act as an
    occasion. Underscores become spaces because the widened phrase is embedded
    as text, and "beach_casual" is not a phrase anyone writes.

    Anything already in the vocabulary, and every other key, is untouched.
    """
    key, sep, raw = value.partition(":")
    if not sep or key.strip() != "occasion":
        return value
    if raw.strip().lower() in OCCASIONS:
        return value
    return f"use_case:{raw.strip().replace('_', ' ').replace('-', ' ')}"


def merge_answers(structured: StructuredQuery, answers: list[str]) -> StructuredQuery:
    """Fold tapped chip values into the query.

    Deterministic on purpose -- this is what keeps the clarification round-trip
    free. Values look like "price_max:500" or "gender:men"; several may be
    comma-joined inside one chip.
    """
    for answer in answers:
        for pair in answer.split(","):
            # Relabel a chip whose prefix promises more than it can deliver,
            # before anything reads the key -- see canonical_answer_value().
            key, _, value = canonical_answer_value(pair).partition(":")
            key, value = key.strip(), value.strip()
            if not value:
                continue
            if key in ("price_min", "price_max"):
                if value.isdigit():
                    setattr(structured.filters, key, int(value))
            elif key == "gender":
                # LLM options may use natural labels such as ``male`` while
                # ProductAttributes uses the canonical enum ``men``. Always
                # normalize at the answer boundary before hard filtering.
                structured.filters.gender = _GENDER_ALIASES.get(value.lower(), value)
            elif key == "category":
                if value not in structured.filters.categories:
                    structured.filters.categories.append(value)
            elif key in ("occasion", "use_case"):
                # Surfaced to retrieval by widening the bucket phrasing, since
                # these are soft signals rather than hard filters.
                for bucket in structured.buckets:
                    bucket.search_phrases.append(value.replace("-", " "))
            elif key == "start_date":
                # From `context_slots._dates_question`: a resolvable date,
                # not a mood -- setting this (rather than just recording an
                # assumption, as the old vague timing buckets did) is what
                # lets Open-Meteo actually resolve weather from this answer
                # on the same turn.
                try:
                    structured.context.start_date = date.fromisoformat(value)
                except ValueError:
                    pass
            elif key == "duration_days":
                if value.isdigit():
                    days = int(value)
                    structured.context.duration_days = days
                    if structured.context.start_date is not None:
                        structured.context.end_date = (
                            structured.context.start_date + timedelta(days=days - 1)
                        )
            elif key == "timing":
                # Only reached for the "not sure yet" option, or a model-asked
                # timing question with no resolvable date -- a real date (see
                # "start_date" above) is preferred whenever one is offered.
                structured.assumptions.append(
                    f"Assumed trip timing: {value.replace('-', ' ')}."
                )
            elif key == "budget":
                # A model-generated budget question may use a semantic value
                # such as ``budget:flexible`` rather than a price bound.
                # Either way, the shopper has answered the budget slot.
                structured.assumptions.append(
                    f"Budget preference: {value.replace('-', ' ')}."
                )
    return structured


def offline_interpret(query: str, answers: list[str] | None = None) -> StructuredQuery:
    """Rule-based fallback used when no LLM is reachable.

    Intentionally shallow: it keyword-maps into categories and cannot infer that
    late October at 4,200m means sub-zero. Responses built on it are flagged
    `degraded_mode` so the difference is visible rather than passed off as
    reasoning.
    """
    from app.services.offline import build_offline_query

    return build_offline_query(query, answers or [])
