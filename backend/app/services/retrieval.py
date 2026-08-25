"""Retrieval: turn a StructuredQuery into scored candidate products.

This layer runs entirely locally -- no LLM, no network, no per-query cost. It is
also the layer that decides *what the ranker even gets to see*, so its job is
recall and diversity rather than final ordering.

Two design choices carry most of the weight:

**Search runs per bucket, not once per query.** A single global search for a trek
request returns ten jackets, because jackets dominate the similarity space for
"warm clothing for a Himalayan trek". Searching once per bucket -- layering,
footwear, navigation -- guarantees the shortlist spans the actual need.

**Category gating is a hard filter, not a soft signal.** A slot declares which
canonical `Category/Subcategory` paths may fill it, and anything outside them is
not a candidate at any similarity score. This exists because cosine similarity
is *relative*: women's jeans scored 0.52 against "formal trousers" -- above every
threshold -- simply because jeans genuinely are trousers. No absolute floor can
separate "closest available" from "actually appropriate", so the type constraint
has to be categorical.

**Each search phrase is searched separately, then merged by best score.** A
bucket's phrases are distinct sub-needs -- "headlamp LED", "trekking pole",
"sunscreen high SPF" -- and averaging their vectors produces a centroid that
means "generic trekking" and matches none of them. In testing that centroid
returned down jackets for a navigation-and-safety bucket while real headlamps
sat unretrieved in the catalogue. Scoring each phrase and keeping each product's
best match preserves every sub-need.

**Attribute boosts are multiplicative, not additive.** This matters more than it
sounds. With flat additive bonuses, a product that is merely *tagged* right can
outrank one that genuinely matches: in testing, a jacket at cosine 0.42 collected
+0.64 in trekking/temperature bonuses and beat a trekking boot at cosine 0.63 in
a bucket explicitly asking for footwear. Scaling boosts by the semantic score
instead -- `score = semantic * (1 + boosts)` -- means attributes can reorder
products that are already relevant but can never rescue one that is not.

So: cosine decides what is in contention, attributes decide the order within it.
A jacket rated to -10C still beats one rated to 5C for a sub-zero trek, because
they sit next to each other semantically and `temp_rating_c` breaks the tie.
"""

import numpy as np

from app.schemas.product import Product
from app.schemas.query import Bucket, ContextConstraints, QueryFilters, ResolvedContext
from app.services.catalogue import Catalogue
from app.services import suitability
from app.services.taxonomy import ALL_PATHS, PRODUCT_TAXONOMY

# Boost weights, applied multiplicatively against the semantic score. Their sum
# is the most a perfectly-tagged product can gain (here ~1.6x), which keeps them
# influential without letting them override relevance.
BOOST_USE_CASE = 0.15
BOOST_OCCASION = 0.12
BOOST_SEASON = 0.08
BOOST_TEMP_FIT = 0.25
BOOST_POPULARITY = 0.05

# Semantic floors, relaxed deliberately once category gating landed.
#
# These were calibrated when a bucket searched the entire catalogue, where a
# floor was the only thing stopping a laptop backpack appearing under
# "Outerwear". Gating now guarantees every candidate is of the right product
# type, so the floor is filtering *within* an already-correct set -- and at the
# old 0.30/0.42 it discarded genuine trekking boots for scoring 0.35 against a
# footwear slot, emptying entire groups. Type-correctness is the relevance test
# now; similarity only orders within it.
MIN_SEMANTIC = 0.05
MIN_GROUP_BEST_SEMANTIC = 0.05

# Condition thresholds, in Celsius. COLD_THRESHOLD_C is where insulation rating
# starts being evidence rather than noise -- below it a warmer jacket is
# genuinely better, above it it is just heavier.
COLD_THRESHOLD_C = 5.0

# Season inference bounds. These previously sat at 5C and 30C with wetness as a
# 20mm window total, which left a dead zone: anywhere between 5C and 30C with
# ordinary rainfall produced *no season at all*, so nothing could be boosted for
# matching the conditions and -- worse -- nothing could be marked as conflicting
# with them. That covers most of India for most of the year, and it is why a
# winter snow jacket ranked first for Mumbai in monsoon.
WINTER_SEASON_MAX_C = 10.0
SUMMER_SEASON_MIN_C = 28.0
WET_MM_PER_DAY = 3.0

# An item declaring a lower comfort bound of R is usable up to about R + span.
# Past that it is not merely unnecessary, it is the wrong garment.
THERMAL_SPAN_C = 15.0
# How far above that ceiling counts as fully wrong rather than borderline.
SEVERE_EXCESS_C = 15.0

# Penalties are subtracted after scaling: unlike boosts, these are statements
# about the listing itself rather than about how well it fits the request.
PENALTY_UNVERIFIED_LINK = 0.02
PENALTY_OUT_OF_STOCK = 0.25
# Deliberately large. A thermal conflict is a statement that the product is
# wrong for the conditions, not that it is slightly less apt -- so it has to be
# able to sink a candidate that wins on raw text similarity, which is exactly
# how the snow jacket beat every rain shell.
PENALTY_THERMAL_MISMATCH = 0.5
# Same shape and rationale as PENALTY_THERMAL_MISMATCH, for the suitability
# axes in services/suitability.py (rain protection, formality) rather than
# temperature. Scaled by Verdict.strong_penalty, which is itself a fraction
# in the same 0..1 sense as temperature_fit's returned score.
PENALTY_SUITABILITY_STRONG = 0.5

# How informative each kind of evidence is, lowest first. This is a separate
# axis from the boost weights above: a temperature margin that only earns a
# small boost is still the most useful sentence we can offer, because it is
# specific, measured, and about this trip rather than about the product in
# general.
EVIDENCE_TEMPERATURE = 0
# Rain protection and formality are explicit, request-specific facts in the
# same way a measured temperature margin is -- ranked just below it and
# above season/use-case, which are broader tags rather than a direct check
# against a stated requirement.
EVIDENCE_SUITABILITY = 1
EVIDENCE_SEASON = 2
EVIDENCE_USE_CASE = 3
EVIDENCE_OCCASION = 4
EVIDENCE_MATERIAL = 5
EVIDENCE_POPULARITY = 6

# Below this a rating is too thinly sourced to quote at a shopper.
MIN_REVIEWS_TO_CITE = 50


class ScoredProduct:
    __slots__ = ("product", "score", "semantic", "reasons")

    def __init__(self, product: Product, score: float, semantic: float, reasons: list[str]):
        self.product = product
        self.score = score
        self.semantic = semantic
        # Machine-readable notes on *why* this scored well. The ranker gets
        # these as evidence so its explanations stay grounded in the data
        # rather than invented from the product title.
        self.reasons = reasons

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{self.product.id} score={self.score:.3f}>"


def _normalize_path(path: str) -> str:
    """Tolerate the formatting noise a model adds around an otherwise-correct
    path -- a leading/trailing slash was the concrete case (a local-model
    fuzzing run returned "/Men's Apparel/Jackets & Coats"), which fails an
    exact-match test just as completely as a wrong path does. Normalizing
    both sides the same way can only ever accept a path that was already
    right in substance; it cannot make a genuinely wrong path match.
    """
    return path.strip().strip("/")


def matches_paths(product: Product, paths: list[str]) -> bool:
    """Whether a product is of a type this slot accepts.

    An empty `paths` list means the slot declared no catalogue match -- a gap.
    Nothing fills it; the caller reports it instead.
    """
    if not paths:
        return False
    product_path = f"{product.category}/{product.subcategory}"
    return product_path in paths or any(_normalize_path(p) == product_path for p in paths)


def sanitize_categories(filters: QueryFilters) -> QueryFilters:
    """Drop any `filters.categories` entry that isn't a real category or a
    real Category/Subcategory path.

    Nothing in the interpreter's schema tells the model what this field
    should contain -- confirmed by fuzzing it against a local model, which
    filled it with plain topic words ("trekking", "outdoor") rather than
    anything from the taxonomy. Matched literally, that zeroes out every
    candidate in every bucket: the exact "nothing matched" failure this
    exists to prevent.

    An unrecognized value is far more likely to be model confusion than a
    deliberate "show nothing" instruction, so it is dropped rather than
    enforced. If every supplied value turns out unrecognized, this leaves no
    category constraint at all -- `bucket.catalogue_paths` already does the
    real type-correctness gating per bucket (and is far more reliably
    populated, since the prompt describes its exact format at length), so a
    fully-noisy `filters.categories` costs nothing to ignore.
    """
    if not filters.categories:
        return filters
    known = [c for c in filters.categories if c in PRODUCT_TAXONOMY or c in ALL_PATHS]
    if len(known) == len(filters.categories):
        return filters
    return filters.model_copy(update={"categories": known})


def _relevant_categories(filters: QueryFilters, bucket: Bucket) -> list[str]:
    """The category constraint that actually applies to this bucket.

    `filters.categories` is one request-global value, but a multi-bucket
    request legitimately spans several different top-level categories -- a
    trek needs Men's Apparel *and* Footwear *and* Bags & Luggage, each in its
    own bucket. Enforcing one global category list against every bucket
    equally can reject a bucket's own, more specific and far more reliably
    populated `catalogue_paths` outright: fuzzing surfaced exactly this,
    where the model set filters.categories=["Outdoor & Camping Gear"] for
    the whole request while separately (and correctly) planning a Footwear
    bucket, and the global filter silently zeroed it.

    If nothing in `filters.categories` overlaps the categories this bucket's
    own paths already require, `catalogue_paths` wins and the global
    constraint is dropped for this bucket only -- it was never describing
    this bucket in the first place.
    """
    if not filters.categories:
        return []
    bucket_categories = {path.split("/")[0] for path in bucket.catalogue_paths}
    relevant = [
        c for c in filters.categories
        if c in bucket_categories or c.split("/")[0] in bucket_categories
    ]
    return relevant


def passes_filters(product: Product, filters: QueryFilters) -> bool:
    """Hard constraints. A product failing any of these is never a candidate.

    `filters.categories` is meant to hold bare top-level category names, but
    nothing in the interpreter's prompt or schema tells the model that --
    `catalogue_paths` uses full "Category/Subcategory" strings everywhere in
    the same prompt, and a live capture showed the model reusing that form
    here too ("Men's Apparel/Casual Shirts"). Matched against a bare
    `product.category` ("Men's Apparel"), that silently failed every product
    in the category: zero candidates, no error. Accepting either form at this
    boundary is what keeps that model-phrasing variance from reaching the
    shopper as an empty screen.
    """
    if filters.price_min is not None and product.price_inr < filters.price_min:
        return False
    if filters.price_max is not None and product.price_inr > filters.price_max:
        return False
    # "unisex" means two different things depending which side of this check
    # it's on. As a product tag it means "fits any requested gender" (handled
    # by the `"unisex"` in the tuple below). As a *requested* filter it means
    # "no preference between men's and women's" -- the GENDER_QUESTION chip
    # is literally labelled "Anyone / unisex" -- not "show only the
    # literally-unisex-tagged 4% of the catalogue". Fuzzing against a local
    # model surfaced the gap concretely: a men's-dominant category (77
    # products, 3 tagged unisex) collapsed to those 3 the moment the
    # interpreter set gender="unisex", emptying a bucket whose
    # catalogue_paths were otherwise entirely correct.
    #
    # "kids" stays excluded even under a unisex request: it's a distinct,
    # deliberate category (age-appropriateness, not cut), and nobody asked
    # for it just by expressing no preference between men's and women's.
    if filters.gender == "unisex":
        if product.attributes.gender == "kids":
            return False
    elif filters.gender and product.attributes.gender not in (filters.gender, "unisex"):
        return False
    if filters.categories:
        product_path = f"{product.category}/{product.subcategory}"
        if product.category not in filters.categories and product_path not in filters.categories:
            return False
    return True


# Occasions specific enough that a mismatch is a real content problem, not
# a missed nice-to-have -- deliberately excludes broader, less exclusive
# contexts (office, party, travel, everyday) where requiring an exact tag
# match would reject perfectly good products that simply weren't tagged
# that precisely. A live capture is the reason this list exists at all: a
# "Genshin Merch Box" (occasion=[birthday, festive]) surfaced as a top pick
# for a parents' 25th anniversary hamper, because nothing outside the
# wedding-specific check below verified occasion compatibility for anything
# but weddings.
_GATED_OCCASIONS = {"wedding", "anniversary", "birthday", "festive", "interview"}


def passes_occasion_context(product: Product, bucket: Bucket) -> bool:
    """Reject explicit occasion conflicts for occasion-specific buckets.

    Product type gating alone is insufficient for a sparse archive catalogue:
    a Halloween/cosplay accessory can be a semantic match for wedding
    accessories, and a birthday-themed gift box can be a semantic match for
    an anniversary hamper. When enrichment explicitly records occasions,
    require a compatible one; products with no occasion metadata remain
    eligible either way, because missing evidence is not evidence of a
    conflict.
    """
    bucket_text = " ".join(
        [bucket.name, bucket.why_needed, *bucket.search_phrases]
    ).lower()
    product_occasions = {occasion.lower() for occasion in product.attributes.occasion}

    if "wedding" in bucket_text:
        product_path = f"{product.category}/{product.subcategory}"
        asks_for_accessories = "accessor" in bucket_text or "jewellery" in bucket_text
        if asks_for_accessories and product_path == "Gifting/Keepsakes":
            return False
        # "festive" is treated as wedding-compatible here specifically --
        # Indian wedding season overlaps heavily with festive-occasion
        # tagging -- which is why this stays its own branch rather than
        # folding into the general check below.
        return not product_occasions or bool(product_occasions & {"wedding", "festive"})

    named = {occasion for occasion in _GATED_OCCASIONS if occasion in bucket_text}
    if named and product_occasions and not (product_occasions & named):
        return False
    return True


def temperature_fit(product: Product, context: ResolvedContext) -> tuple[float, str | None]:
    """Reward insulation that actually covers the conditions.

    `temp_rating_c` is the lowest comfortable temperature, so a *lower* number
    is warmer: a jacket rated -10C covers a -5C night, one rated 5C does not.

    Ranks on measured numbers when we have them. The string-matching path below
    is kept only for the case where we do not -- an unverifiable location, or a
    request with no dates -- because scoring on a sentence is what produced the
    original defect: a fabricated "-5C to -10C" note ranked a -5C jacket top for
    nights that are actually -14.7C.

    The returned fit is signed, in [-1, 1]:

        > 0   insulation genuinely covers the cold in the request
        = 0   no opinion -- either uninsulated, or conditions say nothing
        < 0   insulation is excessive for the conditions

    The negative half is not symmetry for its own sake. Without it the rule only
    ever *rewarded* warmth and never objected to it, so a jacket rated to -5C
    drew no penalty at all for a 29C day in Mumbai and won its bucket on text
    similarity.
    """
    rating = product.attributes.temp_rating_c
    if rating is None:
        return 0.0, None

    climate = context.climate
    if climate is not None and climate.has_numbers:
        return _fit_from_numbers(rating, climate.temp_min_c, climate.temp_max_c)
    return _fit_from_note(rating, context.climate_note)


def _fit_from_numbers(
    rating: int, coldest_night_c: float | None, warmest_day_c: float | None
) -> tuple[float, str | None]:
    """Score insulation against the conditions, in both directions.

    The margin is what matters, not the raw rating. A -20C jacket is not
    "better" in the abstract -- it is correct for -17C nights and wrong for +29C
    days, and the reason string says which.

    The cold side is checked first and wins outright. A high desert can be +20C
    by day and -18C at night, and it is the night the kit has to survive -- so
    once the conditions genuinely call for insulation, a warm daytime high is
    not an objection to carrying it.
    """
    if coldest_night_c is not None and coldest_night_c <= COLD_THRESHOLD_C:
        return _covers_the_cold(rating, coldest_night_c)

    if warmest_day_c is not None:
        excess = warmest_day_c - (rating + THERMAL_SPAN_C)
        if excess > 0:
            severity = min(1.0, excess / SEVERE_EXCESS_C)
            return -severity, (
                f"rated for {rating}C and far too warm for the "
                f"{warmest_day_c:.0f}C days at your dates"
            )

    return 0.0, None


def _covers_the_cold(rating: int, coldest_night_c: float) -> tuple[float, str | None]:
    """How well insulation covers a genuinely cold night."""
    margin = rating - coldest_night_c
    if margin <= 0:
        return 1.0, f"rated to {rating}C, covering the {coldest_night_c:.0f}C nights at your dates"
    if margin <= 5:
        return 0.6, (
            f"rated to {rating}C, close to the {coldest_night_c:.0f}C nights but "
            "not fully covering them"
        )
    if margin <= 10:
        return 0.25, f"rated to {rating}C, short of the {coldest_night_c:.0f}C nights at your dates"
    return 0.0, None


def _fit_from_note(rating: int, note: str | None) -> tuple[float, str | None]:
    """Fallback for when no measured temperature could be obtained.

    Deliberately coarse. It cannot distinguish -5C from -15C, which is exactly
    why it is the fallback and not the primary path.
    """
    if not note:
        return 0.0, None
    lowered = note.lower()
    cold = any(
        token in lowered
        for token in ("sub-zero", "subzero", "below freezing", "snow", "-", "cold", "altitude")
    )
    if not cold:
        # Same asymmetry fixed as coarsely as this path allows: when the only
        # evidence is a sentence and that sentence describes heat, insulation is
        # still an objection rather than a neutral fact.
        hot = any(
            token in lowered
            for token in ("humid", "monsoon", "tropical", "hot", "summer", "muggy")
        )
        if hot and rating <= 10:
            return -1.0, f"rated for {rating}C, far too warm for the conditions described"
        return 0.0, None

    if rating <= -5:
        return 1.0, f"rated to {rating}C, covers sub-zero nights"
    if rating <= 0:
        return 0.7, f"rated to {rating}C, handles freezing conditions"
    if rating <= 5:
        return 0.35, f"rated to {rating}C, suits chilly but not freezing weather"
    return 0.0, None


def implied_seasons(context: ResolvedContext) -> set[str]:
    """Which product seasons the conditions actually call for.

    Previously this was a substring test against the model's climate sentence,
    so it fired whenever the model happened to type the word "winter". Derived
    from measured numbers it fires when the weather is actually wintry.
    """
    climate = context.climate
    if climate is None or not climate.has_numbers:
        return set()

    seasons: set[str] = set()
    if climate.temp_min_c is not None and climate.temp_min_c <= WINTER_SEASON_MAX_C:
        seasons.add("winter")
    if climate.temp_max_c is not None and climate.temp_max_c >= SUMMER_SEASON_MIN_C:
        seasons.add("summer")
    rainfall = climate.precipitation_mm_per_day
    if rainfall is not None and rainfall >= WET_MM_PER_DAY:
        seasons.add("monsoon")
    return seasons


def _overlap(left: list[str], right: set[str]) -> list[str]:
    return [item for item in left if item.lower() in right]


def score_product(
    product: Product,
    semantic: float,
    bucket: Bucket,
    context: ResolvedContext,
    constraints: ContextConstraints | None = None,
) -> ScoredProduct:
    """Fuse semantic similarity with attribute evidence.

    `score = semantic * (1 + boosts) - penalties`. See the module docstring for
    why the boosts multiply rather than add.

    Evidence is collected as `(rank, text)` and sorted before it reaches
    `ScoredProduct.reasons`, because the explanation layer shows only the first
    two. Insertion order would put whichever check happens to run first in
    front; `EVIDENCE_*` orders them by how much the clause actually tells a
    shopper. A measured temperature margin is the most informative thing we can
    say about a jacket and must never be crowded out by a generic tag match.
    """
    if constraints is None:
        constraints = ContextConstraints()

    boost = 0.0
    evidence: list[tuple[int, str]] = []

    phrase_tokens = {
        token.lower()
        for phrase in [*bucket.search_phrases, bucket.name, bucket.why_needed]
        for token in phrase.replace(",", " ").split()
    }

    matched_use = _overlap(product.attributes.use_case, phrase_tokens)
    if matched_use:
        boost += BOOST_USE_CASE
        evidence.append((EVIDENCE_USE_CASE, f"made for {', '.join(matched_use)}"))

    matched_occasion = _overlap(product.attributes.occasion, phrase_tokens)
    if matched_occasion:
        boost += BOOST_OCCASION
        evidence.append((EVIDENCE_OCCASION, f"suits {', '.join(matched_occasion)}"))

    seasons = implied_seasons(context)
    if not seasons and context.climate_note:
        # No measured numbers: fall back to reading the sentence, coarsely.
        lowered = context.climate_note.lower()
        seasons = {s for s in ("winter", "summer", "monsoon") if s in lowered}
    for season in product.attributes.season:
        if season != "all-season" and season in seasons:
            boost += BOOST_SEASON
            evidence.append((EVIDENCE_SEASON, f"suited to {season} conditions"))
            break

    thermal_penalty = 0.0
    temp_score, temp_reason = temperature_fit(product, context)
    if temp_score > 0:
        boost += BOOST_TEMP_FIT * temp_score
    elif temp_score < 0:
        # Subtracted after scaling rather than folded into `boost`, so it cannot
        # be diluted by a strong similarity score. A product that is wrong for
        # the conditions has to sink however well its text reads -- diluting the
        # objection is precisely how a snow jacket held first place at 29C.
        thermal_penalty = PENALTY_THERMAL_MISMATCH * -temp_score
    if temp_reason:
        # Recorded either way: an objection is at least as worth showing the
        # shopper as an endorsement.
        evidence.append((EVIDENCE_TEMPERATURE, temp_reason))

    suitability_penalty = 0.0
    verdict = suitability.evaluate(product, constraints)
    if verdict.soft_boost:
        boost += verdict.soft_boost
    if verdict.strong_penalty:
        # Same shape as the thermal penalty above: subtracted after scaling
        # so a strong mismatch sinks regardless of how well the text matched.
        suitability_penalty = PENALTY_SUITABILITY_STRONG * verdict.strong_penalty
    for reason in verdict.reasons:
        evidence.append((EVIDENCE_SUITABILITY, reason))

    # Explanation-only: material earns no boost because there is no evidence it
    # predicts fit, but "built with down" is a concrete thing to say about a
    # product whose tags happened to match nothing.
    if product.attributes.material:
        evidence.append(
            (EVIDENCE_MATERIAL, f"built with {product.attributes.material.lower()}")
        )

    # Social proof, compressed hard: this should break ties between comparable
    # products, never lift a poor match above a good one.
    if product.rating and product.review_count:
        popularity = (product.rating / 5.0) * min(1.0, np.log1p(product.review_count) / 10.0)
        boost += BOOST_POPULARITY * float(popularity)
        # Only cited once enough people have weighed in for the number to mean
        # something -- "4.5* from 3 reviews" is noise dressed as evidence.
        if product.review_count >= MIN_REVIEWS_TO_CITE:
            evidence.append(
                (
                    EVIDENCE_POPULARITY,
                    f"{product.rating:.1f}★ from "
                    f"{product.review_count:,} reviews",
                )
            )

    # Stable sort: equal ranks keep the order the checks ran in.
    reasons = [text for _, text in sorted(evidence, key=lambda item: item[0])]

    score = semantic * (1.0 + boost)

    if thermal_penalty:
        score -= thermal_penalty
    if suitability_penalty:
        score -= suitability_penalty
    if not product.in_stock:
        score -= PENALTY_OUT_OF_STOCK
    if product.link_status != "verified":
        # Mildly prefer products whose link we actually confirmed resolves.
        score -= PENALTY_UNVERIFIED_LINK

    return ScoredProduct(product, score, semantic, reasons)


def search_bucket(
    catalogue: Catalogue,
    query_vectors: np.ndarray,
    bucket: Bucket,
    filters: QueryFilters,
    context: ResolvedContext,
    limit: int,
    constraints: ContextConstraints | None = None,
) -> list[ScoredProduct]:
    """Return the best candidates for one bucket.

    `query_vectors` is (n_phrases, dim). Each product keeps its best score
    across the phrases rather than its score against their average -- see the
    module docstring.
    """
    if query_vectors.ndim == 1:
        query_vectors = query_vectors[None, :]

    if constraints is None:
        constraints = ContextConstraints()

    if not catalogue.has_vectors:
        semantic_scores = np.zeros(len(catalogue), dtype=np.float32)
    else:
        # Vectors are L2-normalised, so a dot product is cosine similarity and
        # every phrase scores against the whole catalogue in one matmul.
        semantic_scores = (catalogue.vectors @ query_vectors.T).max(axis=1)

    # An unmapped slot is a declared gap, not an invitation to search widely.
    if not bucket.catalogue_paths:
        return []

    bucket_filters = filters.model_copy(
        update={"categories": _relevant_categories(filters, bucket)}
    )

    scored: list[ScoredProduct] = []
    for index, product in enumerate(catalogue.products):
        if not matches_paths(product, bucket.catalogue_paths):
            continue
        if not passes_filters(product, bucket_filters):
            continue
        if not passes_occasion_context(product, bucket):
            continue
        if suitability.evaluate(product, constraints).hard_mismatch:
            continue
        scored.append(
            score_product(
                product, float(semantic_scores[index]), bucket, context, constraints
            )
        )

    scored = [s for s in scored if s.semantic >= MIN_SEMANTIC]
    scored.sort(key=lambda s: s.score, reverse=True)
    return scored[:limit]


def is_group_worth_showing(items: list[ScoredProduct]) -> bool:
    """Whether a bucket produced anything worth a heading.

    Post-gating this is close to "did anything survive": a candidate that
    reached here is already of a type the slot accepts, so suppressing it would
    hide a legitimate product rather than an irrelevant one.
    """
    return bool(items) and items[0].semantic >= MIN_GROUP_BEST_SEMANTIC


def dedupe_across_buckets(
    results: dict[str, list[ScoredProduct]],
) -> dict[str, list[ScoredProduct]]:
    """Keep each product in only its best-scoring bucket.

    Buckets overlap by nature -- a fleece is plausible under both "layering" and
    "mid layers" -- and the same product appearing twice in one shortlist reads
    as a bug to the user even when both placements are defensible.
    """
    best: dict[str, tuple[str, float]] = {}
    for bucket_name, items in results.items():
        for item in items:
            current = best.get(item.product.id)
            if current is None or item.score > current[1]:
                best[item.product.id] = (bucket_name, item.score)

    return {
        bucket_name: [
            item for item in items if best[item.product.id][0] == bucket_name
        ]
        for bucket_name, items in results.items()
    }
