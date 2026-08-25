# Scope

This file captures known limitations and future scope for the project.

## Known Limitations

### Catalogue Coverage

Catalogue coverage is a smaller constraint than it used to be: the Kaggle
ingestion (below) took the catalogue from 289 to 1,738 products and filled
formal shoes, suits and dresses. 11 taxonomy paths are still empty, mainly
women's outerwear (Blazers, Jackets & Coats), Footwear/Heels, and most of
Electronics & Accessories, Home & Kitchen and Beauty & Personal Care. Requests
landing there get an honest `unfilled_slots` report rather than a weak
substitute.

### Overlapping Bucket Paths

When two buckets share the same catalogue paths, cross-bucket dedupe can thin
one group too aggressively. A bucket should become an unfilled slot when dedupe
empties it, rather than showing a leftover weak match.

### Clarification Budgets

The planner is not yet shown per-subcategory price ranges, so it can offer
budget chips that are technically impossible for a thin category. The taxonomy
already stores price ranges; the prompt should include them.

### Non-Deterministic Ask/Do-Not-Ask

The ask-vs-assume decision comes from the interpreter and can vary. More
deterministic post-processing should decide when missing context truly requires
clarification.

### Catalogue Misclassification

Some sock products are currently classified under thermals/base layers. These
should be re-shelved and used as regression examples for catalogue validation.

### Latency

A completed full-quality search can take 20-25 seconds, mostly inside the one
interpretation call. Server-sent events make the wait legible but do not reduce
the underlying latency.

### Deployment

The project is local only. There is no container, CI pipeline or live deployment
URL yet.

## Future Scope

### Scale Retrieval

Move from in-process vectors to pgvector or another vector database when the
catalogue grows beyond assignment scale. The current `Catalogue` interface is
the intended boundary for that change.

### Reduce Latency

Potential improvements:

- stream interpreter output as buckets are planned
- resolve climate concurrently with early retrieval work
- cache common interpretations more aggressively
- move the response cache to Redis for shared cache hits across replicas

### Improve Catalogue Coverage

Add missing formalwear, footwear, women's outerwear, gifting and accessory
categories. Improve validation so mis-shelved records are caught before build
output is committed.

### Use Price Ranges in Planning

Pass taxonomy price ranges into the interpreter so generated clarification
options are always satisfiable by the current catalogue.

### Personalisation

Add optional user preferences such as size, fit, brand affinity, price comfort,
previous rejections and gifting style. The existing `ContextVariable` model is a
good place to carry these signals.

### Server-Side Conversation State

Introduce conversation IDs and server-held state if deeper multi-turn
refinement becomes important. Today, the browser folds follow-ups into a
self-contained query.

### Deployment and CI

Containerise both services, run `pytest`, `npm run build` and `npm run lint` in
CI, deploy the API behind a managed host, and serve the frontend through a CDN.

### Evaluation Harness

Create a fixed benchmark of shopping requests with expected buckets, product
types and unacceptable substitutions. This would make prompt, ranking and
catalogue changes measurable.

### Frontend Test Coverage

Add browser-level tests for:

- clarification flow
- degraded-mode banner
- unfilled-slot display
- product links
- follow-up composition

## Remaining Planned Work (as of 2026-08-25)

Discovered and scoped during a recommendation-quality review, not yet
implemented. Listed in the order they were prioritised.

### Contextual Suitability Layer (done)

A generalized suitability mechanism was scoped in response to a real defect: a
monsoon request in Mumbai (26-29C) surfaced a -5C winter jacket as the top
pick, because retrieval had a type filter and a similarity score but nothing
modelling *fit for conditions*.

The threshold fix landed first: `temperature_fit()` in `services/retrieval.py`
is signed and symmetric (previously it only ever rewarded warmth, never
objected to it), `implied_seasons()` no longer has a 5C-30C dead zone, and
precipitation is read as a rate rather than a window total. This closed the
reported case, and is deliberately left as-is -- it is well-tested and
correct, and the generalized layer below did not need to absorb it.

The generalized mechanism landed on top, for the axes that had no mechanism
at all:

- `services/constraints.py::derive_constraints(structured) -> ContextConstraints`
  turns resolved climate and occasion wording into an explicit requirement
  set -- conservatively: a constraint is only set when there is a real
  signal, never a default.
- `services/suitability.py::evaluate(product, constraints) -> Verdict`
  classifies a mismatch as hard (veto before ranking -- reserved for
  water_resistance="none" against a genuinely waterproof-tier requirement),
  strong (heavy multiplicative penalty -- the rest of a water-resistance or
  formality conflict), or soft (ordering only). Missing product evidence is
  never treated as a conflict, matching `temperature_fit()`'s own rule.
- `ProductAttributes` gained `water_resistance` (none/repellent/waterproof),
  `layer` (base/mid/outer/standalone), `formality`
  (casual/smart_casual/formal), and `breathability` (low/medium/high).
  `scripts/enrich_suitability.py` targets the 1,201 apparel/footwear/
  outdoor-gear products where these axes are meaningful (Electronics, Home &
  Kitchen, Beauty, Gifting, Bags & Luggage and Watches & Jewellery are left
  alone -- a water-resistance judgement on a Bluetooth speaker is noise).
  **Coverage is partial: 107/1,201 enriched, stopped short of the full run
  to bound API spend.** The mechanism is fully correct either way -- an
  unenriched product's suitability attributes are `null`, and `null` is
  never treated as a conflict by `suitability.evaluate()` -- so the app
  behaves safely with partial coverage, it just has suitability signal for
  fewer products than it eventually could. Re-run the script (it skips
  already-enriched products automatically) to extend coverage.

Thermal mismatch was deliberately *not* moved into `suitability.py` --
scope.md's original note framed that as the ideal shape, but it would have
meant touching well-tested, working code for a cosmetic reorganisation with
no behavioural upside. The two mechanisms coexist: `temperature_fit()` for
temperature, `suitability.evaluate()` for rain protection and formality.

### Non-Blocking Initial Recommendation

Today a request that needs clarification returns *only* questions -- no
products -- on the first turn. The intended model is: interpret, retrieve and
show the best available recommendation immediately using whatever is known,
then ask targeted follow-up questions alongside it, and re-rank on the full
accumulated conversation once answered. This is a change to
`services/recommend.py::recommend_events()` (the `mode="clarify"` branch
currently early-returns with empty `groups`) and to `context_slots.py` (a
question should only be asked when answering it would actually change the
derived constraint set, not from a fixed list).

### Upfront Date/Time Question

`_DATES_QUESTION` in `context_slots.py` currently offers vague timing buckets
("within 2 weeks", "next 1-3 months") that cannot be turned into a date range
for the Open-Meteo lookup. Asking for an actual date (or date range) up front
would let climate resolve correctly on the first pass, removing a follow-up
round-trip for any request that implies weather-dependent recommendations.

### Kaggle Fashion Ingestion -- Completion

289 -> 1,738 products via a bulk Kaggle Amazon-fashion archive, filling 18 of
34 empty taxonomy paths (apparel, footwear, bags, jewellery). Two items
remain:

- **151 of the original 1,600-item sample** were never successfully classified
  -- a small residual of batches still truncate under `max_tokens=16000` even
  after the fix that resolved the bulk of the original 34% loss. Queued at
  `backend/data/raw/kaggle_fashion_remaining.json`, resumable at near-zero
  marginal cost via `enrich_kaggle.py` pointed at that file.
- **Electronics & Accessories, Home & Kitchen, and Beauty & Personal Care**
  (11 of the 34 originally-empty paths) are outside this dataset's coverage --
  it is a fashion archive -- and need their own sourcing, likely via
  additional Apify scrape queries following the existing `sources.py` pattern.

### Archival Provenance in the UI

`link_status` now has a third value, `"archival"` (`schemas/product.py`),
for the ~1,449 Kaggle-sourced records: US-priced (converted at a fixed,
documented rate, not a live FX lookup), dated Feb 2024, and never live-checked
by construction. The backend correctly never claims these are verified, but
`ProductCard.tsx` does not yet render `link_status` at all -- it should
visually distinguish archival listings from the Amazon.in/Myntra-verified
ones, so a shopper is never left to infer provenance from the URL domain.
