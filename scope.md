# Scope

This file captures known limitations and future scope for the project.

## Known Limitations

### Catalogue Coverage

Catalogue coverage is a smaller constraint than it used to be: the Kaggle
ingestion (below) took the catalogue from 289 to 1,738 products and filled
formal shoes, suits and dresses. 12 taxonomy paths are still empty, mainly
women's outerwear (Blazers, Jackets & Coats), Footwear/Heels, and most of
Electronics & Accessories, Home & Kitchen and Beauty & Personal Care. Requests
landing there get an honest `unfilled_slots` report rather than a weak
substitute.

### Overlapping Bucket Paths

When two buckets share the same catalogue paths, cross-bucket dedupe can thin
one group too aggressively. A bucket should become an unfilled slot when dedupe
empties it, rather than showing a leftover weak match.

### Clarification Budgets (done)

A shopper asked for women's wedding wear, was offered a "Rs 2,000-5,000" chip,
tapped it, and got nothing: the catalogue holds 24 women's ethnic items but the
dearest is Rs 1,955. Retrieval was correct and the empty result honest -- the
defect was offering a choice that could only end that way.

Fixed on both sides. `format_taxonomy()` now shows each populated path's real
price range next to its count ("Lehengas(8, Rs1245-1955)"), so the planner is
no longer inventing ranges blind, and the prompt requires options to be checked
against the paths actually planned. Because a prompt cannot *guarantee* this --
the model is asked several questions at once and cannot reason about the
cross-product, and its output is not deterministic --
`drop_unsatisfiable_budget_options()` enforces the same rule in code. It runs
in `recommend.py` after `apply_context_audit`, where model-generated questions
and the deterministic ones `context_slots.py` appends both converge; the
reported chip came from the latter, so a guard on model output alone would have
missed it. Satisfiability is measured against the *required* buckets only:
checking all of them let "Rs 3,000+" pass on the strength of a Rs 9,684 potli
clutch in an accessories bucket while the outfit itself topped out at Rs 1,955.

### Non-Deterministic Ask/Do-Not-Ask

The ask-vs-assume decision comes from the interpreter and can vary. More
deterministic post-processing should decide when missing context truly requires
clarification.

### Catalogue Misclassification (done)

A systematic sanity pass found 121 sock products (7% of the catalogue)
scattered across 11 wrong buckets -- not just the 34 under
`Thermals & Base Layers` originally noticed, but as far afield as
`Footwear/Formal Shoes` (9), `Footwear/Casual Sneakers` (8),
`Women's Apparel/Trousers` (6), and one under `Suits & Blazers`. Root cause:
no `Socks` subcategory existed anywhere in `PRODUCT_TAXONOMY`, so each
Kaggle-sourced sock landed wherever its original enrichment pass guessed
closest -- these are `kag-` records added after `reclassify.py`'s one-time
type/occasion-axis correction already ran, so they never got that fix.

Fixed by adding `Socks & Hosiery` under both `Men's Apparel` and
`Women's Apparel` in `PRODUCT_TAXONOMY` and re-filing all 121 by
`attributes.gender` (unisex defaults to Men's Apparel, matching the existing
convention for the 30 other unisex-tagged apparel items already in the
catalogue). Two smaller single-item mis-shelvings were caught and fixed in the
same pass: a pair of hiking shoes filed as `Outdoor & Camping Gear/Trekking
Equipment`, and an Oxford dress shirt filed under `Sports & Fitness/Activewear`.
A related, narrower bug was also found and fixed: an "Emideary 1 Year Sobriety
Engraved Wallet Card" carried `occasion: ["anniversary", "birthday"]` --
plausible-sounding but wrong for a recovery-milestone keepsake. Cleared to `[]`
rather than force-fit into the closed `OCCASIONS` vocabulary, since no value in
it actually describes what the item is for, and an empty tag is never treated
as a conflict. A full sweep for the same pattern (narrow personal-milestone
themes force-tagged with a generic occasion) across all 482 giftable/Gifting
products, and a commemorative-keyword sweep across the full catalogue, found
no other instances.

One side effect worth noting: removing the mis-shelved sock revealed
`Sports & Fitness/Cycling` had no real product in it at all -- it is now
honestly empty rather than falsely showing 1 item, bringing the empty-path
count from 11 to 12 (see Catalogue Coverage above).

### Latency

Almost all of it is the single interpretation call, and the dominant factor is
*which model serves it*, not the app's own work. Full measurements — startup,
per-stage and per-model — are in [performance.md](performance.md); the summary:

| Path | Time |
|---|---|
| Cache hit | ~0.4s |
| Full search, warm, fast hosted model (`claude-haiku-4-5`) | ~7.5s end-to-end |
| Full search, free-tier model (`nvidia/nemotron-3-super-120b-a12b:free`) | 37-67s interpretation, and it returned unparseable JSON on roughly half of attempts |

Retrieval, dedupe and explanation are deterministic local work and are not a
meaningful share of the total.

Two consequences the deployment has to live with:

- **Free-tier models are a correctness problem, not just a speed one.** Their
  structured-output failures fall through to `offline_interpret()` and the
  response is honestly marked `degraded_mode: true` -- but the user waited a
  long time for a weaker answer.
- **A slow primary used to burn the whole budget before the fallback ran.**
  `FallbackProvider` falls through on *failure* only -- a rejected key,
  exhausted credit, a rate limit, a transport error, or the provider's own
  timeout expiring. It deliberately does not abandon a hop for being slow:
  an earlier version did, with an 8s deadline, and aborted a healthy
  provider mid-call because a real interpretation takes ~11s.

For a responsive deployment, put a fast reliable model first and keep the free
model as the fallback, not the other way round.

Server-sent events make the wait legible but do not reduce it.

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

### Personalisation

Add optional user preferences such as size, fit, brand affinity, price comfort,
previous rejections and gifting style. The existing `ContextVariable` model is a
good place to carry these signals.

### Server-Side Conversation State

Introduce conversation IDs and server-held state if deeper multi-turn
refinement becomes important. Today, the browser folds follow-ups into a
self-contained query.

### CI

Deployment itself is done -- frontend on Vercel, backend on Railway, both
auto-deploying from `main` (see [DEPLOYMENT.md](DEPLOYMENT.md) and the URLs in
[README.md](README.md)). What is still missing is a CI pipeline: `pytest`,
`npm run build` and `npm run lint` should run on every push, so a broken
commit is caught before it auto-deploys rather than after. Containerising both
services would also make the deploy reproducible rather than
platform-configured.

### Evaluation Harness

Create a fixed benchmark of shopping requests with expected buckets, product
types and unacceptable substitutions. This would make prompt, ranking and
catalogue changes measurable.

### Frontend Test Coverage (unit/component done; E2E still open)

Vitest + React Testing Library now cover degraded-mode banner, unfilled-slot
display, product links, and follow-up composition (44 tests, see
[testing.md](testing.md)) -- this caught a real bug in `composeFollowUp`
along the way: a near-cap-length follow-up could exceed the schema's
`max_length` because the fixed label overhead wasn't accounted for in the
final truncation fallback.

Still open: browser-level (Playwright) tests for the full clarification
round-trip against a live backend -- the component tests exercise
`ClarifyPanel`/`ResultsView` with fixture data, not the actual multi-turn
fetch/answer/re-render cycle.

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
  `scripts/enrich_suitability.py` targets the 1,250 apparel/footwear/
  outdoor-gear products where these axes are meaningful (Electronics, Home &
  Kitchen, Beauty, Gifting, Bags & Luggage and Watches & Jewellery are left
  alone -- a water-resistance judgement on a Bluetooth speaker is noise).
  **Coverage is partial: 678/1,250 enriched**, stopped short of the full run
  by OpenRouter's free-tier daily request cap (50/day on an unfunded account).
  The mechanism is fully correct either way -- an
  unenriched product's suitability attributes are `null`, and `null` is
  never treated as a conflict by `suitability.evaluate()` -- so the app
  behaves safely with partial coverage, it just has suitability signal for
  fewer products than it eventually could.

  The script is provider-agnostic (`--provider anthropic|openrouter|local`,
  same `LLMProvider` protocol the runtime uses), specifically so finishing
  this doesn't have to mean spending API credits: `--list-free-models`
  queries OpenRouter's current $0, structured-output-capable models live
  (`z-ai/glm-5.2:free` and `nvidia/nemotron-3-super-120b-a12b:free` were
  the strongest of five available at last check), and `--provider local`
  runs against Ollama with no external account at all. Re-run the script
  (it skips already-enriched products automatically) to extend coverage --
  spot-check a `--limit` batch first, since judgement quality varies with
  model size and a weak free/local model can mislabel more than it helps.

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

### Upfront Date/Time Question (done)

`_dates_question()` now carries concrete `start_date`/`duration_days` pairs
anchored to today, so answering it resolves climate on the same turn rather
than recording a prose assumption.

Two further defects were found on top of that. The slot was only marked
"needed" when the text matched trek vocabulary (*trek, hike, mountain, pass,
altitude*), so "suggest dress for my trip to goa" never had its missing date
recognised -- Hampta Pass only worked because "Pass" is in that list. And even
when the question existed, `apply_context_audit` merged model questions ahead
of deterministic ones before truncating to four, so a chatty model could push
the date question out entirely. The gate now covers any trip whose conditions
drive the packing list, and `_BLOCKING_SLOTS` is asked first: a date is a
dependency, while gender and budget only narrow an answer that already
exists.

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
