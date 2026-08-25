# Product Catalogue

The app uses a committed catalogue of 1,738 real, purchasable products:

- 156 products from Amazon.in
- 133 products from Myntra
- 1,449 products from a Kaggle Amazon-fashion archive (archival: US-priced at a
  fixed conversion rate, dated Feb 2024, never live-verified -- see
  `link_status` below and [scope.md](scope.md))
- 56 populated subcategories, across 68 taxonomy paths total (12 still empty)

The built catalogue lives in `backend/data/products.json`, with embeddings in
`backend/data/embeddings.npy` and taxonomy metadata in
`backend/data/taxonomy.json`.

## Why the Catalogue Is Committed

The assignment requires catalogue data from the public domain or generated data.
This project keeps the finished catalogue in the repository so the app can run
without an Apify token or enrichment credentials.

Raw source files are also kept under `backend/data/raw/` for provenance.

## Product Schema

Each product contains:

- stable `id`
- title and brand
- category and subcategory
- price and optional MRP
- description
- structured attributes
- rating and review count
- retailer
- product URL
- optional image URL
- stock flag
- link verification status

Important structured attributes include:

- `gender`
- `season`
- `use_case`
- `occasion`
- `material`
- `temp_rating_c`
- `is_giftable`
- `water_resistance`, `layer`, `formality`, `breathability` -- targets the
  1,250 apparel/footwear/outdoor-gear products where they are meaningful, via
  `scripts/enrich_suitability.py`; 678 enriched so far (partial run, see
  [scope.md](scope.md)). Feed the suitability layer
  (`services/constraints.py`, `services/suitability.py`); see
  [design-decisions.md](design-decisions.md). Unenriched products are safe --
  a `null` value is never treated as a conflict.

These attributes are used for filters, ranking boosts and explanation evidence.

## Offline Build Pipeline

```text
scripts/fetch_raw.py       Apify datasets -> data/raw/*.json
scripts/normalize.py       clean and dedupe -> validated Product records
scripts/enrich.py          LLMProvider (anthropic/openrouter/local) -> use_case,
                            occasion, season, temp_rating_c, material
scripts/reclassify.py      canonical two-axis taxonomy
scripts/verify_links.py    link_status: verified | blocked
scripts/build_catalogue.py data/products.json
scripts/build_index.py     data/embeddings.npy
scripts/build_taxonomy.py  data/taxonomy.json
scripts/enrich_suitability.py  patches water_resistance, layer, formality,
                            breathability onto the already-built products.json
                            (apparel/footwear/outdoor-gear only; sync API call,
                            not part of the batch pipeline above)
```

To rebuild everything:

```bash
cd backend
uv run python scripts/build_catalogue.py
uv run python scripts/build_index.py
uv run python scripts/build_taxonomy.py
```

Rebuilding raw data requires `APIFY_TOKEN`. Rebuilding enriched attributes
requires a configured `LLM_PROVIDER` and its matching key -- `ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, or a local Ollama server (no key).

## Enrichment Strategy

Retailers usually do not publish assignment-useful fields such as "good for
anniversary gifting" or "comfortable to -10C". Those are inferred once offline
and stored as structured product attributes.

The live app does not call a model to enrich products at request time. It only
uses the committed enriched catalogue.

## Link Provenance

Every product includes a retailer URL. Link verification records whether a live
product page was confirmed or whether bot protection blocked verification.

The UI still shows the retailer link in both cases, but the backend keeps
verification status available for auditing.
