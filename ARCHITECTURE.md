# Architecture

The Personal Shopping Assistant turns a natural-language shopping request into
grouped, explained product recommendations. This document covers how the runtime
works and why the major boundaries exist.

The system has two pipelines that meet at `backend/data/`:

- an **offline catalogue pipeline** that builds products, taxonomy and embeddings
- a **runtime recommendation pipeline** that interprets a request and retrieves
  against those artifacts

| Document | Covers |
|---|---|
| [README.md](README.md) | Project overview, setup, API summary, stack |
| **this file** | Runtime design, component boundaries, data flow, constraints |
| [api.md](api.md) | Request/response contract and examples |
| [ai-approach.md](ai-approach.md) | AI, retrieval and degraded-mode strategy |
| [catalogue.md](catalogue.md) | Catalogue source, schema, enrichment and indexing |
| [performance.md](performance.md) | Measured startup, per-request and per-model timings |
| [scope.md](scope.md) | Known limitations and future scope |

## A Request, End to End

Request:

```text
I am going for a trek to Hampta Pass in the last week of October for one week.
Please find me trekking essentials and clothing.
```

| Stage | What happens |
|---|---|
| **interpret** | The LLM returns a `StructuredQuery`: trek intent, date window, location hint, buckets such as insulation, footwear and essentials, allowed catalogue paths, assumptions and possible clarifying questions. |
| **conditions** | The backend resolves weather/elevation through Open-Meteo when possible. Measured climate is attached with provenance; invented weather is not used for ranking. |
| **context audit** | Known, needed and unobtainable variables are produced for the UI. If a missing value would materially change the answer, the response becomes `mode: "clarify"`. |
| **retrieve** | Each bucket gets its own embedding search. Category gates prevent a high-similarity but wrong-type product from filling a required need. |
| **dedupe** | Products are deduplicated across buckets so one jacket does not appear in every group. |
| **explain** | Reasons are composed deterministically from product attributes, match evidence and resolved context. |
| **response** | The API returns grouped recommendations, assumptions, context variables, metadata and any unfilled slots. |

## Runtime Pipeline

The main orchestration lives in
`backend/app/services/recommend.py::recommend_events()`. The blocking endpoint
drains this generator; the streaming endpoint forwards its stages to the UI.
One implementation supports two transports.

```text
query
  |
  +- cache lookup
  |
  +- 1. interpret     LLM   -> StructuredQuery
  |                         event: interpreting
  |
  +- 2. conditions    HTTP  -> ClimateContext, provenance
  |                         event: checking conditions
  |
  +- 3. context audit code  -> known / needed / unobtainable variables
  |
  +- 4. clarify?      code  -> mode="clarify" when needed
  |
  +- 5. retrieve      vec   -> candidates per bucket
  |                         event: searching
  |
  +- 6. dedupe        code  -> product-level dedupe across groups
  |
  +- 7. explain       code  -> grounded product reasons
  |
  `- response               -> mode="results" or mode="clarify"
```

The only live LLM call in the completed full-quality search path is
interpretation. Retrieval, dedupe, unfilled-slot handling and explanations are
deterministic.

## Stage 1 - Intent and Buckets

`backend/app/services/interpreter.py`

The interpreter receives:

- the user's request
- today's date
- any tapped clarification answers
- the live catalogue taxonomy with product counts

It returns a `StructuredQuery` with:

- `intent_summary`
- `buckets`
- `filters`
- `context`
- `assumptions`
- `questions`
- `is_shopping_request`

The core decision is decomposition. "Trekking essentials and clothing" becomes
multiple shopping buckets, and each bucket becomes a result group. The model
does not directly choose final products.

Zero-count taxonomy paths are still shown to the model. That lets it report a
real catalogue gap instead of pretending a substitute exists.

## Stage 2 - Conditions

`backend/app/services/context.py`

Trip context is resolved outside the model when possible:

- place
- coordinates
- elevation
- temperature range
- precipitation
- forecast or climatology source

Open-Meteo supplies measured data. If a place is obscure, the model may propose
coordinates, but measured elevation is used to corroborate the proposal before
weather is trusted.

If conditions cannot be established, the climate value becomes `unobtainable`
and the response records that temperature evidence was not used.

## Stage 3 - Clarification and Context Audit

`backend/app/services/context_slots.py`

The context audit turns request variables into UI-visible state:

```text
known        user or external source established it
needed       missing, and could change the recommendation
unobtainable the system tried and could not establish it
```

When the missing detail would change the answer, the API returns
`mode: "clarify"` with chip options. Answers come back as machine values such as
`gender:men` or `price_max:5000`, and are merged without another model call.

The user can also skip clarification. In that case, the backend proceeds with
visible assumptions.

## Stage 4 - Retrieval

`backend/app/services/retrieval.py`

Retrieval runs once per bucket:

```text
Bucket search phrases -> local embeddings -> category gate -> score -> candidates
```

Key rules:

- product type is a hard gate through `catalogue_paths`
- price and gender filters are applied before ranking
- semantic similarity is the base score
- use-case, occasion, season and climate fit can boost relevant products
- boosts multiply the semantic score rather than overriding it
- penalties reduce weak or mismatched candidates

Each phrase in a bucket is scored separately, then merged by best match. This
keeps a bucket with "headlamp", "trekking pole" and "sunscreen" from becoming one
blurry "generic trekking" vector.

### Suitability gates

`backend/app/services/constraints.py` and `backend/app/services/suitability.py`

Alongside temperature fit, two more axes are checked against explicit context:
rain protection and occasion formality. `derive_constraints()` reads measured
climate and the request's own wording once per turn into a `ContextConstraints`
object; `suitability.evaluate()` checks each candidate against it and returns
a verdict at one of three severities:

- **hard** -- excluded from the bucket before scoring. Reserved for one
  narrow case: a trip that genuinely needs waterproof gear, matched against a
  product explicitly marked as offering none.
- **strong** -- a heavy multiplicative penalty, same shape as the thermal
  mismatch penalty. Can still surface, but sinks below type-correct
  alternatives.
- **soft** -- ordering only, folded into the existing boost.

A product's suitability attributes default to `null` until enriched, and
`null` is never treated as a conflict -- only an explicit, opposing value is.

## Stage 5 - Dedupe and Unfilled Slots

`backend/app/services/retrieval.py` and `backend/app/services/recommend.py`

Products are deduplicated across buckets. If a planned bucket has no good
candidate, the API does not hide the need. It returns an `UnfilledSlot` with:

- slot name
- role: `required`, `recommended` or `optional`
- reason

Required unfilled slots are surfaced in the UI as incomplete coverage.

## Stage 6 - Explanation

`backend/app/services/explain.py`

Explanations are not generated by a second model call. They are composed from:

- product attributes
- retrieval evidence
- resolved climate and dates
- use case and occasion matches
- material, price, rating and review evidence

This keeps reasons grounded. A product can be described as suitable for a trek
because its attributes and retrieval evidence support that claim; the system
does not invent product specifications.

## Data Trust Boundaries

The system treats data differently depending on where it came from.

| Source | Examples | May filter | May rank | May be shown |
|---|---|---:|---:|---:|
| Retailer fields | price, URL, image, rating, title | yes | yes | yes |
| Offline enrichment | use case, occasion, season, giftable, temperature rating | limited | yes | yes, as catalogue metadata |
| External lookup | weather, elevation, climatology | yes | yes | yes, with provenance |
| Model inference | assumptions, planned buckets, proposed obscure coordinates | no by itself | yes after checks | yes, labelled as inferred |
| Missing/unavailable | unresolved climate, absent category | no | no | yes, as a gap |

The design principle is simple: a weak inference may nudge ranking, but it should
not silently hide products or fabricate unavailable facts.

## Providers and Fallbacks

### LLM Provider

The LLM is behind the `LLMProvider` protocol
(`backend/app/adapters/llm/base.py`), so the interpreter and orchestrator do
not know or care which implementation is live. `LLM_PROVIDER` selects it:
`anthropic` (default, `AnthropicProvider`), `openrouter` (`OpenRouterProvider`,
an OpenAI-compatible chat-completions client), or `local` (`LocalProvider`,
talking to an Ollama-compatible endpoint at `LOCAL_LLM_BASE_URL`, no API key).
All three speak the same `structured(system, user, schema, model, ...)`
operation and turn a transport failure into `LLMUnavailable`. `INTERPRET_MODEL`
must be a model id valid for whichever provider is selected -- it does not
default sensibly across providers (the class default,
`claude-haiku-4-5`, is Anthropic-only).

An optional second provider (`LLM_FALLBACK_PROVIDER` +
`FALLBACK_INTERPRET_MODEL`) can be configured alongside the primary.
`app/adapters/llm/fallback_provider.py::FallbackProvider` wraps both behind
the same protocol -- the primary is tried first, and only on `LLMUnavailable`
(missing key, rate limit, transport error) does it try the fallback, so
recommend.py's interpret step needs no changes to benefit from it.

The provider is optional either way. If its key is absent or it fails, the
system uses `offline_interpret()` and marks the response as
`degraded_mode: true`.

Degraded responses are deliberately not cached, because they may be the result of
a temporary provider outage.

### Embeddings

The preferred embedding provider is local
`sentence-transformers/all-MiniLM-L6-v2`. The model downloads on first startup
and is cached locally.

If the semantic model is unavailable, the app falls back to a hashing embedder
and records that semantic matching is degraded.

## Storage

There is no runtime database. The backend loads immutable catalogue artifacts at
startup:

| File | Purpose |
|---|---|
| `backend/data/products.json` | product records |
| `backend/data/embeddings.npy` | embedding matrix aligned to products |
| `backend/data/embedding_ids.json` | product IDs aligned to embedding rows |
| `backend/data/taxonomy.json` | category counts, coverage and price ranges |

At the current size, this is simpler and faster than adding a database:

- 1,738 products fit comfortably in process memory.
- Vector scoring is a local NumPy operation.
- The catalogue is read-only at runtime.
- There is no second store to seed or keep in sync.

The intended scale boundary is `Catalogue.load()`. At larger catalogue sizes,
this can move to pgvector or another vector database without changing the API
contract.

## Backend Layout

```text
backend/app/main.py                 FastAPI app, CORS, lifespan
backend/app/api/v1/routes/          health and recommendation routes
backend/app/core/                   config, dependency loading, cache, errors
backend/app/schemas/                Pydantic request/response/product models
backend/app/adapters/llm/           Anthropic and OpenRouter providers, shared protocol
backend/app/adapters/embeddings/    local and hashing embedders
backend/app/adapters/weather/       Open-Meteo client
backend/app/services/interpreter.py natural language -> StructuredQuery
backend/app/services/recommend.py   orchestration
backend/app/services/retrieval.py   filtering, scoring and bucket search
backend/app/services/constraints.py context -> explicit requirement set
backend/app/services/suitability.py requirement set -> per-product verdict
backend/app/services/context.py     climate and location resolution
backend/app/services/explain.py     deterministic recommendation reasons
```

Dependencies run inward: routes call services, services call adapters, and
adapters wrap external I/O.

## Frontend Layout

```text
frontend/app/page.tsx                 single-page assistant and thread state
frontend/components/QueryBar.tsx      natural-language input
frontend/components/ThinkingState.tsx streamed stage display
frontend/components/ClarifyPanel.tsx  clarification chips
frontend/components/ResultsView.tsx   grouped results, assumptions, gaps
frontend/components/ProductCard.tsx   product image, price, reason, link
frontend/lib/api.ts                   fetch client and SSE parser
frontend/lib/thread.ts                follow-up composition
frontend/lib/types.ts                 TypeScript mirror of backend schemas
```

The frontend owns conversation state. The backend stays stateless. Follow-ups
are folded into self-contained requests before being sent.

## Errors and Metadata

Every recommendation response includes operational metadata:

```json
{
  "latency_ms": 1234,
  "llm_calls": 1,
  "cached": false,
  "degraded_mode": false,
  "catalogue_size": 1738,
  "notes": []
}
```

This is part of the product experience. A weaker answer should look weaker to
the user and to the evaluator.

## How This Is Tested

The backend tests exercise the design claims directly:

- category gates beat high but wrong similarity
- boosts cannot rescue irrelevant products
- required unmapped slots are reported
- LLM outage falls back to degraded mode
- climate proposal corroboration rejects bad coordinates
- API contract supports clarify, results, answers and skip

See [testing.md](testing.md) for the file-by-file test map.

## Known Constraints

The important constraints are tracked in [scope.md](scope.md):

- catalogue coverage is thin in some formalwear and outerwear categories
- overlapping bucket paths can be thinned too much by dedupe
- clarification budget options need taxonomy price-range awareness
- latency is dominated by interpretation
- both services are deployed (Vercel + Railway), but there is no CI pipeline yet

Those are product and scale constraints, not hidden runtime dependencies.
