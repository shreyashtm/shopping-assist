# Personal Shopping Assistant

A natural-language shopping assistant for the Confluxe assignment. Describe a
trip, occasion or recipient in plain English, and it returns product
recommendations grouped by need, each with a short reason and a retailer link.

Example:

```text
I am going for a trek to Hampta Pass in the last week of October for one week.
Please find me trekking essentials and clothing.
```

The interesting part is that the sentence is a situation, not a product search.
The app turns that situation into trip context, shopping buckets, catalogue
constraints and recommendations.

## What Makes It Different From a Search Box

**It decomposes the request into needs.** A trek query is not one search for
"trekking essentials"; it becomes separate buckets such as insulation, footwear
and safety accessories. Each bucket gets its own retrieval pass and result group.

**It separates judgement from arithmetic.** The LLM interprets the request.
Filtering, category gates, vector retrieval, dedupe and explanations are
deterministic Python so prices, product types and catalogue gaps remain
auditable.

**It shows uncertainty instead of hiding it.** Missing details either become
clarifying chips or visible assumptions. Required needs the catalogue cannot
cover are returned as `unfilled_slots`, not filled with weak nearest-neighbour
matches.

**It uses measured context when context matters.** For trip requests, weather and
elevation are resolved through Open-Meteo where possible. A model may propose a
place, but measured numbers drive temperature-sensitive ranking.

## Setup Instructions

Run the backend and frontend in separate terminals.

Backend:

```bash
cd backend
cp .env.example .env
uv sync --extra dev
uv run uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

Open <http://localhost:3000>. API docs are at <http://localhost:8000/docs>.

The app works without any LLM key configured; it falls back to keyword
interpretation and marks responses as `degraded_mode: true`. Set
`LLM_PROVIDER` plus the matching key (`ANTHROPIC_API_KEY`,
`OPENROUTER_API_KEY`, or a local Ollama server) in `backend/.env` for full AI
interpretation.

More setup detail: [setup.md](setup.md).

## Architecture Overview

```text
query
  -> interpret     LLM   -> StructuredQuery: context, buckets, filters, questions
  -> conditions    HTTP  -> Open-Meteo weather/elevation when useful
  -> retrieve      vec   -> local embedding search once per bucket
  -> dedupe        code  -> avoid repeating products across groups
  -> explain       code  -> grounded reasons from product/context evidence
  -> respond             -> results or clarification questions
```

The backend is stateless and layered as:

```text
routes -> services -> adapters
```

The frontend is a single Next.js page that owns the conversation thread and
streams backend stages through SSE-over-POST.

Full architecture: [ARCHITECTURE.md](ARCHITECTURE.md).

## API

| Method | Path | Behaviour |
|---|---|---|
| `GET` | `/api/v1/health` | Runtime readiness and capability check |
| `POST` | `/api/v1/recommend` | Blocking recommendation response |
| `POST` | `/api/v1/recommend/stream` | Same pipeline streamed as server-sent events |

Request body:

```json
{
  "query": "Find me good traditional wear for my friend's wedding in March next year.",
  "answers": [],
  "skip_clarification": false
}
```

The response is either `mode: "clarify"` with questions, or `mode: "results"`
with grouped recommendations. See [api.md](api.md) for examples.

## Stack

- Backend: Python 3.12, FastAPI, Pydantic v2, NumPy, sentence-transformers.
- Frontend: Next.js 16, React 19, TypeScript, Tailwind v4.
- AI: Anthropic, OpenRouter, or a local Ollama model (configurable) for
  structured intent interpretation; deterministic explanation composition.
- Retrieval: local `all-MiniLM-L6-v2` embeddings over the committed catalogue.
- Context: Open-Meteo for weather, elevation and climatology.

## Design Decisions

- Search runs once per bucket, not once per whole query.
- Category gates are hard filters, not similarity thresholds.
- Attribute boosts multiply the semantic score instead of rescuing weak matches.
- Empty required slots are reported rather than substituted.
- The backend stores no conversation state; follow-ups are composed client-side.
- Suitability mismatches have three severities (veto / heavy penalty / ordering only), not one.

See [design-decisions.md](design-decisions.md) for the rationale behind each.

## AI Approach

There is one model call per completed full-quality search: interpretation. It
turns the natural-language request into a structured plan. Retrieval and
explanations then run locally against the catalogue.

This keeps the expensive, non-deterministic part narrow while still allowing
requests like "a premium anniversary gift for my parents" to map onto catalogue
language. See [ai-approach.md](ai-approach.md).

## Product Catalogue

The app ships with a committed catalogue of 1,738 real products from Amazon.in,
Myntra and a Kaggle Amazon-fashion archive, with product URLs, images, prices,
ratings and enriched shopping attributes.

The live app does not need an Apify token or catalogue-generation credentials.
Those are only needed for offline rebuild scripts. See [catalogue.md](catalogue.md).

## Testing

```bash
cd backend
uv run pytest
```

```bash
cd frontend
npm run build
npm run lint
```

The backend tests cover retrieval gates, degraded mode, normalization, context
resolution and API contracts. See [testing.md](testing.md).

## Future Scope

Known limitations and future work are tracked in [scope.md](scope.md). The most
important items are broader catalogue coverage, deterministic clarification
rules, lower latency, deployment/CI, and an evaluation harness for recommendation
quality.

## Documentation

| File | Covers |
|---|---|
| [setup.md](setup.md) | Setup, environment variables and local checks |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Runtime design, data flow, boundaries and constraints |
| [api.md](api.md) | API contract and request/response examples |
| [design-decisions.md](design-decisions.md) | Engineering and UX decisions |
| [ai-approach.md](ai-approach.md) | LLM, retrieval, context and fallback strategy |
| [catalogue.md](catalogue.md) | Product data, enrichment and indexing |
| [testing.md](testing.md) | Test strategy and regression cases |
| [scope.md](scope.md) | Known limitations and future scope |
| [demo.md](demo.md) | Suggested demo video flow and deployment status |

## Status

Local application only. No live deployment URL is included yet.
