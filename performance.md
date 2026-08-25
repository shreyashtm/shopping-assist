# Performance

Measured, not estimated. Numbers below come from the live deployment
(Railway backend, Vercel frontend) and from local timing of the offline
stages, on 2026-08-25 with a catalogue of 1,738 products.

The short version: **one LLM call dominates everything.** All the local work
combined — loading the catalogue, embedding the query, scoring all 1,738
products, dedupe and explanation — is a fraction of a second. Which
interpretation model you configure decides whether a search feels instant or
takes a minute.

## Startup (once per process)

| Stage | Time |
|---|---|
| Import catalogue module | 0.13s |
| `Catalogue.load()` — 1,738 products + `embeddings.npy` | 0.02s |
| Embedding model load (`all-MiniLM-L6-v2`, already cached on disk) | 6.83s |

The embedding model is the whole startup cost. All four loaders run in
`main.py`'s lifespan, so the container finishes booting *ready* — the embedder
is warmed there rather than lazily inside the first search. That single change
took the first real search from **22.6s to 10.4s**; without it, the first user
after every deploy paid the model load personally.

On a **fresh deploy** startup is slower still, because the ~90 MB model is
downloaded from Hugging Face rather than read from disk — then cached for the
life of that container.

A platform that spins containers down when idle pays all of this again on every
wake. Render's free tier does this after ~15 minutes; Railway runs a persistent
container and does not. Note that a `/health` ping does **not** prevent it:
health only reads already-loaded state and never touches the embedder, so
keep-warm cron jobs keep a container alive without keeping it *warm* in the
sense that matters.

## Per request

| Stage | Time | Notes |
|---|---|---|
| Cache hit | **~0.4s** | Whole response, no LLM call, no retrieval |
| `/api/v1/health` | ~0.36s | Round trip to Railway, no real work |
| Embed one query | 0.43s | Local `all-MiniLM-L6-v2` |
| Vector search across all 1,738 products | **0.0009s** | One NumPy matrix multiply |
| Dedupe, suitability, explanation | negligible | Deterministic Python |
| **Interpretation (the one LLM call)** | **2s – 67s** | Entirely model-dependent, see below |

Sub-millisecond vector search is why there is no vector database here: at this
catalogue size a NumPy dot product over an in-memory matrix is simply faster
than a network round trip to one would be. See
[ARCHITECTURE.md](ARCHITECTURE.md#storage) for where that stops being true.

## The interpretation call is the whole story

Measured end-to-end (`meta.latency_ms`, the whole request), not the model call
in isolation:

| Model | Full search, warm | First search after boot | Reliability |
|---|---|---|---|
| `claude-haiku-4-5` (Anthropic, paid) | **7.5s** | 10.4s | Reliable |
| `nvidia/nemotron-3-super-120b-a12b:free` (OpenRouter, free) | 12–14.5s *when it answered first try* | — | **Returned unparseable JSON on roughly half of attempts**; 37–67s for the interpretation call alone |

All measured with the real interpretation prompt — a 5,607-character system
prompt plus the live catalogue taxonomy — not a toy request. An isolated
model call is much faster than these numbers suggest; the prompt is the
difference, which is why only end-to-end figures are quoted here.

The free model's failure mode matters as much as its latency: a
structured-output failure falls through to `offline_interpret()`, so the user
waits a long time and *then* gets a keyword-quality answer, honestly marked
`degraded_mode: true`. Two turns of a clarification round-trip, each with a
stalled first attempt, is how a search reaches a minute or more.

### Practical configuration

Put a fast, reliable model first and keep the free one as the fallback:

```
LLM_PROVIDER=anthropic
INTERPRET_MODEL=claude-haiku-4-5
LLM_FALLBACK_PROVIDER=openrouter
FALLBACK_INTERPRET_MODEL=nvidia/nemotron-3-super-120b-a12b:free
```

Reversing this is what makes the app feel slow. The free tier cannot be both
free and fast here — that is a real tradeoff, not a tuning problem.

`FALLBACK_AFTER_S` (default 8s) bounds the damage either way: every hop except
the last is abandoned after that long, so a stalling primary no longer burns
the entire `INTERPRET_TIMEOUT_S` before the working provider is even tried.
The last hop keeps the full budget — nothing follows it, so giving up early
there would only lose answers.

## What is already fast, and why

- **Retrieval is not an LLM call.** Product matching is local vector search
  over embeddings computed once at build time (`scripts/build_index.py`).
- **Explanations are not an LLM call.** They are composed deterministically
  from stored product attributes and retrieval evidence
  (`services/explain.py`).
- **Answering a clarification is not a new interpretation.** Chip answers are
  merged into the existing structured query without another model call.
- **Identical repeat queries are cached**, returning in ~0.4s with
  `cached: true` in the response metadata. Degraded responses are deliberately
  *not* cached, so a transient provider outage cannot poison later results.

Every response reports its own `latency_ms`, `llm_calls`, `cached` and
`degraded_mode` in `meta` — the numbers above are reproducible from any
request.

## Known remaining costs

- **Clarification doubles the LLM cost of a request**, because the follow-up
  turn re-interprets the enriched query. [scope.md](scope.md) tracks the
  planned fix: return a best-effort recommendation on the first turn instead
  of questions alone.
- **Free-tier hosting constrains everything above.** See
  [DEPLOYMENT.md](DEPLOYMENT.md) for the RAM floor and why a 512 MB instance
  could not run this workload at all.
