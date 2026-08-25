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

### Free-tier models, measured

Every OpenRouter free model that advertises structured output was tested with
the real interpretation prompt. None was usable as the sole provider:

| Model | Successful interpretations | Latency |
|---|---|---|
| `nvidia/nemotron-3-super-120b-a12b:free` | **1 of 4** | 37-67s (one run 115s) |
| `dots-studio/dots-3-note-preview:free` | 0 of 2 | 52s |
| `openrouter/free` | 0 of 2 | 103s |
| `z-ai/glm-5.2:free` | 0 of 2 | 0.5s (rate-limited) |

The failures are structured-output failures -- the model returns text that does
not conform to the JSON schema -- so they surface as `degraded_mode: true`
after a long wait. Free tiers are also capped at 50 requests/day on an unfunded
account, and the later measurements above are partly confounded by that cap;
the nemotron figure was taken on a fresh quota.

The paid variant of the same model (`nvidia/nemotron-3-super-120b-a12b`,
without the `:free` suffix) costs roughly **$0.0006 per search** at ~2k input
and ~1k output tokens. That is the cheapest way to make this reliable.

**Conclusion, stated plainly:** this workload cannot run on free-tier
inference. It is not a tuning problem.

### Timeouts must clear a *real* call, not a trivial one

`INTERPRET_TIMEOUT_S` defaults to 30s. On a developer machine a real
interpretation takes 10.7-11.4s, so 30s looks generous. On throttled
free-tier hosting the same call measured 30-50s+, so the default silently
cut off calls that would have succeeded and every search fell to keyword
matching. Raise it (90s) on any host with shared or throttled CPU.

This is the same mistake twice: an 8s fallback deadline was also calibrated
from a trivial 2.5s call before being removed entirely. Calibrate against the
real workload.

### Practical configuration

Put a fast, reliable model first and keep the free one as the fallback:

```
LLM_PROVIDER=anthropic
INTERPRET_MODEL=claude-haiku-4-5
INTERPRET_TIMEOUT_S=90
```

Measured at 10.7-11.4s per interpretation and reliable across every run.
`nvidia/nemotron-3-super-120b-a12b` on OpenRouter (without the `:free` suffix)
is the cheaper alternative at roughly $0.0006 per search.

Either way the model must be a **paid** one -- see the free-tier table above.
A free model as *fallback* is worse than no fallback at all: it adds ~40s of
waiting and then fails anyway.

Falling through between providers is driven by failure, not elapsed time:
only a rejected key, exhausted credit, a rate limit or a transport error
moves to the next provider. A slow-but-working provider is left alone --
an earlier 8s deadline aborted `claude-haiku-4-5` mid-call, since a real
interpretation takes 10.7-11.4s, and sent every search to keyword matching.

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
