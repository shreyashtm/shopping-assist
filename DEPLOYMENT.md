# Deployment

**Current state — both services are live:**

| Surface | Platform | URL |
|---|---|---|
| Frontend | Vercel | <https://shopping-assist-iota.vercel.app> |
| Backend | Railway | <https://shopping-assist-production.up.railway.app> |

Both auto-deploy from `main`. The rest of this document is the runbook for
reproducing that setup (or moving it elsewhere), plus the reasoning behind each
platform choice.

## What actually went wrong the first time

Recorded because these cost real debugging time and none of them are obvious
from a green build log:

- **The backend was first deployed to Render's free tier (512 MB) and it did
  not hold up.** Health checks intermittently took 14s or timed out entirely
  under `torch` + `sentence-transformers`. Moved to Railway, whose trial tier
  gives 1 GB. See the RAM note below.
- **A trailing slash in `CORS_ORIGINS` silently rejected every browser
  request.** Browser `Origin` headers never carry a trailing slash and
  Starlette matches exactly. Fixed in `cors_origin_list` (`app/core/config.py`),
  which now strips it -- but the symptom was an unhelpful "Could not reach the
  assistant" in the UI with a CORS error only visible in the browser console.
- **`INTERPRET_MODEL` must match the selected provider.** An Anthropic model id
  under `LLM_PROVIDER=openrouter` fails every call and silently degrades the
  app to keyword matching. There is no cross-provider default.
- **Env var changes need a redeploy.** Vercel bakes `NEXT_PUBLIC_*` in at build
  time, so saving a variable does nothing until you rebuild. Railway does not
  always auto-redeploy on a variable change either -- check the Deployments tab.
- **Railway needs the root directory set to `backend`.** Otherwise its builder
  analyses the repo root, finds no Python project, and fails with "could not
  determine how to build the app".
- **`INTERPRET_TIMEOUT_S` (default 30s) is too low for throttled hosting.** A
  real interpretation takes ~11s locally but 30-50s+ on free-tier shared CPU,
  so the default silently aborted calls that would have succeeded and every
  search degraded to keyword matching. Set 90s.
- **Free-tier inference does not work for this workload.** Measured across
  every OpenRouter free model advertising structured output: at best 1
  successful interpretation in 4, at 37-67s each, plus a 50 request/day cap.
  Budget for a paid model -- roughly $0.0006 per search. See
  [performance.md](performance.md).

## Before you start: two things worth knowing

**The backend needs a real server, not a serverless function.** It loads
`sentence-transformers` (which pulls in `torch`, ~514 MB installed) and keeps
the embedding model and the whole product catalogue in process memory for the
life of the process. Vercel/Netlify-style functions cold-start per request and
typically cap memory well below what this needs — use a platform that runs a
persistent container (Render, Railway, Fly.io, a VPS). Budget **at least 1 GB
RAM**; a 512 MB free tier will likely OOM on startup.

**Latency is dominated by which interpretation model you configure**, not by
the app's own work. A full warm search on `claude-haiku-4-5` measures ~7.5s
end-to-end; every OpenRouter free model tested managed at best 1 successful
interpretation in 4, at 37-67s each. Use a **paid** model as the primary.

A free model is still defensible as `LLM_FALLBACK_PROVIDER`: it only engages
when the primary actually fails, so it costs nothing in the normal path, and a
1-in-4 chance of a real answer beats keyword matching. The premium is that an
outage takes ~40s longer to surface. Resilience or speed -- both are
reasonable, but choose deliberately. See [performance.md](performance.md).

---

## 1. Backend — Railway (the current deployment)

Web-dashboard flow, no CLI or login needed.

1. Go to <https://railway.app> → **New Project** → **Deploy from GitHub repo**
   → select this repo.
2. Service Settings → **Root Directory**: `backend`. Not optional — without it
   the builder analyses the repo root, finds no Python project, and fails.
3. Service Settings → **Build**:
   - Build command: `pip install uv && uv sync --extra dev --no-dev`
   - Start command: `uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT`

   Railway injects `$PORT`; bind to it, not to 8000. Its Python auto-detection
   does not understand `uv`-managed projects, so set both commands explicitly.
4. **Variables** tab:

   | Key | Value |
   |---|---|
   | `LLM_PROVIDER` | `anthropic` |
   | `INTERPRET_MODEL` | `claude-haiku-4-5` — must be valid for *that* provider |
   | `ANTHROPIC_API_KEY` | your key |
   | `INTERPRET_TIMEOUT_S` | `90`. The 30s default aborts calls that would have succeeded on throttled hosting |
   | `CORS_ORIGINS` | the Vercel URL from step 3, e.g. `https://your-project.vercel.app` |

   Optional, and only worth setting if the second provider is also **paid**:
   `LLM_FALLBACK_PROVIDER` (must differ from the primary) plus
   `FALLBACK_INTERPRET_MODEL` (a model id for the *fallback's* provider, not
   the primary's).

5. Settings → **Networking** → **Generate Domain**. Railway assigns no public
   URL until you ask for one — the `*.railway.internal` address it shows by
   default is private and unreachable from a browser.
6. Confirm:
   ```bash
   curl https://your-service.up.railway.app/api/v1/health
   ```
   Check `catalogue_loaded: true`, `embeddings_ready: true`, and
   `llm_configured: true`. If `interpret_model` in that response is not the
   value you set, the variables did not take effect — redeploy.

**RAM**: Railway's 30-day trial gives 1 GB and shared vCPU on a one-time $5
credit — a 24/7 backend will likely exhaust that credit well before 30 days.
The ongoing Free plan afterwards drops to 0.5 GB, the same ceiling that made
Render's free tier unusable here. The Hobby plan (~$5/month + usage) is the
real answer if this needs to stay up.

## 2. Alternative backend — Render

Also a persistent container, and needs no config file for a project this size.
Used for this project's first deployment; **its 512 MB free tier could not run
this workload** (intermittent 14s health checks and timeouts under `torch` +
`sentence-transformers`), so budget for the paid Starter tier if you go this
route.

1. Go to <https://dashboard.render.com> → **New** → **Web Service**.
2. Connect your GitHub repo, select the `backend/` directory as the root
   (Render's "Root Directory" setting).
3. Environment: **Python 3**.
4. Build command:
   ```bash
   pip install uv && uv sync --extra dev --no-dev
   ```
5. Start command:
   ```bash
   uv run uvicorn app.main:app --host 0.0.0.0 --port $PORT
   ```
   Render injects `$PORT`; the app must bind to it, not to 8000.
6. Instance type: pick at least **1 GB RAM** (Render's free tier is 512 MB —
   too small; use the cheapest paid "Starter" tier or equivalent).
7. Environment variables (Render → your service → Environment):
   | Key | Value |
   |---|---|
   | `ANTHROPIC_API_KEY` | your key |
   | `CORS_ORIGINS` | leave blank for now — you'll set this in step 4, after the frontend has a URL |
8. Deploy. First boot will download the `all-MiniLM-L6-v2` model (~90 MB) from
   Hugging Face on demand — expect the first request after a cold start to be
   slower than normal. This re-downloads on every fresh deploy unless you
   configure a persistent disk for the HF cache (optional; not required to
   function, just avoids repeating the download).
9. Once live, note the URL Render gives you, e.g.
   `https://your-backend.onrender.com`. Confirm it works:
   ```bash
   curl https://your-backend.onrender.com/api/v1/health
   ```
   Check `catalogue_loaded: true` and `embeddings_ready: true` in the response.

**Other options:**

### Fly.io

`fly launch` from `backend/`, needs a `Dockerfile`. Same RAM floor applies.
Note that Fly removed its permanent free tier in 2024 — new accounts get only
a short trial, so this is a paid option in practice.

### Oracle Cloud Free Tier

The only genuinely always-free option with enough RAM (ARM instances go well
past 1 GB). The tradeoff is that it is a raw VM, not a connect-your-repo
platform: you provision the machine and install Python, `uv`, a reverse proxy
and a service unit yourself, and wire up your own deploys.

## 3. Frontend — Vercel (recommended)

Next.js's own platform; zero-config for an App Router project.

1. Go to <https://vercel.com/new>, import the same GitHub repo.
2. Set **Root Directory** to `frontend/`.
3. Framework preset: Vercel auto-detects Next.js — leave defaults.
4. Environment variable:
   | Key | Value |
   |---|---|
   | `NEXT_PUBLIC_API_BASE_URL` | the backend URL from step 1.6, e.g. `https://your-service.up.railway.app` |
5. Deploy. Vercel gives you a URL like `https://your-project.vercel.app`.

`NEXT_PUBLIC_*` values are baked in at **build** time, so changing this
variable does nothing until you trigger a fresh deploy (Deployments → ⋯ →
Redeploy). A stale value here is the usual cause of the UI reporting
"Could not reach the assistant" or a bare 404.

## 4. Close the loop: CORS

The backend only accepts browser requests from origins listed in
`CORS_ORIGINS`. Go back to the backend service (Railway → Variables, or
Render → Environment) and set:

```
CORS_ORIGINS=https://your-project.vercel.app
```

Redeploy the backend (Render redeploys automatically on env var changes, or
trigger manually). Without this step the frontend will load but every
`/api/v1/recommend` call will fail with a CORS error in the browser console.

If you also want to keep testing against localhost, comma-separate:

```
CORS_ORIGINS=https://your-project.vercel.app,http://localhost:3000
```

## 5. Verify end-to-end

1. Open the Vercel URL in a browser.
2. Submit one of the example queries.
3. Confirm a result comes back (or a clarify round-trip, then a result).
4. Check the browser console and Network tab for errors — a CORS or
   `NEXT_PUBLIC_API_BASE_URL` mismatch is the most likely failure at this
   stage.

## 6. Update the docs with the live URL

Done for the current deployment — the URLs are in [README.md](README.md) and
[demo.md](demo.md). If you redeploy elsewhere, update both, plus
`NEXT_PUBLIC_API_BASE_URL` on Vercel and `CORS_ORIGINS` on the backend.

## Costs, roughly

- **Backend**: Railway Hobby ~$5/month + usage, or Render Starter ~$7/month.
  No free tier is viable long-term at this memory footprint.
- **Frontend**: Vercel's free tier is enough for this traffic level.
- **LLM API**: pay-per-use, one interpretation call per completed search (see
  [ai-approach.md](ai-approach.md)). OpenRouter's free models cost nothing but
  are slow and unreliable enough that they are better used as the *fallback*
  than the primary — see [scope.md](scope.md#latency).

## If something fails

| Symptom | Likely cause |
|---|---|
| Backend deploy OOMs / crashes on boot | Instance below 1 GB RAM |
| Frontend loads, every search fails, CORS error in console | `CORS_ORIGINS` not set to the Vercel URL, or backend not redeployed after setting it |
| Frontend loads, search fails with "Could not reach the assistant" or a 404 | `NEXT_PUBLIC_API_BASE_URL` wrong or stale (Vercel bakes it in at build time — redeploy after changing it), or backend still deploying |
| `/api/v1/health` shows `catalogue_loaded: false` | `backend/data/products.json` wasn't committed — check `git ls-files backend/data/` |
| Every response is `degraded_mode: true` | No LLM key set for the selected `LLM_PROVIDER`, credit exhausted, or `INTERPRET_MODEL` is not a valid id for that provider |
| `llm_configured: true` but searches still degrade | `INTERPRET_MODEL` names a model the provider rejects — e.g. an embedding model (`...-embedding-...`) where a chat model is required |
| Health check itself is slow (10s+) or times out | Instance under-resourced for `torch` + `sentence-transformers`; move to ≥1 GB RAM |
| Searches take a minute or more | A free-tier interpretation model. Switch to a paid one; do not keep the free model as a fallback |
| `degraded_mode: true` with `llm_calls: 0`, after ~30s or more | `INTERPRET_TIMEOUT_S` is aborting a call that would have succeeded. Set 90 on throttled hosting |
| Build fails: "could not determine how to build the app" | Root directory not set to `backend` on the platform |
