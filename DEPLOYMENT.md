# Deployment

Manual steps to take the project live. Nothing here has been executed —
this is a guide for you to run through yourself.

## Before you start: two things worth knowing

**The backend needs a real server, not a serverless function.** It loads
`sentence-transformers` (which pulls in `torch`, ~514 MB installed) and keeps
the embedding model and the whole product catalogue in process memory for the
life of the process. Vercel/Netlify-style functions cold-start per request and
typically cap memory well below what this needs — use a platform that runs a
persistent container (Render, Railway, Fly.io, a VPS). Budget **at least 1 GB
RAM**; a 512 MB free tier will likely OOM on startup.

**Nothing is pushed to a remote yet.** The repo has one local commit (the
scaffold) and a large uncommitted working tree — all of `backend/app/`,
`backend/data/`, `backend/scripts/`, `backend/tests/`, `frontend/app/`,
`frontend/components/`, `frontend/lib/`, and the doc files. Step 1 below
covers this.

---

## 1. Commit and push

```bash
cd /Users/shreyash/PROJECTS/confluxe-assignment
git add -A
git status                 # review what's staged before committing
git commit -m "feat: complete recommendation pipeline, catalogue, frontend, docs"
```

Create a GitHub repo (via the GitHub website, or `gh repo create` if you have
the CLI), then:

```bash
git remote add origin <your-repo-url>
git push -u origin main
```

## 2. Backend — Render (recommended)

Render runs a persistent container, has a straightforward free/starter path,
and needs no config file for a project this size.

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

**Alternatives**, if you'd rather not use Render: **Railway** (similar flow,
`railway up` from the `backend/` directory) or **Fly.io** (`fly launch` from
`backend/`, needs a `Dockerfile` — ask if you want one written). Same RAM
floor applies to both.

## 3. Frontend — Vercel (recommended)

Next.js's own platform; zero-config for an App Router project.

1. Go to <https://vercel.com/new>, import the same GitHub repo.
2. Set **Root Directory** to `frontend/`.
3. Framework preset: Vercel auto-detects Next.js — leave defaults.
4. Environment variable:
   | Key | Value |
   |---|---|
   | `NEXT_PUBLIC_API_BASE_URL` | the Render backend URL from step 2.9, e.g. `https://your-backend.onrender.com` |
5. Deploy. Vercel gives you a URL like `https://your-project.vercel.app`.

## 4. Close the loop: CORS

The backend only accepts browser requests from origins listed in
`CORS_ORIGINS`. Go back to Render → your service → Environment, and set:

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

Once confirmed working, add the URL to `README.md` (the assignment asks for
it under Deliverables, "if deployed"). This repo's docs are otherwise
final per your note — this is the one line that becomes true only after
deployment, so it can't have been written earlier.

## Costs, roughly

- Render Starter (backend, 1 GB): ~$7/month, no free tier viable at this
  memory footprint.
- Vercel (frontend): free tier is enough for this traffic level.
- Anthropic API: pay-per-use, driven by search volume — see `ai-approach.md`
  for the per-search cost.

## If something fails

| Symptom | Likely cause |
|---|---|
| Backend deploy OOMs / crashes on boot | Instance below 1 GB RAM |
| Frontend loads, every search fails, CORS error in console | `CORS_ORIGINS` not set to the Vercel URL, or backend not redeployed after setting it |
| Frontend loads, search fails with "Could not reach the assistant" | `NEXT_PUBLIC_API_BASE_URL` wrong, or backend still deploying/asleep |
| `/api/v1/health` shows `catalogue_loaded: false` | `backend/data/products.json` wasn't committed — check `git ls-files backend/data/` |
| Every response is `degraded_mode: true` | `ANTHROPIC_API_KEY` not set on the backend service, or account credit balance is low |
