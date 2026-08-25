# Setup

This project runs as two local processes: a FastAPI backend and a Next.js
frontend.

## Requirements

- Python 3.12
- `uv`
- Node.js 20+
- npm

## Backend

```bash
cd backend
cp .env.example .env
uv sync --extra dev
uv run uvicorn app.main:app --reload --port 8000
```

The backend starts on <http://localhost:8000>. API docs are available at
<http://localhost:8000/docs>.

Readiness:

```bash
curl http://localhost:8000/api/v1/health
```

## Frontend

```bash
cd frontend
cp .env.local.example .env.local
npm install
npm run dev
```

The frontend starts on <http://localhost:3000>.

## Environment Variables

| File | Variable | Required | Purpose |
|---|---|---:|---|
| `backend/.env` | `LLM_PROVIDER` | No | `anthropic` (default), `openrouter`, or `local` -- picks the adapter, see [ai-approach.md](ai-approach.md) |
| `backend/.env` | `ANTHROPIC_API_KEY` | No | Used when `LLM_PROVIDER=anthropic`. Without a configured provider the backend runs in transparent degraded mode |
| `backend/.env` | `OPENROUTER_API_KEY` | No | Used when `LLM_PROVIDER=openrouter` |
| `backend/.env` | `LOCAL_LLM_BASE_URL` | No | Used when `LLM_PROVIDER=local` (e.g. Ollama). No key needed; free, but materially weaker structured-output reliability than a hosted model |
| `backend/.env` | `INTERPRET_MODEL` | No | Defaults to `claude-haiku-4-5`, which only works with `LLM_PROVIDER=anthropic`. Set an OpenRouter-routed model id when `LLM_PROVIDER=openrouter`, or an `ollama list` model name when `LLM_PROVIDER=local` -- there is no cross-provider default |
| `backend/.env` | `INTERPRET_TIMEOUT_S` | No | Seconds allowed for the one interpretation call (default 30). A real interpretation measures ~11s locally but 30-50s+ on throttled free-tier hosting, where the default silently degrades every search -- set 90 there |
| `backend/.env` | `LLM_FALLBACK_PROVIDER` | No | `anthropic`, `openrouter`, or `local` -- a second provider tried when the primary is unconfigured or a live call fails. Must differ from `LLM_PROVIDER` and needs its own key set |
| `backend/.env` | `FALLBACK_INTERPRET_MODEL` | No | Required if `LLM_FALLBACK_PROVIDER` is set -- a model id valid for *that* provider, not `INTERPRET_MODEL`'s |
| `backend/.env` | `CORS_ORIGINS` | No | Comma-separated browser origins allowed to call the API |
| `backend/.env` | `APIFY_TOKEN` | No | Only needed to rebuild raw catalogue data |
| `frontend/.env.local` | `NEXT_PUBLIC_API_BASE_URL` | No | Defaults to `http://localhost:8000` |

The committed catalogue is enough to run the app. `APIFY_TOKEN` and catalogue
enrichment credentials are only needed for the offline data rebuild scripts.

## Running Without an API Key

`ANTHROPIC_API_KEY` is optional. Without it, the app still boots and returns
catalogue matches using keyword interpretation. Every response sets
`degraded_mode: true`, and the UI shows a reduced-mode notice.

This is intentional: the demo remains usable offline, but weaker answers are
not passed off as full AI recommendations.

## First Run Note

The semantic embedding model `sentence-transformers/all-MiniLM-L6-v2` downloads
on first backend startup and is cached locally. If it cannot be downloaded, the
backend falls back to hashing-based matching and reports degraded retrieval.

## Local Checks

Backend tests:

```bash
cd backend
uv run pytest
```

Frontend production build:

```bash
cd frontend
npm run build
```

Frontend lint:

```bash
cd frontend
npm run lint
```
