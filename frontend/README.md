# Frontend

Next.js 16 (App Router, React 19, Tailwind v4) single-page client for the
Personal Shopping Assistant.

```bash
cp .env.local.example .env.local   # NEXT_PUBLIC_API_BASE_URL
npm install
npm run dev
```

Needs the backend running on the URL in `.env.local` (default
`http://localhost:8000`).

Setup, architecture, design decisions and limitations for the whole project are
in the [root README](../README.md).

## Layout

| Path | Purpose |
|---|---|
| `app/page.tsx` | The only route. Conversation thread state and turn orchestration |
| `app/layout.tsx` | Fonts, metadata, pre-paint theme script |
| `app/globals.css` | Dual palette tokens, zero-radius system, motion |
| `components/` | Thread turns, clarify panel, results view, product cards |
| `lib/api.ts` | Fetch client and SSE stream parsing |
| `lib/thread.ts` | Turn model and follow-up composition |
| `lib/types.ts` | Hand-written mirror of the backend Pydantic schemas |

`lib/types.ts` is written by hand rather than generated: the surface is small,
and an explicit copy makes contract drift show up as a TypeScript error at the
call site.

```bash
npm run build     # production build + typecheck
npm run lint
```
