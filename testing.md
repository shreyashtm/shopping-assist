# Testing

Backend tests are the main automated quality guard.

Run them with:

```bash
cd backend
uv run pytest
```

The suite is written around important behavioural claims rather than raw line
coverage.

| File | Covers |
|---|---|
| `test_retrieval.py` | category gating, multiplicative boosts, penalties, dedupe, per-phrase search |
| `test_robustness.py` | LLM outage, no provider, cache TTL/LRU/keying, off-topic decline, filter precedence |
| `test_normalize.py` | title cleaning, brand guessing, MRP sanity, dedupe keys |
| `test_context.py` | elevation corroboration, nearest-place selection, provenance |
| `test_api_contract.py` | response shape, clarify vs results, answers, skip |
| `test_context_slots.py` | which planning variables count as needed |

Important regression examples:

- `test_category_gate_beats_high_similarity`
- `test_boosts_cannot_rescue_an_irrelevant_product`
- `test_resolve_climate_rejects_bad_proposal`
- `test_unmapped_required_slot_is_reported_not_substituted`

`test_api_contract.py` runs end to end against the real catalogue. With an API
key configured, it exercises the live model path; without a key, it still checks
the fallback contract.

Frontend checks:

```bash
cd frontend
npm run build
npm run lint
npm run test:run
```

Unit and component tests run on Vitest + React Testing Library (`vitest.config.mts`),
44 tests across:

| File | Covers |
|---|---|
| `lib/thread.test.ts` | follow-up composition length budgeting, established-context formatting, chip-answer labels |
| `components/ProductCard.test.tsx` | price/discount formatting, brand fallback, `link_status` provenance badges |
| `components/QueryBar.test.tsx` | Enter-to-send vs Shift+Enter, minimum-length gate, busy state, example queries |
| `components/ResultsView.test.tsx` | grouped results, unfilled-slot reporting, degraded-mode notice, empty-catalogue state |

`npm run test` runs Vitest in watch mode for local development.

There is not yet a Playwright/E2E suite covering the full clarify round-trip
against a live backend. That remains future scope in [scope.md](scope.md).
