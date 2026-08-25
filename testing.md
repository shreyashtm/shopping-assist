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
```

There is not yet a Playwright or component-test suite. That is listed as future
scope in [scope.md](scope.md).
