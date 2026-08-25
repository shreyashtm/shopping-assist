# API

The backend exposes a small versioned API under `/api/v1`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/api/v1/recommend` | Blocking recommendation request |
| `POST` | `/api/v1/recommend/stream` | Same pipeline streamed through server-sent events |
| `GET` | `/api/v1/health` | Runtime capability and readiness check |

## Recommendation Request

```json
{
  "query": "I need a premium gifting hamper for my parents 25th anniversary next month",
  "answers": [],
  "skip_clarification": false
}
```

Fields:

- `query`: natural-language shopping request.
- `answers`: machine values from clarification chips, such as
  `["gender:men", "price_max:5000"]`.
- `skip_clarification`: set when the user chooses "Just show me now".
- `filters`: optional explicit constraints, applied after inferred filters and
  tapped answers.

Precedence is:

```text
inferred intent -> tapped answers -> explicit filters
```

## Response Modes

One endpoint returns two modes because clarification and recommendations are two
states of the same request.

`mode: "clarify"` returns questions and no recommendation groups:

```json
{
  "mode": "clarify",
  "intent_summary": "Traditional wedding wear for a March event.",
  "questions": [
    {
      "slot": "gender",
      "question": "Who are you shopping for?",
      "options": [
        { "label": "Men", "value": "gender:men" },
        { "label": "Women", "value": "gender:women" }
      ],
      "allow_multiple": false
    }
  ],
  "groups": []
}
```

`mode: "results"` returns grouped recommendations:

```json
{
  "mode": "results",
  "intent_summary": "Essentials and clothing for a one-week Hampta Pass trek.",
  "groups": [
    {
      "name": "Layering & Insulation",
      "why_needed": "Late October at altitude needs warm, packable layers.",
      "items": [
        {
          "reason": "Rated for sub-zero conditions and matched to the trek use case.",
          "match_score": 0.98,
          "product": {
            "title": "Example insulated jacket",
            "product_url": "https://example.com/product"
          }
        }
      ]
    }
  ],
  "unfilled_slots": []
}
```

## Metadata

Every response includes `meta`:

```json
{
  "latency_ms": 21340,
  "llm_calls": 1,
  "cached": false,
  "degraded_mode": false,
  "catalogue_size": 1738,
  "notes": []
}
```

This makes weaker paths visible. If the LLM or semantic embedding model is
unavailable, the response is still returned, but `degraded_mode` and `notes`
explain the fallback.

## Example Curl

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find me good traditional wear for my friend wedding in March next year."
  }'
```

Clarification follow-up:

```bash
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Content-Type: application/json" \
  -d '{
    "query": "Find me good traditional wear for my friend wedding in March next year.",
    "answers": ["gender:men", "price_max:5000"]
  }'
```

## Streaming Events

`POST /api/v1/recommend/stream` emits real stage boundaries:

```text
event: stage
data: {"stage":"interpreting"}

event: stage
data: {"stage":"checking conditions"}

event: stage
data: {"stage":"searching"}

event: result
data: { ...RecommendResponse }
```

The frontend uses `fetch` plus `ReadableStream` instead of `EventSource` because
the request body should remain a POST payload rather than being placed into a
URL.
