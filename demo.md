# Demo

The project is currently local only. There is no live application URL yet.

## Suggested Video Flow

1. Start the backend and frontend.
2. Open <http://localhost:3000>.
3. Submit:

```text
I am going for a trek to Hampta Pass in the last week of October for one week.
Please find me trekking essentials and clothing.
```

4. Show the streamed stages:
   - reading the request
   - checking conditions
   - searching the catalogue
5. Show the resolved trip context, assumptions and context variables.
6. Show grouped recommendations with explanations and product links.
7. Submit an incomplete request:

```text
Find me traditional wear for my friend's wedding.
```

8. Show clarification chips or visible assumptions.
9. Show a catalogue-gap case where the UI reports `Could not cover` instead of
   forcing unrelated products into the answer.

## What to Highlight

- Single-page conversational UX.
- Intent interpretation beyond keyword matching.
- Grouped recommendations by shopping need.
- Explanation attached to every product.
- Real product links from the catalogue.
- Graceful fallback for missing context.
- Transparent degraded mode when the LLM is unavailable.

## Deployment Status

Not deployed yet. Future deployment work is listed in [scope.md](scope.md).
