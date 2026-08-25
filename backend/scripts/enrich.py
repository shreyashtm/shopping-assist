"""Fill in the shopping attributes that retailers do not publish.

Listing pages give a title, a price and a category -- and nothing that lets a
recommender reason about *fit for purpose*. Ajio's `fabric`/`occasion` fields
come back empty in search mode, Amazon has no attributes and no brand field at
all, and none of the retailers provides a description.

So the attributes retrieval depends on -- what season a garment suits, what
activity it is for, how cold a jacket is good to -- are inferred here, once,
offline, and committed with the catalogue. The runtime never pays for this: a
search costs one embedding lookup, not an LLM call per product.

Provider-agnostic through the same `LLMProvider` protocol the runtime uses
(app/adapters/llm/), same as scripts/enrich_suitability.py -- Anthropic,
OpenRouter, or a local model server, picked with `--provider`. This means
enrichment does not have to mean spending API credits: OpenRouter's free-tier
models run this at $0 (`--list-free-models` shows which ones right now), and a
local Ollama model runs it at $0 with no external account at all.

Anthropic alone keeps its Batch API path (50% cheaper, queued up to 24h) since
that is a genuine Anthropic-specific cost optimization with no equivalent on
the other providers. Every other path -- Anthropic `--sync`, OpenRouter, and
local -- runs synchronously and saves `data/products_enriched.json` after every
batch: a batch's cost (or a rate-limited free model's quota) is spent the
moment the call succeeds, whether or not the process is still alive to record
it, so saving only at the end would mean an interrupted run throws away
already-paid-for work.

    uv run python scripts/enrich.py                                    # Anthropic Batch API, 50% cheaper, queued
    uv run python scripts/enrich.py --sync                             # Anthropic, immediate
    uv run python scripts/enrich.py --provider openrouter --model z-ai/glm-5.2:free
    uv run python scripts/enrich.py --provider local --model llama3.2:3b --limit 24
    uv run python scripts/enrich.py --list-free-models                 # what's free on OpenRouter right now
"""

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.adapters.llm.base import LLMProvider, LLMUnavailable  # noqa: E402
from enrich_suitability import build_provider, list_free_models  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
IN_FILE = BASE / "data" / "products_normalized.json"
OUT_FILE = BASE / "data" / "products_enriched.json"

BATCH_SIZE = 12

# Anthropic Batch API state: where the in-flight batch id is parked so a
# re-run resumes instead of submitting a second batch. Anthropic-only -- the
# other providers have no batch queue to resume.
STATE_FILE = BASE / "data" / ".enrich_batch.json"
MAX_WAIT_SECONDS = 60 * 60 * 24

VALID_GENDERS = ["men", "women", "unisex", "kids"]
VALID_SEASONS = ["summer", "monsoon", "winter", "all-season"]

# Hand-written rather than generated from the Pydantic model: model_json_schema()
# emits $defs/$ref, which the structured-output validator does not accept. Flat
# and explicit is the working shape.
ENRICHMENT_SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "brand": {
                        "type": ["string", "null"],
                        "description": "Real brand name if the title contains one "
                        "(e.g. 'Boldfit', 'Lux Cottswool'). Null when the listing has "
                        "no brand -- never promote a descriptive word like 'Winter'.",
                    },
                    "description": {
                        "type": "string",
                        "description": "One or two factual sentences a shopper would "
                        "find useful. Only what the title and category support.",
                    },
                    "gender": {"type": "string", "enum": VALID_GENDERS},
                    "season": {
                        "type": "array",
                        "items": {"type": "string", "enum": VALID_SEASONS},
                    },
                    "use_case": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Activity tags, lowercase-hyphenated: trekking, "
                        "wedding, gifting, daily-wear, camping, layering, gym, office.",
                    },
                    "occasion": {"type": "array", "items": {"type": "string"}},
                    "material": {"type": ["string", "null"]},
                    "temp_rating_c": {
                        "type": ["integer", "null"],
                        "description": "Lowest comfortable temperature in Celsius. Set "
                        "ONLY for insulating items. Null for everything else.",
                    },
                    "is_giftable": {"type": "boolean"},
                },
                "required": [
                    "id", "brand", "description", "gender", "season",
                    "use_case", "occasion", "material", "temp_rating_c", "is_giftable",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["products"],
    "additionalProperties": False,
}

SYSTEM = """You are a product data specialist for an Indian e-commerce catalogue.

For each product you are given a real retailer listing: title, brand, category and
price. Infer the shopping attributes that the retailer did not publish.

Rules:
- Ground every field in the title, category and price. If the title does not
  support a claim, do not make it. Never invent a specification.
- temp_rating_c is the judgement that matters most for trekking gear. Estimate it
  from the garment type and price: a budget fleece is around 5C, a mid-range
  padded jacket around 0C, a proper down jacket -10C or lower, thermal innerwear
  around 5C, heavy wool socks around 0C. Leave it null for anything that does not
  insulate -- footwear, bags, poles, gifts, electronics.
- Amazon listings have no brand field, so brand must be read out of the title.
  Many titles begin with a descriptive word rather than a brand ("Winter Jacket
  For Men...", "Men Down Jacket"). Return null in those cases.
- Indian seasons: winter gear matters Nov-Feb and at altitude year-round; ethnic
  wear is usually all-season; monsoon matters for waterproofing.
- Return exactly one entry per input product, with the id copied verbatim."""


def build_prompt(batch: list[dict]) -> str:
    lines = [
        f"id: {p['id']}\n"
        f"title: {p['raw_title'][:180]}\n"
        f"brand: {p['brand']} | category: {p['category']} / {p['subcategory']} | "
        f"price: Rs.{p['price_inr']} | retailer: {p['retailer']}"
        for p in batch
    ]
    return "Enrich these products:\n\n" + "\n\n".join(lines)


def merge(products: list[dict], enriched: dict[str, dict]) -> list[dict]:
    merged, missing = [], 0
    for product in products:
        extra = enriched.get(product["id"])
        if not extra:
            missing += 1
            continue
        # The LLM-recovered brand wins when it found one; otherwise keep the
        # heuristic guess so the field is never empty.
        if extra.get("brand"):
            product["brand"] = extra["brand"]
        product["description"] = extra["description"]
        product["attributes"] = {
            "gender": extra["gender"] if extra["gender"] in VALID_GENDERS else "unisex",
            "season": [s for s in extra["season"] if s in VALID_SEASONS] or ["all-season"],
            "use_case": extra["use_case"],
            "occasion": extra["occasion"],
            "material": extra["material"],
            "temp_rating_c": extra["temp_rating_c"],
            "is_giftable": extra["is_giftable"],
        }
        product.pop("raw_title", None)
        for key in ("gender_hint", "season_hint", "material_hint",
                    "occasion_hint", "is_giftable_hint"):
            product.pop(key, None)
        merged.append(product)
    if missing:
        print(f"warning: {missing} products had no enrichment result and were dropped")
    return merged


# --- Anthropic Batch API path (unique to Anthropic, kept as-is) -------------


def submit(client, products: list[dict], model: str) -> str:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = []
    for start in range(0, len(products), BATCH_SIZE):
        batch = products[start : start + BATCH_SIZE]
        requests.append(
            Request(
                custom_id=f"batch-{start // BATCH_SIZE:03d}",
                params=MessageCreateParamsNonStreaming(
                    model=model,
                    max_tokens=8000,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": build_prompt(batch)}],
                    output_config={
                        "format": {"type": "json_schema", "schema": ENRICHMENT_SCHEMA}
                    },
                ),
            )
        )
    created = client.messages.batches.create(requests=requests)
    STATE_FILE.write_text(
        json.dumps({"batch_id": created.id, "model": model, "requests": len(requests)})
    )
    print(f"submitted {len(requests)} batched requests on {model} -> {created.id}")
    return created.id


def poll_interval(elapsed: int) -> int:
    """Poll briskly at first, then back off. Batches rarely finish in seconds."""
    if elapsed < 120:
        return 15
    if elapsed < 900:
        return 30
    return 60


def wait(client, batch_id: str) -> bool:
    """Block until the batch ends. Returns False if it is still running.

    Deliberately does not raise on timeout: an unfinished batch is not an
    error, it is an unfinished batch. Raising would suggest the work was lost
    when in fact it is queued server-side and its results live for 29 days.
    """
    waited = 0
    last_line = ""
    while waited < MAX_WAIT_SECONDS:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts
        if batch.processing_status == "ended":
            print(f"\nbatch ended: succeeded={counts.succeeded} errored={counts.errored}")
            return True
        line = (
            f"  {batch.processing_status}: {counts.processing} queued, "
            f"{counts.succeeded} done ({waited // 60}m elapsed)"
        )
        if line != last_line:
            print(line)
            last_line = line
        interval = poll_interval(waited)
        time.sleep(interval)
        waited += interval
    return False


def collect(client, batch_id: str) -> dict[str, dict]:
    enriched: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            print(f"  {result.custom_id}: {result.result.type}")
            continue
        text = next(
            (b.text for b in result.result.message.content if b.type == "text"), ""
        )
        for item in json.loads(text)["products"]:
            enriched[item["id"]] = item
    return enriched


def resume_id() -> str | None:
    """The batch id from a previous run, if one is still outstanding."""
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text()).get("batch_id")
    except json.JSONDecodeError:
        return None


def run_anthropic_batch(products: list[dict], model: str, resubmit: bool) -> list[dict]:
    import anthropic

    client = anthropic.Anthropic()
    batch_id = None if resubmit else resume_id()
    if batch_id:
        print(f"resuming existing batch {batch_id} (no new charge)")
    else:
        batch_id = submit(client, products, model)

    if not wait(client, batch_id):
        print(
            f"\nBatch {batch_id} is still running. Nothing was lost -- results are "
            f"kept for 29 days.\nRe-run to resume, or use --sync to bypass the "
            f"queue at double the token cost."
        )
        raise SystemExit(2)

    enriched = collect(client, batch_id)
    if not enriched:
        print("Batch ended but returned no usable results; leaving state file for inspection.")
        raise SystemExit(1)
    merged = merge(products, enriched)
    STATE_FILE.unlink(missing_ok=True)
    return merged


# --- Synchronous path, any provider (Anthropic --sync, OpenRouter, local) --


def run_sync_provider(
    provider: LLMProvider, model: str, targets: list[dict], already_merged: list[dict]
) -> list[dict]:
    """Enrich `targets`, writing `OUT_FILE` after every batch.

    A batch's cost is spent the moment the call succeeds, whether or not the
    process is still alive to record the result -- saving only at the end
    meant an interrupted run paid for work it then threw away. `already_merged`
    carries over results from a previous interrupted run so a re-run does not
    redo them.
    """
    merged = list(already_merged)
    total = (len(targets) + BATCH_SIZE - 1) // BATCH_SIZE

    for index, start in enumerate(range(0, len(targets), BATCH_SIZE), start=1):
        batch = targets[start : start + BATCH_SIZE]
        for attempt in range(3):
            try:
                payload = provider.structured(
                    system=SYSTEM,
                    user=build_prompt(batch),
                    schema=ENRICHMENT_SCHEMA,
                    model=model,
                    max_tokens=8000,
                )
                items = payload["products"]
                break
            except (LLMUnavailable, KeyError) as exc:
                print(f"  [{index}/{total}] attempt {attempt + 1} failed: {exc}")
                time.sleep(5)
        else:
            print(f"  [{index}/{total}] giving up on this batch after 3 attempts")
            continue

        enriched = {item["id"]: item for item in items}
        merged.extend(merge(batch, enriched))
        OUT_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"  [{index}/{total}] {len(items)}/{len(batch)} enriched, saved")

    return merged


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--provider", choices=["anthropic", "openrouter", "local"], default="anthropic"
    )
    parser.add_argument(
        "--model", default=None,
        help="Defaults: claude-sonnet-5 (anthropic), required for openrouter/local "
        "(e.g. z-ai/glm-5.2:free, llama3.2:3b)",
    )
    parser.add_argument("--sync", action="store_true",
                         help="Anthropic only: immediate call instead of the Batch API.")
    parser.add_argument("--resubmit", action="store_true",
                         help="Anthropic batch only: force a new batch instead of resuming.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--list-free-models", action="store_true",
                         help="Query OpenRouter's current free, structured-output-capable "
                         "models and exit -- does not touch any catalogue file.")
    args = parser.parse_args()

    if args.list_free_models:
        list_free_models()
        return 0

    if not IN_FILE.exists():
        print(f"Missing {IN_FILE}. Run scripts/normalize.py first.")
        return 1

    model = args.model or ("claude-sonnet-5" if args.provider == "anthropic" else None)
    if model is None:
        print(f"--model is required for --provider {args.provider}. "
              f"Use --list-free-models to see OpenRouter's current $0 options.")
        return 1

    products = json.loads(IN_FILE.read_text())
    if args.limit:
        products = products[: args.limit]

    if args.provider == "anthropic" and not args.sync:
        merged = run_anthropic_batch(products, model, args.resubmit)
        OUT_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
        print(f"\nenriched={len(merged)}/{len(products)} -> {OUT_FILE}")
        return 0

    try:
        provider = build_provider(args.provider, args.timeout)
    except SystemExit as exc:
        print(exc)
        return 1

    already_merged: list[dict] = []
    done_ids: set[str] = set()
    if OUT_FILE.exists():
        already_merged = json.loads(OUT_FILE.read_text())
        done_ids = {p["id"] for p in already_merged}

    targets = [p for p in products if p["id"] not in done_ids]
    print(f"enriching {len(targets)} products ({len(done_ids)} already done, skipped) "
          f"via {args.provider}/{model}")

    merged = run_sync_provider(provider, model, targets, already_merged)
    print(f"\nenriched={len(merged)}/{len(products)} -> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
