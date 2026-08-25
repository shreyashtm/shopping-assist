"""Fill in the shopping attributes that retailers do not publish.

Listing pages give a title, a price and a category -- and nothing that lets a
recommender reason about *fit for purpose*. Ajio's `fabric`/`occasion` fields
come back empty in search mode, Amazon has no attributes and no brand field at
all, and none of the retailers provides a description.

So the attributes retrieval depends on -- what season a garment suits, what
activity it is for, how cold a jacket is good to -- are inferred here, once,
offline, and committed with the catalogue. Two consequences follow:

* **The runtime never pays for this.** A search costs one embedding lookup, not
  an LLM call per product.
* **It runs through the Batch API at 50% cost.** Enrichment is not latency
  sensitive -- nobody is waiting on it -- so there is no reason to pay the
  synchronous price. 289 products cost roughly $0.49 instead of $0.98.

    uv run python scripts/enrich.py           # batched, 50% cheaper, queued
    uv run python scripts/enrich.py --sync    # immediate, full price
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, str(Path(__file__).parent))

BASE = Path(__file__).resolve().parent.parent
IN_FILE = BASE / "data" / "products_normalized.json"
OUT_FILE = BASE / "data" / "products_enriched.json"

BATCH_SIZE = 12
MODEL = os.environ.get("ENRICH_MODEL", "claude-sonnet-5")

# Where the in-flight batch id is parked so a re-run resumes instead of
# submitting a second batch. Without this, re-running after a timeout would
# silently pay for the same work twice.
STATE_FILE = BASE / "data" / ".enrich_batch.json"

# The Batch API guarantees completion within 24h, and most finish inside an
# hour -- but "most" is not "all", and queue time is not ours to control. The
# earlier 30-minute ceiling turned normal queueing into a crash.
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


def load_api_key() -> bool:
    if os.environ.get("ANTHROPIC_API_KEY"):
        return True
    env_file = BASE / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            if line.startswith("ANTHROPIC_API_KEY="):
                value = line.split("=", 1)[1].strip()
                if value:
                    os.environ["ANTHROPIC_API_KEY"] = value
                    return True
    return False


def submit(client: anthropic.Anthropic, products: list[dict]) -> str:
    requests = []
    for start in range(0, len(products), BATCH_SIZE):
        batch = products[start : start + BATCH_SIZE]
        requests.append(
            Request(
                custom_id=f"batch-{start // BATCH_SIZE:03d}",
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
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
        json.dumps({"batch_id": created.id, "model": MODEL, "requests": len(requests)})
    )
    print(f"submitted {len(requests)} batched requests on {MODEL} -> {created.id}")
    return created.id


def poll_interval(elapsed: int) -> int:
    """Poll briskly at first, then back off. Batches rarely finish in seconds."""
    if elapsed < 120:
        return 15
    if elapsed < 900:
        return 30
    return 60


def wait(client: anthropic.Anthropic, batch_id: str) -> bool:
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


def collect(client: anthropic.Anthropic, batch_id: str) -> dict[str, dict]:
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


def run_sync(client: anthropic.Anthropic, products: list[dict]) -> dict[str, dict]:
    """Enrich without the Batch API.

    Costs twice as much per token but returns in about two minutes instead of
    waiting in a queue. Worth it whenever enrichment is on the critical path --
    which, since the embedding index and everything downstream depend on it,
    is most of the time during development.
    """
    enriched: dict[str, dict] = {}
    total = (len(products) + BATCH_SIZE - 1) // BATCH_SIZE

    for index, start in enumerate(range(0, len(products), BATCH_SIZE), start=1):
        batch = products[start : start + BATCH_SIZE]
        response = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(batch)}],
            output_config={"format": {"type": "json_schema", "schema": ENRICHMENT_SCHEMA}},
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        items = json.loads(text)["products"]
        for item in items:
            enriched[item["id"]] = item
        print(f"  [{index}/{total}] {len(items)}/{len(batch)} enriched")

    return enriched


def resume_id() -> str | None:
    """The batch id from a previous run, if one is still outstanding."""
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text()).get("batch_id")
    except json.JSONDecodeError:
        return None


def main() -> int:
    if not IN_FILE.exists():
        print(f"Missing {IN_FILE}. Run scripts/normalize.py first.")
        return 1
    if not load_api_key():
        print("ANTHROPIC_API_KEY not set. Add it to backend/.env and re-run.")
        return 1

    products = json.loads(IN_FILE.read_text())
    client = anthropic.Anthropic()

    if "--sync" in sys.argv:
        print(f"enriching {len(products)} products synchronously on {MODEL}")
        enriched = run_sync(client, products)
    else:
        force_new = "--resubmit" in sys.argv
        batch_id = None if force_new else resume_id()
        if batch_id:
            print(f"resuming existing batch {batch_id} (no new charge)")
        else:
            batch_id = submit(client, products)

        if not wait(client, batch_id):
            print(
                f"\nBatch {batch_id} is still running. Nothing was lost -- results are "
                f"kept for 29 days.\nRe-run to resume, or use --sync to bypass the "
                f"queue at double the token cost:"
                f"\n    uv run python scripts/enrich.py --sync"
            )
            return 2

        enriched = collect(client, batch_id)
    if not enriched:
        print("Batch ended but returned no usable results; leaving state file for inspection.")
        return 1

    merged = merge(products, enriched)
    OUT_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False))
    STATE_FILE.unlink(missing_ok=True)
    print(f"\nenriched={len(merged)}/{len(products)} -> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
