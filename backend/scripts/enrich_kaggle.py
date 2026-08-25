"""Classify and enrich the Kaggle sample in one LLM pass.

For the Apify-sourced catalogue, classification and attribute enrichment are
two separate scripts (reclassify.py, enrich.py), run at different points in the
catalogue's history for reasons specific to that history -- the taxonomy was
redesigned after the first enrichment pass had already run.

The Kaggle sample has no such history: it has never been classified at all.
sample_kaggle.py's `candidate_path` is a keyword guess used only to decide
which items got sampled for which gap (see that script's docstring), and it is
visibly wrong often enough to matter -- "Merino Wool Dress Socks for Men" was
sampled toward Dresses on the strength of the word "dress". So classification
and enrichment are combined into one call here: the model sees the full title
and is told explicitly that the candidate path is a hint, not a fact, and must
choose the correct path from the whole taxonomy regardless of which gap the
item happened to be sampled for.

    uv run python scripts/enrich_kaggle.py           # batched, cheaper, queued
    uv run python scripts/enrich_kaggle.py --sync    # immediate, full price
"""

import json
import sys
import time
from pathlib import Path

import anthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from enrich import (  # noqa: E402
    MAX_WAIT_SECONDS,
    MODEL,
    VALID_GENDERS,
    VALID_SEASONS,
    load_api_key,
    poll_interval,
)

from app.services.taxonomy import ALL_PATHS, OCCASIONS, USE_CASES  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
IN_FILE = BASE / "data" / "raw" / "kaggle_fashion_sample.json"
OUT_FILE = BASE / "data" / "raw" / "kaggle_fashion_enriched.json"
STATE_FILE = BASE / "data" / ".enrich_kaggle_batch.json"

# Smaller batch and more headroom than enrich.py's, and for a specific reason:
# this schema adds a 66-value category_path enum on top of the original
# attributes, and that heavier schema visibly pushed the model into spending
# thousands of tokens on internal reasoning before writing any JSON -- observed
# up to 4,579 of an 8,000 budget on a single request, truncating the output
# mid-string. A smaller batch means less JSON to produce per request, and more
# headroom means reasoning no longer competes with the response for the same
# budget.
BATCH_SIZE = 6
MAX_TOKENS = 16000

SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "category_path": {
                        "type": "string",
                        "enum": ALL_PATHS,
                        "description": "The product's REAL Category/Subcategory, "
                        "chosen from the full taxonomy. The candidate path supplied "
                        "with each product is only a keyword guess and is frequently "
                        "wrong -- verify it against the title yourself and correct it "
                        "when it does not fit, including moving to a path that "
                        "already holds other products.",
                    },
                    "brand": {
                        "type": ["string", "null"],
                        "description": "Real brand name if the title contains one. "
                        "Null when the listing has no brand -- never promote a "
                        "descriptive word.",
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
                        "description": "What conditions this suits. Rain, "
                        "waterproof, quick-dry or breathable wording means "
                        "'monsoon' belongs here alongside whatever else applies -- "
                        "this dataset is being added specifically to give the "
                        "catalogue real monsoon-appropriate options, so do not "
                        "default to all-season when the title gives a real signal.",
                    },
                    "use_case": {
                        "type": "array",
                        "items": {"type": "string", "enum": USE_CASES},
                    },
                    "occasion": {
                        "type": "array",
                        "items": {"type": "string", "enum": OCCASIONS},
                    },
                    "material": {"type": ["string", "null"]},
                    "temp_rating_c": {
                        "type": ["integer", "null"],
                        "description": "Lowest comfortable temperature in Celsius. "
                        "Set ONLY for insulating items -- jackets, coats, thermals, "
                        "sweaters. Null for everything else, including anything "
                        "rain-focused rather than warmth-focused: a rain shell is "
                        "not insulation.",
                    },
                    "is_giftable": {"type": "boolean"},
                },
                "required": [
                    "id", "category_path", "brand", "description", "gender",
                    "season", "use_case", "occasion", "material", "temp_rating_c",
                    "is_giftable",
                ],
                "additionalProperties": False,
            },
        }
    },
    "required": ["products"],
    "additionalProperties": False,
}

SYSTEM = """You are a product data specialist building an Indian shopping
catalogue from a bulk US Amazon dataset.

For each product you are given a real (if dated) retailer listing: title,
price, and a *candidate* category path that a keyword search guessed. Two jobs,
in one pass:

1. Confirm or correct the category path. The candidate is often wrong --
   "Merino Wool Dress Socks" was guessed toward Dresses on the word "dress"
   alone. Read the actual title and choose the correct Category/Subcategory
   from the full taxonomy you are given, even if that means moving the product
   to a path that is not the one it was sampled for, or to a path that already
   holds other products.

2. Infer the shopping attributes the listing does not publish, exactly as you
   would for any other product: ground every field in the title, never invent
   a specification, and set temp_rating_c only for genuinely insulating items.

This dataset is specifically being added to fix a real defect: the existing
catalogue recommended a winter jacket for a monsoon request, because nothing in
it was tagged as suited to rain or heat. When a title says rain, waterproof,
water-resistant, quick-dry, or breathable, that is real signal for `season`
and must not be dropped in favour of a lazy 'all-season' default -- but it must
also not be invented where the title gives no such evidence.

Return exactly one entry per input product, with the id copied verbatim."""


def build_prompt(batch: list[dict]) -> str:
    lines = [
        f"id: {p['id']}\n"
        f"title: {p['raw_title'][:180]}\n"
        f"candidate_path: {p['candidate_path']} (unverified guess)\n"
        f"price: Rs.{p['price_inr']}"
        for p in batch
    ]
    return "Classify and enrich these products:\n\n" + "\n\n".join(lines)


def submit(client: anthropic.Anthropic, products: list[dict]) -> str:
    requests = []
    for start in range(0, len(products), BATCH_SIZE):
        batch = products[start : start + BATCH_SIZE]
        requests.append(
            Request(
                custom_id=f"kaggle-{start // BATCH_SIZE:03d}",
                params=MessageCreateParamsNonStreaming(
                    model=MODEL,
                    max_tokens=MAX_TOKENS,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": build_prompt(batch)}],
                    output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
                ),
            )
        )
    created = client.messages.batches.create(requests=requests)
    STATE_FILE.write_text(
        json.dumps({"batch_id": created.id, "model": MODEL, "requests": len(requests)})
    )
    print(f"submitted {len(requests)} batched requests on {MODEL} -> {created.id}")
    return created.id


def wait(client: anthropic.Anthropic, batch_id: str) -> bool:
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
    """Pull every succeeded result.

    One malformed response must not cost the other batches. A too-small
    token budget for a batch that draws heavier reasoning can still come back
    truncated -- the API reports that request "succeeded" (it did produce
    content), the failure is only visible when this side tries to parse it. A
    truncated batch is skipped and named so it can be resubmitted alone,
    rather than losing everything already collected.
    """
    enriched: dict[str, dict] = {}
    for result in client.messages.batches.results(batch_id):
        if result.result.type != "succeeded":
            print(f"  {result.custom_id}: {result.result.type}")
            continue
        text = next(
            (b.text for b in result.result.message.content if b.type == "text"), ""
        )
        try:
            items = json.loads(text)["products"]
        except (json.JSONDecodeError, KeyError) as exc:
            print(f"  {result.custom_id}: unparseable response ({exc}) -- skipped")
            continue
        for item in items:
            enriched[item["id"]] = item
    return enriched


def run_sync(client: anthropic.Anthropic, products: list[dict]) -> dict[str, dict]:
    enriched: dict[str, dict] = {}
    total = (len(products) + BATCH_SIZE - 1) // BATCH_SIZE
    for index, start in enumerate(range(0, len(products), BATCH_SIZE), start=1):
        batch = products[start : start + BATCH_SIZE]
        response = client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(batch)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        items = json.loads(text)["products"]
        for item in items:
            enriched[item["id"]] = item
        print(f"  [{index}/{total}] {len(items)}/{len(batch)} enriched")
    return enriched


def merge(products: list[dict], enriched: dict[str, dict]) -> list[dict]:
    merged, missing = [], 0
    for product in products:
        extra = enriched.get(product["id"])
        if not extra:
            missing += 1
            continue
        path = extra["category_path"]
        category, _, subcategory = path.partition("/")
        product["category"] = category
        product["subcategory"] = subcategory
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
        for key in ("raw_title", "candidate_path", "monsoon_candidate", "source"):
            product.pop(key, None)
        merged.append(product)
    if missing:
        print(f"warning: {missing} products had no enrichment result and were dropped")
    return merged


def resume_id() -> str | None:
    if not STATE_FILE.exists():
        return None
    try:
        return json.loads(STATE_FILE.read_text()).get("batch_id")
    except json.JSONDecodeError:
        return None


def main() -> int:
    if not IN_FILE.exists():
        print(f"Missing {IN_FILE}. Run scripts/sample_kaggle.py first.")
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
                f"\nBatch {batch_id} is still running. Nothing was lost -- results "
                f"are kept for 29 days.\nRe-run to resume, or use --sync to bypass "
                f"the queue at double the token cost."
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
