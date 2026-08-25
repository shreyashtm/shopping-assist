"""Fill in the four suitability attributes scope.md's generalized mechanism
needs: water_resistance, layer, formality, breathability.

Unlike scripts/enrich.py, this does not build a fresh catalogue -- it patches
four new fields onto the already-committed `data/products.json`, for the
categories where they are meaningful (apparel, footwear, outdoor gear).
Everything already enriched (gender, season, use_case, occasion, material,
temp_rating_c) is left untouched and is itself useful grounding: a product
already tagged use_case=trekking and material=down is strong evidence for
layer=outer, without the model re-deriving it from the title alone.

Runs synchronously (not through the Batch API) so it finishes within one
session instead of queueing for up to 24h -- worth the 2x token cost here
because the result needs to be checked and wired up in the same sitting.

    uv run python scripts/enrich_suitability.py           # all target products
    uv run python scripts/enrich_suitability.py --limit 24  # smoke test
"""

import json
import os
import sys
import time
from pathlib import Path

import anthropic

BASE = Path(__file__).resolve().parent.parent
PRODUCTS_FILE = BASE / "data" / "products.json"
BACKUP_FILE = BASE / "data" / "products.pre_suitability.json"

# Suitability axes are only meaningful for wearable/gear categories -- a
# water_resistance judgement on a Bluetooth speaker or a saucepan is noise,
# not signal.
TARGET_CATEGORIES = {
    "Men's Apparel", "Women's Apparel", "Ethnic Wear", "Footwear",
    "Outdoor & Camping Gear",
}

BATCH_SIZE = 12
MODEL = os.environ.get("ENRICH_MODEL", "claude-sonnet-5")

WATER_RESISTANCE_VALUES = ["none", "repellent", "waterproof"]
LAYER_VALUES = ["base", "mid", "outer", "standalone"]
FORMALITY_VALUES = ["casual", "smart_casual", "formal"]
BREATHABILITY_VALUES = ["low", "medium", "high"]

SUITABILITY_SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    # No "type" key on these four: Anthropic's structured-output
                    # validator rejects `enum` combined with a nullable union
                    # type (`type: ["string", "null"]`), full stop, regardless
                    # of what the enum values are -- confirmed by isolating it
                    # against a trivial schema. The supported nullable-enum
                    # shape is `enum` alone with `null` as one of its members.
                    "water_resistance": {
                        "enum": [*WATER_RESISTANCE_VALUES, None],
                        "description": "'none' is a real judgement (this garment "
                        "offers no rain protection), not the same as null. Use "
                        "null only when the category makes the question "
                        "meaningless (jewellery, bags with no fabric shell).",
                    },
                    "layer": {
                        "enum": [*LAYER_VALUES, None],
                        "description": "Where this sits in a layering system. "
                        "'standalone' for anything not meant to be layered "
                        "(a t-shirt, a pair of jeans). Null for non-apparel "
                        "(footwear, accessories).",
                    },
                    "formality": {"enum": [*FORMALITY_VALUES, None]},
                    "breathability": {
                        "enum": [*BREATHABILITY_VALUES, None],
                        "description": "Comfort signal only. Null when the "
                        "material/construction gives no basis for a judgement.",
                    },
                },
                "required": ["id", "water_resistance", "layer", "formality", "breathability"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["products"],
    "additionalProperties": False,
}

SYSTEM = """You are a product data specialist for an Indian e-commerce catalogue.

For each product, judge four suitability attributes from its title, category,
and existing attributes (already-inferred use case, occasion, season, material,
temperature rating). Those existing fields are strong evidence -- a product
already tagged use_case "trekking" and material "down" is very likely an outer
layer with real rain resistance; a cotton daily-wear t-shirt is standalone,
casual, and offers no rain protection.

water_resistance: 'waterproof' only for garments explicitly built for sustained
wet conditions (rain shells, waterproof boots). 'repellent' for water-resistant
but not fully waterproof (most puffer jackets, treated softshells). 'none' is a
real, useful judgement -- most everyday apparel has none, and saying so is more
useful than leaving it null. Use null only when the question does not apply to
the product type at all (jewellery, most bags, accessories).

layer: base (worn against skin -- thermals, base-layer tees), mid (insulating
layer -- fleece, sweaters, light jackets), outer (the outermost shell -- heavy
jackets, rain shells, coats), or standalone (not part of a layering system --
most t-shirts, jeans, dresses, suits). Null for footwear and non-apparel.

formality: casual (everyday wear, gym kit, trekking gear), smart_casual
(collared shirts, chinos, non-suit blazers), or formal (suits, sherwanis,
formal shoes, wedding wear). Judge from garment type and the category, not price.

breathability: low/medium/high based on material and construction. Null when
there is no real basis to judge it.

Ground every field in the title, category and existing attributes. Never
invent a specification the data does not support -- null is the honest answer
when genuinely unsure, and 'none' is the honest answer when a garment plainly
lacks the property rather than merely being unevaluated for it."""


def build_prompt(batch: list[dict]) -> str:
    lines = []
    for p in batch:
        attrs = p.get("attributes", {})
        lines.append(
            f"id: {p['id']}\n"
            f"title: {p['title'][:180]}\n"
            f"category: {p['category']} / {p['subcategory']}\n"
            f"existing: use_case={attrs.get('use_case')} occasion={attrs.get('occasion')} "
            f"season={attrs.get('season')} material={attrs.get('material')} "
            f"temp_rating_c={attrs.get('temp_rating_c')}"
        )
    return "Judge suitability attributes for these products:\n\n" + "\n\n".join(lines)


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


def _already_enriched(product: dict) -> bool:
    """True once a product has been through this script before.

    Checked against `water_resistance` specifically, which every target
    category gets a real judgement for (never null-by-design, unlike `layer`
    for footwear) -- so its presence reliably means "done", not "the model
    happened to say null this field".
    """
    return product.get("attributes", {}).get("water_resistance") is not None


def _apply(product: dict, extra: dict) -> None:
    attrs = product.setdefault("attributes", {})
    for key, valid in (
        ("water_resistance", WATER_RESISTANCE_VALUES),
        ("layer", LAYER_VALUES),
        ("formality", FORMALITY_VALUES),
        ("breathability", BREATHABILITY_VALUES),
    ):
        value = extra.get(key)
        attrs[key] = value if value in valid else None


def run_sync(
    client: anthropic.Anthropic, all_products: list[dict], targets: list[dict]
) -> int:
    """Enrich `targets` in place on `all_products`, saving after every batch.

    A batch's cost is spent the moment the API call succeeds, whether or not
    the process is still alive to record the result -- saving only at the
    end (the previous shape of this function) meant an interrupted run paid
    for work it then threw away. Saving per batch bounds that loss to at most
    one in-flight batch.
    """
    by_id = {p["id"]: p for p in all_products}
    total = (len(targets) + BATCH_SIZE - 1) // BATCH_SIZE
    updated = 0

    for index, start in enumerate(range(0, len(targets), BATCH_SIZE), start=1):
        batch = targets[start : start + BATCH_SIZE]
        for attempt in range(3):
            try:
                response = client.messages.create(
                    model=MODEL,
                    max_tokens=4000,
                    system=SYSTEM,
                    messages=[{"role": "user", "content": build_prompt(batch)}],
                    output_config={
                        "format": {"type": "json_schema", "schema": SUITABILITY_SCHEMA}
                    },
                )
                text = next((b.text for b in response.content if b.type == "text"), "")
                items = json.loads(text)["products"]
                break
            except (anthropic.APIError, json.JSONDecodeError, KeyError) as exc:
                print(f"  [{index}/{total}] attempt {attempt + 1} failed: {exc}")
                time.sleep(5)
        else:
            print(f"  [{index}/{total}] giving up on this batch after 3 attempts")
            continue

        for item in items:
            product = by_id.get(item["id"])
            if product is not None:
                _apply(product, item)
                updated += 1
        PRODUCTS_FILE.write_text(json.dumps(all_products, indent=2, ensure_ascii=False))
        print(f"  [{index}/{total}] {len(items)}/{len(batch)} enriched, saved")

    return updated


def main() -> int:
    if not PRODUCTS_FILE.exists():
        print(f"Missing {PRODUCTS_FILE}.")
        return 1
    if not load_api_key():
        print("ANTHROPIC_API_KEY not set. Add it to backend/.env and re-run.")
        return 1

    products = json.loads(PRODUCTS_FILE.read_text())
    in_scope = [p for p in products if p["category"] in TARGET_CATEGORIES]
    targets = [p for p in in_scope if not _already_enriched(p)]
    already = len(in_scope) - len(targets)

    if "--limit" in sys.argv:
        limit = int(sys.argv[sys.argv.index("--limit") + 1])
        targets = targets[:limit]

    print(f"enriching {len(targets)} products ({already} already done, skipped) "
          f"on {MODEL}")

    if not BACKUP_FILE.exists():
        BACKUP_FILE.write_text(PRODUCTS_FILE.read_text())
        print(f"backed up current catalogue -> {BACKUP_FILE}")

    client = anthropic.Anthropic()
    updated = run_sync(client, products, targets)

    print(f"\nupdated {updated}/{len(targets)} products this run -> {PRODUCTS_FILE}")
    if updated < len(targets):
        print(f"warning: {len(targets) - updated} had no result (transient "
              "failures); re-run to fill the rest -- already-done products "
              "are skipped automatically.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
