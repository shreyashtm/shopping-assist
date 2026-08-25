"""Map the catalogue onto the canonical two-axis taxonomy.

Existing products carry whatever category the scrape query happened to assign,
which mixed product type with use case: a trekking boot was filed under
"Outdoor & Trekking / Footwear" and so was invisible to any query about
footwear. This pass re-files every product by *what it is*, and moves the
use-case information into attributes where it belongs.

Runs once, offline, over data we already have (title, description, existing
attributes). No re-scraping, and product ids, prices and URLs are untouched.

    uv run python scripts/reclassify.py            # batched, cheaper
    uv run python scripts/reclassify.py --sync     # immediate
"""

import json
import os
import sys
from pathlib import Path

import anthropic

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.taxonomy import ALL_PATHS, OCCASIONS, USE_CASES  # noqa: E402

BASE = Path(__file__).resolve().parent.parent
IN_FILE = BASE / "data" / "products.json"
OUT_FILE = BASE / "data" / "products.json"
BACKUP = BASE / "data" / "products.pre_reclassify.json"

BATCH_SIZE = 15
MODEL = os.environ.get("RECLASSIFY_MODEL", "claude-sonnet-5")

SCHEMA = {
    "type": "object",
    "properties": {
        "products": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "path": {
                        "type": "string",
                        "description": "Canonical 'Category/Subcategory' path, "
                        "chosen from the allowed list. What the product IS.",
                    },
                    "use_case": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "What it is FOR. From the allowed use-case list.",
                    },
                    "occasion": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "path", "use_case", "occasion"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["products"],
    "additionalProperties": False,
}

SYSTEM = f"""You re-file products onto a canonical retail taxonomy with two axes.

AXIS 1 - `path`: what the product IS. Choose exactly one from this closed list:
{chr(10).join('  ' + p for p in ALL_PATHS)}

AXIS 2 - `use_case` / `occasion`: what the product is FOR.
  use_case must come from: {', '.join(USE_CASES)}
  occasion must come from: {', '.join(OCCASIONS)}

The distinction is the whole point of this exercise, so apply it strictly:

- A trekking boot is `Footwear/Boots` with use_case ["trekking","hiking"].
  It is NOT an "outdoor" category product -- outdoor is what it is for.
- A down jacket for a trek is `Men's Apparel/Jackets & Coats` with
  use_case ["trekking","layering"]. Apparel is apparel wherever it is worn.
- Thermal innerwear is `Men's Apparel/Thermals & Base Layers`.
- A 50L rucksack is `Bags & Luggage/Backpacks` with use_case ["trekking"].
- A sleeping bag or headlamp genuinely IS camping equipment:
  `Outdoor & Camping Gear/Camp & Sleep` or `.../Navigation & Safety`.
- Wool socks are `Men's Apparel/Thermals & Base Layers` if thermal innerwear,
  otherwise `Outdoor & Camping Gear/Outdoor Accessories`.
- A gift hamper IS a hamper: `Gifting/Hampers`. But a perfume set that happens
  to be sold as a gift is `Beauty & Personal Care/Fragrance` with
  use_case ["gifting"] -- gifting is a use, not a product type.
- Mojaris and juttis are `Footwear/Ethnic Footwear`.
- A potli or clutch is `Bags & Luggage/Handbags & Clutches` with
  use_case ["festive","wedding"].

Pick the single most accurate path. Never invent a path outside the list.
Return exactly one entry per input product, id copied verbatim."""


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


def build_prompt(batch: list[dict]) -> str:
    lines = []
    for p in batch:
        attrs = p.get("attributes", {})
        lines.append(
            f"id: {p['id']}\n"
            f"title: {p['title']}\n"
            f"currently filed as: {p['category']} / {p['subcategory']}\n"
            f"description: {p.get('description', '')[:160]}\n"
            f"existing use_case: {attrs.get('use_case', [])}"
        )
    return "Re-file these products:\n\n" + "\n\n".join(lines)


def main() -> int:
    if not IN_FILE.exists():
        print(f"Missing {IN_FILE}.")
        return 1
    if not load_api_key():
        print("ANTHROPIC_API_KEY not set. Add it to backend/.env.")
        return 1

    products = json.loads(IN_FILE.read_text())
    if not BACKUP.exists():
        BACKUP.write_text(json.dumps(products, indent=2, ensure_ascii=False))
        print(f"backed up original -> {BACKUP.name}")

    client = anthropic.Anthropic(timeout=120.0)
    mapping: dict[str, dict] = {}
    total = (len(products) + BATCH_SIZE - 1) // BATCH_SIZE

    for index, start in enumerate(range(0, len(products), BATCH_SIZE), start=1):
        batch = products[start : start + BATCH_SIZE]
        response = client.messages.create(
            model=MODEL,
            max_tokens=8000,
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(batch)}],
            output_config={"format": {"type": "json_schema", "schema": SCHEMA}},
        )
        text = next((b.text for b in response.content if b.type == "text"), "")
        for item in json.loads(text)["products"]:
            mapping[item["id"]] = item
        print(f"  [{index}/{total}] {len(batch)} products re-filed")

    valid_paths = set(ALL_PATHS)
    rejected = moved = 0
    for product in products:
        item = mapping.get(product["id"])
        if not item or item["path"] not in valid_paths:
            rejected += 1
            continue
        category, sub = item["path"].split("/", 1)
        if (category, sub) != (product["category"], product["subcategory"]):
            moved += 1
        product["category"], product["subcategory"] = category, sub
        attrs = product.setdefault("attributes", {})
        attrs["use_case"] = [u for u in item["use_case"] if u in USE_CASES]
        attrs["occasion"] = [o for o in item["occasion"] if o in OCCASIONS]

    OUT_FILE.write_text(json.dumps(products, indent=2, ensure_ascii=False))
    print(f"\nre-filed={len(products) - rejected}  moved={moved}  "
          f"kept_original={rejected}  -> {OUT_FILE.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
