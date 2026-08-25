"""Generate the catalogue's availability map.

Written from the catalogue rather than maintained by hand, so it cannot drift
from what is actually in stock -- the drift that previously let the planner ask
for a `Formal Trousers` slot that mapped to nothing.

Two layers come out of this:
  * actual   - what exists, with counts and price ranges
  * coverage - target minus actual, i.e. the sourcing checklist

    uv run python scripts/build_taxonomy.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.taxonomy import (  # noqa: E402
    MIN_VIABLE_PER_SUBCATEGORY,
    PRODUCT_TAXONOMY,
)

BASE = Path(__file__).resolve().parent.parent
PRODUCTS = BASE / "data" / "products.json"
OUT = BASE / "data" / "taxonomy.json"


def main() -> int:
    if not PRODUCTS.exists():
        print(f"Missing {PRODUCTS}.")
        return 1

    products = json.loads(PRODUCTS.read_text())
    buckets: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for p in products:
        buckets[p["category"]][p["subcategory"]].append(p)

    actual: dict[str, dict[str, dict]] = {}
    for category, subs in PRODUCT_TAXONOMY.items():
        actual[category] = {}
        for sub in subs:
            items = buckets.get(category, {}).get(sub, [])
            entry: dict[str, object] = {"count": len(items)}
            if items:
                prices = sorted(p["price_inr"] for p in items)
                entry["price_range"] = [prices[0], prices[-1]]
                entry["viable"] = len(items) >= MIN_VIABLE_PER_SUBCATEGORY
            else:
                entry["viable"] = False
            actual[category][sub] = entry

    empty = [
        f"{c}/{s}" for c, subs in actual.items()
        for s, e in subs.items() if e["count"] == 0
    ]
    thin = [
        f"{c}/{s} ({e['count']})" for c, subs in actual.items()
        for s, e in subs.items() if 0 < e["count"] < MIN_VIABLE_PER_SUBCATEGORY
    ]

    OUT.write_text(json.dumps(
        {
            "product_count": len(products),
            "min_viable": MIN_VIABLE_PER_SUBCATEGORY,
            "categories": actual,
            "coverage": {"empty": empty, "thin": thin},
        },
        indent=2,
    ))

    total_paths = sum(len(v) for v in PRODUCT_TAXONOMY.values())
    print(f"{len(products)} products across {total_paths} canonical paths")
    print(f"  viable   : {total_paths - len(empty) - len(thin)}")
    print(f"  thin (<{MIN_VIABLE_PER_SUBCATEGORY}): {len(thin)}")
    print(f"  empty    : {len(empty)}")
    if thin:
        print("\nTHIN:")
        for t in thin:
            print(f"  {t}")
    print("\nEMPTY (sourcing checklist):")
    for e in empty:
        print(f"  {e}")
    print(f"\n-> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
