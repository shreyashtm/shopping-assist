"""Pull raw listings from Apify into backend/data/raw/.

Reads the finished runs recorded in sources.CAPTURED_DATASETS rather than
re-running the actors, so a rebuild is free and reproduces the committed
catalogue exactly. Requires APIFY_TOKEN in the environment (or backend/.env).

    uv run python scripts/fetch_raw.py
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from apify_client import ApifyClient  # noqa: E402
from sources import CAPTURED_DATASETS  # noqa: E402

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"


def main() -> int:
    token = os.environ.get("APIFY_TOKEN")
    if not token:
        # Fall back to backend/.env so the token never has to be exported.
        env_file = Path(__file__).resolve().parent.parent / ".env"
        if env_file.exists():
            for line in env_file.read_text().splitlines():
                if line.startswith("APIFY_TOKEN="):
                    token = line.split("=", 1)[1].strip()
                    break
    if not token:
        print("APIFY_TOKEN not set. Add it to backend/.env and re-run.")
        return 1

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    client = ApifyClient(token)
    total = 0

    for dataset_id, retailer in CAPTURED_DATASETS.items():
        items = list(client.dataset(dataset_id).iterate_items())
        for item in items:
            item["_retailer"] = retailer
            item["_dataset_id"] = dataset_id
        out = RAW_DIR / f"{retailer.replace('.', '_').lower()}_{dataset_id}.json"
        out.write_text(json.dumps(items, indent=2, ensure_ascii=False))
        print(f"{retailer:<12} {dataset_id}  {len(items):>4} items -> {out.name}")
        total += len(items)

    print(f"\n{total} raw listings across {len(CAPTURED_DATASETS)} datasets.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
