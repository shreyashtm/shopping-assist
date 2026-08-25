"""Build the product catalogue end to end.

    uv run python scripts/build_catalogue.py

Runs, in order: fetch raw listings -> normalize and clean -> enrich with Claude
-> verify links. Each stage writes its own file under backend/data/, so a stage
can be re-run on its own during development without repeating the ones before it.

Requires APIFY_TOKEN and ANTHROPIC_API_KEY in backend/.env. The finished
catalogue (data/products.json) is committed, so running the app does not need
either key.
"""

import subprocess
import sys
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parent
STAGES = [
    ("Fetching raw listings from Apify", "fetch_raw.py"),
    ("Normalizing and cleaning", "normalize.py"),
    ("Enriching attributes with Claude", "enrich.py"),
    ("Verifying product links", "verify_links.py"),
]


def main() -> int:
    for index, (label, script) in enumerate(STAGES, start=1):
        print(f"\n[{index}/{len(STAGES)}] {label}")
        print("-" * 60)
        result = subprocess.run([sys.executable, str(SCRIPTS / script)], check=False)
        if result.returncode != 0:
            print(f"\nStage failed: {script}")
            return result.returncode
    print("\nCatalogue built -> backend/data/products.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
