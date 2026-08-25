"""Fold the enriched Kaggle sample into the built catalogue.

Runs after enrich_kaggle.py and before build_index.py / build_taxonomy.py:
this stage decides ids and link status, the embedding matrix and the taxonomy
map both have to be rebuilt afterward to stay in sync with the new products.

    uv run python scripts/merge_kaggle.py
    uv run python scripts/build_index.py
    uv run python scripts/build_taxonomy.py
"""

import json
from datetime import UTC, datetime
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
PRODUCTS_FILE = BASE / "data" / "products.json"
BACKUP_FILE = BASE / "data" / "products.pre_kaggle.json"
KAGGLE_FILE = BASE / "data" / "raw" / "kaggle_fashion_enriched.json"
SAMPLE_FILE = BASE / "data" / "raw" / "kaggle_fashion_sample.json"
REMAINING_FILE = BASE / "data" / "raw" / "kaggle_fashion_remaining.json"

# A batch response can be syntactically valid JSON while an individual item's
# fields inside it are truncated garbage -- observed as description="placeholder"
# and description="," on ~4% of one run, all traceable to the same token-budget
# pressure that corrupted whole batches elsewhere (see enrich_kaggle.py). Syntax
# validity does not imply content validity, so every record is re-checked here
# rather than trusting anything that merely parsed.
MIN_DESCRIPTION_CHARS = 20


def is_corrupt(product: dict) -> bool:
    description = (product.get("description") or "").strip()
    if len(description) < MIN_DESCRIPTION_CHARS:
        return True
    if not any(c.isalpha() for c in description):
        return True
    if not product.get("category") or not product.get("subcategory"):
        return True
    attributes = product.get("attributes") or {}
    return not attributes.get("season") or not attributes.get("gender")


def main() -> int:
    if not KAGGLE_FILE.exists():
        print(f"Missing {KAGGLE_FILE}. Run scripts/enrich_kaggle.py first.")
        return 1

    existing = json.loads(PRODUCTS_FILE.read_text())
    existing_ids = {p["id"] for p in existing}
    kaggle = json.loads(KAGGLE_FILE.read_text())

    added, skipped, corrupt = [], 0, []
    for product in kaggle:
        if product["id"] in existing_ids:
            skipped += 1
            continue
        if is_corrupt(product):
            corrupt.append(product["id"])
            continue
        product["in_stock"] = True
        # Never live-verifiable by construction -- see LinkStatus in
        # app/schemas/product.py. Setting this here rather than running
        # verify_links.py against it: that script checks whether a *live*
        # page resolves, which is not the question for a historical record
        # whose honest status is "never checked", not "checked and dead".
        product["link_status"] = "archival"
        product["link_verified_at"] = None
        added.append(product)

    if corrupt and SAMPLE_FILE.exists():
        # Route corrupted ids back to the retry queue alongside anything the
        # batch never returned at all, so one re-run of enrich_kaggle.py picks
        # up both categories of loss.
        sample_by_id = {p["id"]: p for p in json.loads(SAMPLE_FILE.read_text())}
        already_queued = (
            {p["id"] for p in json.loads(REMAINING_FILE.read_text())}
            if REMAINING_FILE.exists()
            else set()
        )
        retry = [sample_by_id[i] for i in corrupt if i in sample_by_id and i not in already_queued]
        if retry:
            existing_retry = (
                json.loads(REMAINING_FILE.read_text()) if REMAINING_FILE.exists() else []
            )
            REMAINING_FILE.write_text(
                json.dumps(existing_retry + retry, indent=2, ensure_ascii=False)
            )
        print(
            f"{len(corrupt)} corrupted records excluded and queued "
            f"for retry in {REMAINING_FILE.name}"
        )

    if not added:
        print("Nothing new to merge (all ids already present).")
        return 0

    BACKUP_FILE.write_text(json.dumps(existing, indent=2, ensure_ascii=False))
    merged = existing + added
    PRODUCTS_FILE.write_text(json.dumps(merged, indent=2, ensure_ascii=False))

    print(
        f"merged {len(added)} archival products "
        f"({skipped} already present, {len(corrupt)} corrupted+excluded)"
    )
    print(f"catalogue: {len(existing)} -> {len(merged)}")
    print(f"backup of the pre-merge catalogue -> {BACKUP_FILE}")
    print(f"as_of note: archival records carry link_status='archival', "
          f"sourced {datetime.now(UTC).date().isoformat()}")
    print("\nNext: uv run python scripts/build_index.py && "
          "uv run python scripts/build_taxonomy.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
