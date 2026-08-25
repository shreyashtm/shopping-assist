"""Check every catalogue link, and record honestly how well it could be checked.

The catalogue points at real retailer pages, so a dead link is a broken promise.
But the three retailers are not equally checkable from a server, and pretending
otherwise would be worse than not checking at all:

* **Myntra** is fully verifiable. A live product page carries the product name in
  its <title>; a dead one returns HTTP 200 with the generic title
  "Product Details". Status code alone would pass both -- the title is the signal.
* **Amazon.in** is bot-walled. A real ASIN and a nonsense one both return HTTP 200
  with a byte-identical ~3.8KB anti-bot interstitial and no <title>. A naive
  status check would mark a dead ASIN "alive", which is exactly the failure this
  script exists to prevent, so Amazon links are reported as unverifiable rather
  than falsely verified.
* **Ajio** returns 403 to HEAD and GET alike from a datacentre IP.

Where HTTP cannot decide, the fallback evidence is provenance: every URL here was
returned by that retailer's own search results during the catalogue build, which
is why the product existed and was purchasable at that moment. That is weaker
than a live check, and the `link_status` field says so rather than hiding it.

    uv run python scripts/verify_links.py
"""

import asyncio
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import httpx

BASE = Path(__file__).resolve().parent.parent
IN_FILE = BASE / "data" / "products_enriched.json"
OUT_FILE = BASE / "data" / "products.json"

# Per-retailer concurrency. Amazon throttles aggressively: at 8-way concurrency
# only ~23% of its links could be confirmed, versus ~40% when paced. The rest is
# IP reputation rather than rate, so pacing further stops paying for itself --
# hence a modest cap rather than a strictly sequential crawl.
RETAILER_CONCURRENCY = {"Amazon.in": 2, "Myntra": 8}
DEFAULT_CONCURRENCY = 4
RETAILER_DELAY_S = {"Amazon.in": 0.6}
TIMEOUT = 25.0
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-IN,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Myntra serves this exact title for a product id that does not exist.
MYNTRA_DEAD_TITLE = "product details"
# Amazon's anti-bot interstitial is small and title-less; real pages are hundreds of KB.
BOT_WALL_MAX_BYTES = 10_000

TITLE_RE = re.compile(r"<title>(.*?)</title>", re.S | re.I)


def page_title(body: str) -> str | None:
    match = TITLE_RE.search(body)
    return match.group(1).strip() if match else None


def judge(retailer: str, status: int, body: str) -> str:
    """Return one of: verified | dead | blocked."""
    if status in (404, 410):
        return "dead"
    if status in (403, 429):
        return "blocked"
    if status >= 400:
        return "dead"

    title = page_title(body)

    if retailer == "Myntra":
        if not title:
            return "blocked"
        return "dead" if title.lower().startswith(MYNTRA_DEAD_TITLE) else "verified"

    # Amazon and anything else: a tiny, title-less body is an anti-bot page, not
    # a product page. We cannot tell live from dead through it, so say so.
    if len(body) < BOT_WALL_MAX_BYTES or not title:
        return "blocked"
    if "page not found" in body[:20000].lower():
        return "dead"
    return "verified"


async def check(client: httpx.AsyncClient, product: dict) -> str:
    url = product["product_url"]
    try:
        response = await client.get(url, follow_redirects=True)
    except httpx.HTTPError:
        return "blocked"
    return judge(product.get("retailer", ""), response.status_code, response.text)


async def main() -> int:
    if not IN_FILE.exists():
        print(f"Missing {IN_FILE}. Run scripts/enrich.py first.")
        return 1

    products = json.loads(IN_FILE.read_text())
    semaphores = {
        retailer: asyncio.Semaphore(RETAILER_CONCURRENCY.get(retailer, DEFAULT_CONCURRENCY))
        for retailer in {p.get("retailer", "") for p in products}
    }
    now = datetime.now(UTC).isoformat()

    async with httpx.AsyncClient(timeout=TIMEOUT, headers=HEADERS) as client:

        async def run(product: dict) -> tuple[dict, str]:
            retailer = product.get("retailer", "")
            async with semaphores[retailer]:
                verdict = await check(client, product)
                delay = RETAILER_DELAY_S.get(retailer)
                if delay:
                    await asyncio.sleep(delay)
                return product, verdict

        results = await asyncio.gather(*(run(p) for p in products))

    kept: list[dict] = []
    dropped: list[dict] = []
    counts: dict[str, int] = {}
    by_retailer: dict[str, dict[str, int]] = {}

    for product, verdict in results:
        counts[verdict] = counts.get(verdict, 0) + 1
        retailer = product.get("retailer", "?")
        by_retailer.setdefault(retailer, {}).setdefault(verdict, 0)
        by_retailer[retailer][verdict] += 1

        if verdict == "dead":
            dropped.append(product)
            continue

        product["link_status"] = verdict
        # Only a genuinely confirmed live page gets a timestamp. A blocked check
        # leaves this null so nothing downstream can claim it was verified.
        product["link_verified_at"] = now if verdict == "verified" else None
        kept.append(product)

    OUT_FILE.write_text(json.dumps(kept, indent=2, ensure_ascii=False))

    print("verdicts:", ", ".join(f"{k}={v}" for k, v in sorted(counts.items())))
    for retailer, tally in sorted(by_retailer.items()):
        print(f"  {retailer:<12} " + ", ".join(f"{k}={v}" for k, v in sorted(tally.items())))
    if dropped:
        print(f"\ndropped {len(dropped)} dead links:")
        for product in dropped[:10]:
            print(f"  {product['id']}  {product['product_url']}")
    print(f"\n{len(kept)} products -> {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
