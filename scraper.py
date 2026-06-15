import argparse
import random
import requests
import json
import re
import csv
import time

FIELDS = ["brand", "model", "fuel", "build_year", "mileage_km",
          "transmission", "body_type", "title", "price_cents", "url"]

# Be polite: wait a random interval (seconds) between page requests so we don't
# hammer the site. Adjust these bounds if needed.
PAGE_DELAY_MIN = 10.0
PAGE_DELAY_MAX = 15.0


def polite_sleep(lo=PAGE_DELAY_MIN, hi=PAGE_DELAY_MAX):
    """Sleep a random number of seconds in [lo, hi] to respect the site."""
    secs = random.uniform(lo, hi)
    print(f"  ... sleeping {secs:.1f}s to respect the site")
    time.sleep(secs)

BASE = "https://www.marktplaats.nl"
HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/124.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "nl-NL,nl;q=0.9,en;q=0.8",
}

NEXT_DATA_RE = re.compile(
    r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL
)


def get_next_data(url, session):
    """Fetch a page and return the parsed __NEXT_DATA__ JSON."""
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    m = NEXT_DATA_RE.search(resp.text)
    if not m:
        return None
    return json.loads(m.group(1))


def get_brands(session):
    """Return list of brand dicts: {id, key, name} from the all-cars page."""
    data = get_next_data(f"{BASE}/l/auto-s/", session)
    options = data["props"]["pageProps"]["searchRequestAndResponse"]["searchCategoryOptions"]
    brands = []
    for o in options:
        # Skip the top-level "Auto's" category and non-brand buckets
        if o.get("key") in ("auto-s", "bestelauto-s", "vrachtwagens",
                             "oldtimers", "overige-auto-s"):
            continue
        brands.append({"id": o["id"], "key": o["key"], "name": o["name"]})
    return brands


def attrs_to_dict(attribute_list):
    """Flatten the attributes array into a key->value dict."""
    return {a["key"]: a.get("value", "") for a in (attribute_list or [])}


def parse_listing(listing, brand_name):
    """Extract the requested fields from one listing object."""
    a = attrs_to_dict(listing.get("attributes"))
    ext = attrs_to_dict(listing.get("extendedAttributes"))
    return {
        "brand": brand_name,
        "model": a.get("model", ""),
        "fuel": a.get("fuel", ""),               # Benzine / Diesel / Elektrisch / LPG ...
        "build_year": a.get("constructionYear", ""),
        "mileage_km": a.get("mileage", ""),
        "transmission": a.get("transmission", ""),  # Handgeschakeld / Automaat
        "body_type": a.get("body", ""),             # Hatchback / SUV / Cabriolet / ...
        "title": listing.get("title", ""),
        "price_cents": (listing.get("priceInfo") or {}).get("priceCents", ""),
        "url": BASE + listing.get("vipUrl", "") if listing.get("vipUrl") else "",
    }


def scrape_brand(brand, session, max_pages=None):
    """Iterate all pages of a brand and yield parsed listings one at a time."""
    # First page determines total pages
    first_url = f"{BASE}/l/auto-s/{brand['key']}/"
    data = get_next_data(first_url, session)
    if not data:
        return

    sr = data["props"]["pageProps"]["searchRequestAndResponse"]
    total_pages = sr.get("maxAllowedPageNumber", 1)
    if max_pages:
        total_pages = min(total_pages, max_pages)

    print(f"  {brand['name']}: {sr.get('totalResultCount', '?')} ads, "
          f"{total_pages} pages")

    for page in range(1, total_pages + 1):
        if page == 1:
            page_data = data  # reuse the page we already fetched
        else:
            polite_sleep()  # pause before fetching the next page
            url = f"{BASE}/l/auto-s/{brand['key']}/p/{page}/"
            try:
                page_data = get_next_data(url, session)
            except requests.RequestException as e:
                print(f"    page {page} failed: {e}")
                continue

        if not page_data:
            continue

        listings = page_data["props"]["pageProps"][
            "searchRequestAndResponse"].get("listings", [])
        for listing in listings:
            # Skip ad/banner objects that have no car attributes
            if not listing.get("attributes"):
                continue
            yield parse_listing(listing, brand["name"])


def parse_args():
    p = argparse.ArgumentParser(
        description="Scrape used-car listings from Marktplaats into a CSV.")
    p.add_argument("-o", "--output", default="marktplaats_cars.csv",
                   help="Output CSV path (default: marktplaats_cars.csv)")
    p.add_argument("--max-pages", type=int, default=None,
                   help="Max pages to fetch per brand (default: all pages)")
    p.add_argument("--max-brands", type=int, default=None,
                   help="Only scrape the first N brands (handy for testing)")
    return p.parse_args()


def main():
    args = parse_args()
    session = requests.Session()
    brands = get_brands(session)
    print(f"Found {len(brands)} brands")
    if args.max_brands:
        brands = brands[:args.max_brands]
        print(f"Limiting to the first {len(brands)} brands")

    # Open the CSV up front and write each brand's rows as they come in, so a
    # valid file always exists and partial progress survives a Ctrl+C.
    total = 0
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        f.flush()
        try:
            for i, brand in enumerate(brands, 1):
                try:
                    for car in scrape_brand(brand, session,
                                            max_pages=args.max_pages):
                        writer.writerow(car)
                        total += 1
                    f.flush()  # persist this brand before moving on
                except requests.RequestException as e:
                    print(f"  {brand['name']} failed: {e}")
                    continue
                print(f"  [{i}/{len(brands)}] total so far: {total}")
                if i < len(brands):
                    polite_sleep()  # pause before the next brand
        except KeyboardInterrupt:
            print("\nInterrupted — keeping what was scraped so far.")

    print(f"Done. {total} cars written to {args.output}")


if __name__ == "__main__":
    main()