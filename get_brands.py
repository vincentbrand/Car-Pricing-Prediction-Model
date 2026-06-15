import requests
import json
import re
import csv

BASE = "https://www.marktplaats.nl"
ALL_CARS_URL = f"{BASE}/l/auto-s/"

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

# Categories that appear in the list but are not actual car brands.
NON_BRAND_KEYS = {
    "auto-s",         # the top-level category itself
    "bestelauto-s",   # vans
    "vrachtwagens",   # trucks
    "oldtimers",      # not a brand
    "overige-auto-s", # "other"
}


def get_next_data(url, session):
    resp = session.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    m = NEXT_DATA_RE.search(resp.text)
    if not m:
        raise RuntimeError("Could not find __NEXT_DATA__ in page")
    return json.loads(m.group(1))


def get_brands(session):
    """Return a list of brand dicts: {id, key, name, url}."""
    data = get_next_data(ALL_CARS_URL, session)
    options = (data["props"]["pageProps"]
                   ["searchRequestAndResponse"]["searchCategoryOptions"])

    brands = []
    for o in options:
        key = o.get("key", "")
        if key in NON_BRAND_KEYS:
            continue
        brands.append({
            "id": o["id"],
            "key": key,
            "name": o["name"],
            "url": f"{BASE}/l/auto-s/{key}/",
        })

    # Sort alphabetically by name for a tidy file
    brands.sort(key=lambda b: b["name"].lower())
    return brands


def save_csv(brands, path="car_brands.csv"):
    fields = ["id", "name", "key", "url"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(brands)
    print(f"Wrote {len(brands)} brands to {path}")


def save_json(brands, path="car_brands.json"):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(brands, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(brands)} brands to {path}")


def main():
    session = requests.Session()
    brands = get_brands(session)
    print(f"Found {len(brands)} car brands")
    save_csv(brands)
    save_json(brands)


if __name__ == "__main__":
    main()