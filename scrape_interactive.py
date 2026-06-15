"""Interactive Marktplaats scraper.

Pick a car brand from car_brands.json, see how many pages are available, choose
how many to scrape, and save the listings to a timestamped CSV named after the
brand, the date/time, and the number of pages scraped.

Run with:  uv run scrape_interactive.py   (or `make scrape`)
"""
import csv
import json
import os
import sys
from datetime import datetime

import requests

# Reuse the helpers from the bulk scraper so the parsing stays consistent.
from scraper import BASE, FIELDS, get_next_data, parse_listing, polite_sleep

BRANDS_FILE = "car_brands.json"


def load_brands(path=BRANDS_FILE):
    """Load the brand list produced by get_brands.py."""
    if not os.path.exists(path):
        sys.exit(f"'{path}' not found. Run `make brands` first to create it.")
    with open(path, encoding="utf-8") as f:
        brands = json.load(f)
    if not brands:
        sys.exit(f"'{path}' is empty. Re-run `make brands`.")
    return brands


def choose_brand(brands):
    """Show a numbered list and return the brand the user picks."""
    print(f"\nAvailable brands ({len(brands)}):\n")
    labels = [f"{i + 1:>2}. {b['name']}" for i, b in enumerate(brands)]
    cols = 3
    colw = max(len(s) for s in labels) + 3
    nrows = (len(labels) + cols - 1) // cols
    for r in range(nrows):
        cells = []
        for c in range(cols):
            idx = r + c * nrows
            if idx < len(labels):
                cells.append(labels[idx].ljust(colw))
        print("".join(cells).rstrip())
    print()

    while True:
        choice = input("Select a brand by number (or 'q' to quit): ").strip()
        if choice.lower() in ("q", "quit", "exit"):
            sys.exit("Cancelled.")
        if choice.isdigit() and 1 <= int(choice) <= len(brands):
            return brands[int(choice) - 1]
        print(f"  Please enter a number between 1 and {len(brands)}.")


def get_page_info(brand, session):
    """Fetch the first page and return (data, total_pages, total_ads)."""
    url = f"{BASE}/l/auto-s/{brand['key']}/"
    data = get_next_data(url, session)
    if not data:
        sys.exit(f"Could not load listings for {brand['name']}.")
    sr = data["props"]["pageProps"]["searchRequestAndResponse"]
    total_pages = sr.get("maxAllowedPageNumber", 1)
    total_ads = sr.get("totalResultCount", "?")
    return data, total_pages, total_ads


def choose_pages(total_pages):
    """Ask how many pages to scrape: Enter = all, or a number up to the max."""
    while True:
        ans = input(
            f"Scrape all {total_pages} pages? "
            f"Press Enter for all, or type a number (1-{total_pages}): "
        ).strip()
        if ans == "":
            return total_pages
        if ans.isdigit() and 1 <= int(ans) <= total_pages:
            return int(ans)
        print(f"  Enter a number between 1 and {total_pages}, or just press Enter.")


def scrape(brand, session, first_page_data, pages):
    """Scrape the requested number of pages and return parsed rows."""
    rows = []
    for page in range(1, pages + 1):
        if page == 1:
            data = first_page_data  # reuse the page we already fetched
        else:
            polite_sleep()  # pause before fetching the next page
            url = f"{BASE}/l/auto-s/{brand['key']}/p/{page}/"
            try:
                data = get_next_data(url, session)
            except requests.RequestException as e:
                print(f"  page {page} failed: {e}")
                continue

        if not data:
            continue

        listings = data["props"]["pageProps"][
            "searchRequestAndResponse"].get("listings", [])
        for listing in listings:
            if not listing.get("attributes"):
                continue
            rows.append(parse_listing(listing, brand["name"]))
        print(f"  page {page}/{pages}: {len(rows)} listings so far")
    return rows


def output_filename(brand, pages):
    """Build a filename like 'abarth_20260615_134500_2pages.csv'."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{brand['key']}_{stamp}_{pages}pages.csv"


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def main():
    brands = load_brands()
    brand = choose_brand(brands)

    session = requests.Session()
    print(f"\nLoading {brand['name']} ...")
    first_page_data, total_pages, total_ads = get_page_info(brand, session)
    print(f"{brand['name']}: {total_ads} ads across {total_pages} pages.")

    pages = choose_pages(total_pages)

    confirm = input(
        f"\nScrape {pages} page(s) of {brand['name']}? [Y/n]: "
    ).strip().lower()
    if confirm not in ("", "y", "yes"):
        sys.exit("Cancelled.")

    print()
    rows = scrape(brand, session, first_page_data, pages)
    path = output_filename(brand, pages)
    write_csv(rows, path)
    print(f"\nDone. {len(rows)} cars written to {path}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit("\nCancelled.")
