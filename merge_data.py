#!/usr/bin/env python3
"""Merge all per-brand CSV files in data/ into a single combined CSV.

All source files share the same header:
    brand,model,fuel,build_year,mileage_km,transmission,body_type,title,price_cents,url

Usage:
    python merge_data.py [--data-dir data] [--output merged_cars.csv]
"""

import argparse
import csv
import sys
from pathlib import Path

EXPECTED_HEADER = [
    "brand",
    "model",
    "fuel",
    "build_year",
    "mileage_km",
    "transmission",
    "body_type",
    "title",
    "price_cents",
    "url",
]


def merge(data_dir: Path, output: Path) -> None:
    csv_files = sorted(p for p in data_dir.glob("*.csv") if p.resolve() != output.resolve())
    if not csv_files:
        sys.exit(f"No CSV files found in {data_dir}")

    url_index = EXPECTED_HEADER.index("url")
    seen_urls: set[str] = set()
    total_rows = 0
    duplicate_rows = 0
    header_written = False

    with output.open("w", newline="", encoding="utf-8") as out_f:
        writer = csv.writer(out_f)

        for path in csv_files:
            with path.open("r", newline="", encoding="utf-8") as in_f:
                reader = csv.reader(in_f)
                try:
                    header = next(reader)
                except StopIteration:
                    print(f"  skipping empty file: {path.name}")
                    continue

                if header != EXPECTED_HEADER:
                    sys.exit(
                        f"Header mismatch in {path.name}:\n"
                        f"  expected: {EXPECTED_HEADER}\n"
                        f"  found:    {header}"
                    )

                if not header_written:
                    writer.writerow(header)
                    header_written = True

                kept = 0
                dupes = 0
                for row in reader:
                    url = row[url_index]
                    if url in seen_urls:
                        dupes += 1
                        continue
                    seen_urls.add(url)
                    writer.writerow(row)
                    kept += 1

                total_rows += kept
                duplicate_rows += dupes
                suffix = f" ({dupes} duplicates skipped)" if dupes else ""
                print(f"  {path.name}: {kept} rows{suffix}")

    print(
        f"\nMerged {len(csv_files)} files -> {output} "
        f"({total_rows} unique data rows, {duplicate_rows} duplicates removed)"
    )


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Merge per-brand car CSV files into one.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=script_dir / "data",
        help="Directory containing the source CSV files (default: ./data)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=script_dir / "merged_cars.csv",
        help="Path for the merged output CSV (default: ./merged_cars.csv)",
    )
    args = parser.parse_args()

    merge(args.data_dir, args.output)


if __name__ == "__main__":
    main()
