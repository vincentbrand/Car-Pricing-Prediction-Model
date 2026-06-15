# Marktplaats car scraper — convenience commands.
# Run `make` (or `make help`) to see the available targets.

.DEFAULT_GOAL := help
.PHONY: help install brands scrape run sample merge autoresearch lock clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Create the virtualenv and install dependencies
	uv sync

brands: install ## Fetch the brand list (writes car_brands.csv + car_brands.json)
	uv run get_brands.py

scrape: install ## Interactive scrape: pick a brand & page count -> timestamped CSV
	uv run scrape_interactive.py

run: install ## Full scrape of all brands (writes marktplaats_cars.csv)
	uv run scraper.py

sample: install ## Quick test run: first 3 brands, 1 page each
	uv run scraper.py --max-brands 3 --max-pages 1

merge: install ## Merge per-brand CSVs in data/ into merged_cars.csv
	uv run merge_data.py

autoresearch: install merged_cars.csv ## Train the price model with the autoresearch harness (one experiment)
	uv run autoresearch-car-model.py

# Build merged_cars.csv only if it's missing or the per-brand scrapes are newer.
# Not rebuilt mid-series unless data/ changes, since re-merging would shift the val set.
merged_cars.csv: $(wildcard data/*.csv) merge_data.py
	uv run merge_data.py

lock: ## Refresh uv.lock from pyproject.toml
	uv lock

clean: ## Remove the virtualenv and generated output files
	rm -rf .venv
	rm -f marktplaats_cars.csv car_brands.csv car_brands.json
	rm -f *pages.csv
