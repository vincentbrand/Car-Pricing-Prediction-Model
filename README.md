# CarPrices DataScraper

Scrape used-car listings from [Marktplaats](https://www.marktplaats.nl), merge
them into one dataset, and train a neural network to predict asking prices from a
car's attributes. The pipeline has five stages:

1. **Scrape** — collect listings per brand into timestamped CSVs (`scraper.py`,
   `scrape_interactive.py`, `get_brands.py`).
2. **Merge** — fold the per-brand CSVs in `data/` into one `merged_cars.csv`
   (`merge_data.py`).
3. **Explore** — poke at the data in `data-exploration.ipynb`.
4. **Model** — train a PyTorch price model (`price_model.py`).
5. **Autoresearch** — let an AI agent improve that model autonomously
   (`autoresearch-car-model.py` + `car-program.md`).

Each scraped row has these columns:

```
brand, model, fuel, build_year, mileage_km, transmission, body_type, title, price_cents, url
```

## Requirements

- [uv](https://docs.astral.sh/uv/) — handles Python and dependencies for you.

uv reads the Python version from `.python-version` and the dependencies from
`pyproject.toml` (pandas, numpy, matplotlib, seaborn, torch, requests),
downloading the interpreter automatically if needed. The first `make` target you
run creates the virtualenv via `uv sync`.

## Quick start

```bash
make brands    # fetch the brand list -> car_brands.csv + car_brands.json
make scrape    # interactive: pick a brand & page count -> timestamped CSV
make merge     # combine data/*.csv -> merged_cars.csv
```

Then explore the data in `data-exploration.ipynb`, or train a model:

```bash
uv run price_model.py --data merged_cars.csv
```

## 1. Scraping

### Interactive scrape (`make scrape`)

Reads `car_brands.json` (run `make brands` first), lets you pick a brand from a
numbered list, shows how many pages that brand has, and asks how many to scrape
(Enter for all, or a number up to the maximum). The result is saved to a file
named after the brand, timestamp, and page count, e.g.
`abarth_20260615_134623_2pages.csv`.

### Full scrape (`make run` / `scraper.py`)

`make run` scrapes every brand into `marktplaats_cars.csv`. The CSV is written
incrementally, so the file stays valid even if you stop early with Ctrl+C. Run
the scraper directly for more control:

```bash
uv run scraper.py                              # full scrape -> marktplaats_cars.csv
uv run scraper.py --max-brands 5 --max-pages 2 # quick subset
uv run scraper.py -o cars.csv                  # custom output path
```

| Flag           | Default                | Description                    |
| -------------- | ---------------------- | ------------------------------ |
| `-o/--output`  | `marktplaats_cars.csv` | Output CSV path                |
| `--max-pages`  | all                    | Max pages to fetch per brand   |
| `--max-brands` | all                    | Only scrape the first N brands |

To be polite to the site, both scrapers pause a random **10–15 seconds between
each page** (and between brands in the full run), so a full run takes a long
time. The pause bounds live in `PAGE_DELAY_MIN` / `PAGE_DELAY_MAX` in
`scraper.py`.

## 2. Merging (`make merge` / `merge_data.py`)

`merge_data.py` concatenates every CSV in `data/` into a single
`merged_cars.csv`, checking that each file has the expected header first. Drop
the per-brand scrapes you want to keep into `data/`, then:

```bash
uv run merge_data.py                                 # data/ -> merged_cars.csv
uv run merge_data.py --data-dir data --output all.csv # custom paths
```

`merged_cars.csv` is a *generated* file — rebuild it any time the contents of
`data/` change. It is the default training set for the autoresearch model.

## 3. Exploring (`data-exploration.ipynb`)

A Jupyter notebook for inspecting the merged dataset (distributions, price vs.
age/mileage, per-brand breakdowns). Open it with your editor's notebook support
or `uv run jupyter lab` — the `ipykernel` dev dependency is already installed by
`uv sync`.

## 4. Price model (`price_model.py`)

A small PyTorch "tabular" neural network that predicts a car's asking price from
its attributes. The five text columns (brand/model/fuel/transmission/body_type)
each get a learned embedding; `build_year` becomes the car's age, which together
with `mileage_km` forms the standardized numeric inputs. Everything is
concatenated and fed through an MLP that outputs `log(1 + price)`. It reports MAE,
RMSE, and MAPE in euros on a held-out validation split.

```bash
uv run price_model.py --data merged_cars.csv   # train on the merged dataset
uv run price_model.py --epochs 600 --lr 5e-3   # tweak training
uv run price_model.py --save model.pt          # save the trained bundle
```

Run `uv run price_model.py --help` for the full list of flags (epochs, batch
size, learning rate, dropout, validation fraction, seed, etc.).

## 5. Autoresearch (`autoresearch-car-model.py` + `car-program.md`)

A rework of the price model into the **autoresearch** methodology (Karpathy's
autonomous-research harness — see the vendored reference repo in
[`autoresearch/`](autoresearch/)). An AI agent iterates on the model on its own:
tweak the code, train under a fixed wall-clock budget, keep the change if a
single ground-truth metric improved, otherwise revert — and repeat until stopped.

`autoresearch-car-model.py` is split into two banner-marked regions:

- **FIXED HARNESS — do not modify.** Data cleaning, the fixed-seed train/val
  split, the evaluation metric, and constants (`SEED`, `TIME_BUDGET`, …). Keeping
  it frozen is what makes experiments comparable.
- **EDITABLE — the agent iterates here.** Feature engineering, model
  architecture, optimizer, loss, hyperparameters, training loop.

The north-star metric is **`val_mae`** (mean absolute error in euros, lower is
better). Run the baseline to establish a reference score:

```bash
uv run autoresearch-car-model.py --data merged_cars.csv   # baseline
uv run autoresearch-car-model.py --time-budget 30         # shorter experiments
```

**`car-program.md`** is the lightweight "skill" that drives the loop: point an
agent at it for the setup steps, the keep/discard rules, the results log, and
seed ideas for the search.

## Make targets

| Target         | What it does                                  |
| -------------- | --------------------------------------------- |
| `make install` | Create the virtualenv and install deps        |
| `make brands`  | Fetch the brand list (CSV + JSON)             |
| `make scrape`  | Interactive single-brand scrape (timestamped) |
| `make sample`  | Quick test run (3 brands, 1 page each)        |
| `make run`     | Full scrape of all brands (writes the CSV)    |
| `make merge`   | Merge `data/*.csv` into `merged_cars.csv`     |
| `make lock`    | Refresh `uv.lock` from `pyproject.toml`       |
| `make clean`   | Remove the venv and generated CSVs            |

The two model scripts (`price_model.py`, `autoresearch-car-model.py`) have no
make targets — run them with `uv run` as shown above.

## Project layout

```
get_brands.py               — fetch the brand list (car_brands.csv + .json)
scraper.py                  — full multi-brand scraper
scrape_interactive.py       — interactive single-brand scraper
merge_data.py               — merge data/*.csv -> merged_cars.csv
data-exploration.ipynb      — exploratory analysis notebook
price_model.py              — PyTorch price-prediction model
autoresearch-car-model.py   — price model adapted for the autoresearch loop
car-program.md              — agent instructions for the autoresearch loop
autoresearch/               — vendored reference repo (Karpathy's autoresearch)
data/                       — per-brand scrapes, input to `make merge`
```
