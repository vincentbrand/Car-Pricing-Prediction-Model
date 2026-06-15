"""
price_model.py — Predict used-car prices with a PyTorch neural network.

Trains a small "tabular" neural network on the scraped Marktplaats listings
(the CSVs produced by scraper.py) to predict a car's asking price from its
structured attributes: brand, model, fuel type, build year, mileage,
transmission and body type.

How it models the data
----------------------
* The five text columns (brand/model/fuel/transmission/body_type) are
  *categorical*. Each gets its own learned embedding table, which lets the
  network discover, for example, that an "SUV of Terreinwagen" body and a
  "CX-60" model push the price up.
* build_year is turned into the car's `age`, and together with `mileage_km`
  forms the *numeric* inputs, which are standardized (zero mean, unit std).
* The embeddings and numeric features are concatenated and fed through a
  small multi-layer perceptron that outputs one number: log(1 + price).
  Training on log-price keeps the loss well behaved across the wide price
  range (~EUR 2,000 to ~EUR 40,000) and guarantees positive predictions.

Usage
-----
    uv run price_model.py                          # default: the 2-page Mazda CSV
    uv run price_model.py --data some_scrape.csv   # train on a bigger scrape
    uv run price_model.py --epochs 600 --lr 5e-3   # tweak training

Tip: the 2-page Mazda file is tiny (~60 rows) and mainly useful as a smoke
test. Point --data at a larger scrape (e.g. the 110-page Mazda CSV) for a
model that actually generalises.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# --- Columns, matching the scraper's CSV header --------------------------------
CATEGORICAL = ["brand", "model", "fuel", "transmission", "body_type"]
NUMERIC = ["age", "mileage_km"]   # `age` is derived from build_year below
TARGET = "price_eur"

DEFAULT_DATA = "mazda_20260615_135926_2pages.csv"


# ------------------------------------------------------------------------------
# Data loading & cleaning
# ------------------------------------------------------------------------------
def load_dataframe(path, ref_year, min_price):
    """Read the scraper CSV and return a cleaned DataFrame ready for encoding."""
    df = pd.read_csv(path)

    # price_cents -> euros, then drop rows below `min_price`. A missing/zero
    # price or a handful-of-euros price is a placeholder / "price on request" /
    # parts ad, not a real car listing — those would otherwise wreck both the
    # fit and the percentage-error metric.
    df["price_cents"] = pd.to_numeric(df["price_cents"], errors="coerce")
    df[TARGET] = df["price_cents"] / 100.0
    df = df[df[TARGET].notna() & (df[TARGET] >= min_price)].copy()

    # Numeric features. A car without a build year can't get an age, so drop it;
    # a missing mileage is filled with the median (a reasonable default).
    df["build_year"] = pd.to_numeric(df["build_year"], errors="coerce")
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")
    df = df[df["build_year"].notna()].copy()
    df["age"] = ref_year - df["build_year"]
    df["mileage_km"] = df["mileage_km"].fillna(df["mileage_km"].median())

    # Categorical features: treat blanks / NaN as an explicit "Unknown" class.
    for col in CATEGORICAL:
        df[col] = (
            df[col].fillna("Unknown").replace("", "Unknown").astype(str)
        )

    return df.reset_index(drop=True)


# ------------------------------------------------------------------------------
# Feature encoding
# ------------------------------------------------------------------------------
def build_category_maps(df):
    """Map each category value to an integer index. Index 0 is reserved for
    unseen / unknown values so the model degrades gracefully at predict time."""
    maps = {}
    for col in CATEGORICAL:
        cats = sorted(df[col].unique())
        maps[col] = {c: i + 1 for i, c in enumerate(cats)}
    return maps


def encode_categoricals(df, maps):
    """Turn the categorical columns into an int64 matrix of embedding indices."""
    cols = [df[col].map(lambda v: maps[col].get(v, 0)).to_numpy() for col in CATEGORICAL]
    return np.stack(cols, axis=1).astype(np.int64)


def embedding_dim(cardinality):
    """A small heuristic for embedding size (fast.ai-style), clamped to [2, 50]."""
    return int(min(50, max(2, round(1.6 * cardinality ** 0.56))))


# ------------------------------------------------------------------------------
# Model
# ------------------------------------------------------------------------------
class CarPriceNet(nn.Module):
    """Categorical embeddings + numeric features -> MLP -> log-price."""

    def __init__(self, cat_cardinalities, n_numeric, hidden=(128, 64), dropout=0.2):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(card, embedding_dim(card)) for card in cat_cardinalities]
        )
        emb_total = sum(emb.embedding_dim for emb in self.embeddings)

        layers, in_dim = [], emb_total + n_numeric
        for h in hidden:
            layers += [
                nn.Linear(in_dim, h),
                nn.ReLU(),
                nn.LayerNorm(h),     # LayerNorm works with any batch size (incl. 1)
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.mlp(x).squeeze(1)


# ------------------------------------------------------------------------------
# Training / evaluation
# ------------------------------------------------------------------------------
def euro_metrics(model, x_cat, x_num, y_eur, device):
    """Return (MAE, RMSE, MAPE) in euros on the given split."""
    model.eval()
    with torch.no_grad():
        log_pred = model(x_cat.to(device), x_num.to(device)).cpu().numpy()
    pred = np.expm1(log_pred)                       # invert the log1p target
    pred = np.clip(pred, 0, None)
    actual = y_eur.numpy()
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / actual) * 100)
    return mae, rmse, mape, pred


def train(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # --- Load & encode ---------------------------------------------------------
    df = load_dataframe(args.data, args.ref_year, args.min_price)
    if len(df) < 10:
        raise SystemExit(f"Only {len(df)} usable rows in {args.data} — too few to train.")

    maps = build_category_maps(df)
    x_cat = encode_categoricals(df, maps)
    x_num = df[NUMERIC].to_numpy(dtype=np.float32)
    y_eur = df[TARGET].to_numpy(dtype=np.float32)
    y_log = np.log1p(y_eur).astype(np.float32)      # train against log(1 + price)

    # --- Train / validation split ---------------------------------------------
    rng = np.random.RandomState(args.seed)
    perm = rng.permutation(len(df))
    n_val = max(1, int(round(len(df) * args.val_frac)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    # Standardize numeric features using TRAIN statistics only (avoid leakage).
    num_mean = x_num[train_idx].mean(axis=0)
    num_std = x_num[train_idx].std(axis=0) + 1e-6
    x_num_std = (x_num - num_mean) / num_std

    def split(idx):
        return (
            torch.tensor(x_cat[idx]),
            torch.tensor(x_num_std[idx]),
            torch.tensor(y_log[idx]),
            torch.tensor(y_eur[idx]),
        )

    xc_tr, xn_tr, yl_tr, ye_tr = split(train_idx)
    xc_va, xn_va, yl_va, ye_va = split(val_idx)

    train_loader = DataLoader(
        TensorDataset(xc_tr, xn_tr, yl_tr),
        batch_size=min(args.batch_size, len(train_idx)),
        shuffle=True,
        drop_last=False,
    )

    # --- Build model -----------------------------------------------------------
    cardinalities = [len(maps[c]) + 1 for c in CATEGORICAL]  # +1 for the 0/unknown slot
    model = CarPriceNet(cardinalities, n_numeric=len(NUMERIC), dropout=args.dropout).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.SmoothL1Loss()   # robust to the occasional pricing outlier

    print(f"Device: {device} | rows: {len(df)} (train {len(train_idx)} / val {len(val_idx)})")
    print(f"Categorical cardinalities: {dict(zip(CATEGORICAL, cardinalities))}\n")

    # --- Training loop ---------------------------------------------------------
    best_val, best_state = float("inf"), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xc, xn, yl in train_loader:
            xc, xn, yl = xc.to(device), xn.to(device), yl.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xc, xn), yl)
            loss.backward()
            optimizer.step()

        # Validate on log-price (same scale as the loss) and keep the best model.
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(
                model(xc_va.to(device), xn_va.to(device)), yl_va.to(device)
            ).item()
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if epoch % args.log_every == 0 or epoch == 1:
            mae, rmse, mape, _ = euro_metrics(model, xc_va, xn_va, ye_va, device)
            print(f"epoch {epoch:4d} | val_loss {val_loss:.4f} | "
                  f"MAE EUR {mae:,.0f} | RMSE EUR {rmse:,.0f} | MAPE {mape:.1f}%")

    # Restore the best-validation weights before reporting / predicting.
    if best_state is not None:
        model.load_state_dict(best_state)

    # --- Final report ----------------------------------------------------------
    mae, rmse, mape, pred = euro_metrics(model, xc_va, xn_va, ye_va, device)
    print("\n=== Validation performance (best model) ===")
    print(f"MAE  : EUR {mae:,.0f}")
    print(f"RMSE : EUR {rmse:,.0f}")
    print(f"MAPE : {mape:.1f}%")

    print("\nExample validation predictions:")
    print(f"  {'predicted':>12} {'actual':>12}   car")
    for i, row_idx in enumerate(val_idx[: args.show]):
        row = df.iloc[row_idx]
        print(f"  EUR {pred[i]:>8,.0f} EUR {row[TARGET]:>8,.0f}   "
              f"{row['build_year']:.0f} {row['model']} {row['fuel']} "
              f"{row['mileage_km']:,.0f} km")

    # --- Demo: predict the price of an arbitrary car ---------------------------
    example = {
        "brand": "Mazda", "model": "CX-5", "fuel": "Benzine",
        "build_year": 2018, "mileage_km": 90000,
        "transmission": "Automaat", "body_type": "SUV of Terreinwagen",
    }
    price = predict_price(model, example, maps, num_mean, num_std, args.ref_year, device)
    print(f"\nPredicted price for {example['build_year']} {example['model']} "
          f"({example['mileage_km']:,} km): EUR {price:,.0f}")

    if args.save:
        torch.save(
            {"state_dict": model.state_dict(), "maps": maps,
             "num_mean": num_mean, "num_std": num_std,
             "cardinalities": cardinalities, "ref_year": args.ref_year},
            args.save,
        )
        print(f"\nSaved model bundle to {args.save}")

    return model


def predict_price(model, car, maps, num_mean, num_std, ref_year, device):
    """Predict the euro price for a single car given as a plain dict."""
    cat = np.array([[maps[c].get(str(car.get(c, "Unknown")), 0) for c in CATEGORICAL]], dtype=np.int64)
    age = ref_year - float(car["build_year"])
    num = np.array([[age, float(car["mileage_km"])]], dtype=np.float32)
    num = (num - num_mean) / num_std

    model.eval()
    with torch.no_grad():
        log_pred = model(
            torch.tensor(cat).to(device), torch.tensor(num).to(device)
        ).item()
    return float(np.expm1(log_pred))


def parse_args():
    p = argparse.ArgumentParser(description="Train a PyTorch NN to predict used-car prices.")
    p.add_argument("--data", default=DEFAULT_DATA, type=Path, help="Input CSV (scraper output)")
    p.add_argument("--ref-year", type=int, default=2026, help="Reference year for computing car age")
    p.add_argument("--min-price", type=float, default=500.0,
                   help="Drop listings priced below this (placeholder / parts ads)")
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--val-frac", type=float, default=0.2, help="Fraction of rows used for validation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--show", type=int, default=8, help="How many example predictions to print")
    p.add_argument("--save", type=Path, default=None, help="Optional path to save the trained model")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
