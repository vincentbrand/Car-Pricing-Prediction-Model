"""
autoresearch-car-model.py — Used-car price prediction, set up for *autoresearch*.

This is a rework of price_model.py into the "autoresearch" methodology
(Karpathy's autonomous-research harness, see ./autoresearch/). The idea: an AI
agent iterates on the model autonomously — tweak the code, train under a fixed
budget, check whether a single ground-truth metric improved, keep or discard via
git, and repeat — until a human stops it.

To make that work, the file is split into two clearly-marked regions:

  1. FIXED HARNESS (do NOT modify)
     Mirrors autoresearch's prepare.py. It owns the things that must stay
     constant for experiments to be *comparable*:
       - which rows are valid (data cleaning / filters),
       - the target definition (price_eur),
       - the deterministic train/val split (fixed SEED),
       - the evaluation metric (evaluate()).
     If you change anything here, results from before and after are no longer
     comparable — that's the whole point of keeping it frozen.

  2. EDITABLE (the agent iterates HERE)
     Mirrors autoresearch's train.py. Everything is fair game:
       - feature engineering (FeaturePipeline),
       - model architecture (CarPriceNet),
       - optimizer, loss, hyperparameters, training loop.

THE METRIC
----------
The north-star metric is **val_mae** — mean absolute error in euros on the held-out
validation set, lower is better. (val_rmse and val_mape are also reported.) MAE is
robust and directly interpretable ("we're off by ~EUR X on a typical car").

THE BUDGET
----------
Like autoresearch's 5-minute rule, training runs for a fixed wall-clock TIME_BUDGET
(default 60s here — these tabular models are tiny). Best-validation weights are
checkpointed throughout, so a fixed budget fairly compares any change to model
size, batch size, optimizer, etc. Hold TIME_BUDGET constant across an experiment
series; changing it breaks comparability.

THE LOOP (what an autonomous agent does — adapted from autoresearch/program.md)
-------------------------------------------------------------------------------
  Run the baseline first (this file, unmodified) to establish the reference score.
  Then LOOP:
    1. Edit ONLY the EDITABLE region below with one experimental idea.
    2. git commit
    3. uv run autoresearch-car-model.py --data merged_cars.csv > run.log 2>&1
    4. grep "^val_mae:" run.log     (empty output => it crashed; read tail of run.log)
    5. Log the run to results.tsv (tab-separated, leave it untracked by git):
           commit<TAB>val_mae<TAB>val_mape<TAB>status<TAB>description
       status is one of: keep | discard | crash
    6. If val_mae improved (lower) -> keep the commit (advance the branch).
       If it's equal or worse      -> git reset back to the previous commit.
  Simplicity criterion: at equal score, simpler code wins. Don't add ugly
  complexity for a rounding-error gain; deleting code for equal/better score is
  a win. Never pause to ask "should I keep going?" — iterate until stopped.

Usage
-----
    uv run autoresearch-car-model.py                       # baseline on merged_cars.csv
    uv run autoresearch-car-model.py --data data/bmw_*.csv # a single scrape
    uv run autoresearch-car-model.py --time-budget 30      # shorter experiments

Note: merged_cars.csv may be stale (re-run `uv run merge_data.py` to fold in every
brand currently under data/). The fixed cleaning below tolerates the raw junk
(placeholder prices/mileage of 999999, etc.).
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# =============================================================================
# FIXED HARNESS — DO NOT MODIFY
# -----------------------------------------------------------------------------
# This region defines the dataset, the train/val split, and the metric. Keep it
# frozen so every experiment is scored on the exact same held-out cars. It is the
# analogue of autoresearch/prepare.py.
# =============================================================================

# --- Schema (raw columns produced by scraper.py) ------------------------------
CATEGORICAL = ["brand", "model", "fuel", "transmission", "body_type"]
TARGET = "price_eur"

# --- Fixed constants ----------------------------------------------------------
SEED = 42                 # controls the train/val split — fixed so the val set never changes
VAL_FRAC = 0.2            # fraction of valid rows held out for scoring
TIME_BUDGET = 60.0        # wall-clock training budget in seconds (analogue of the 5-min rule)
REF_YEAR = 2026           # reference year for computing car age

# Row-validity filters. These decide *which cars exist* in the experiment and so
# must stay fixed. They strip placeholder/junk listings (Marktplaats encodes
# "price on request" / "mileage unknown" as 999999-style sentinels).
MIN_PRICE = 500.0         # below this: parts ads / "price on request" placeholders
MAX_PRICE = 250_000.0     # above this: 999999-style placeholders / exotics
MIN_YEAR = 1990           # older listings are sparse oldtimers with erratic pricing
MILEAGE_JUNK = 990_000    # mileage at/above this is a "unknown" sentinel, not a real odometer


def load_and_split(path):
    """Read the scraper CSV, apply the FIXED cleaning, and return a deterministic
    (df, train_idx, val_idx) tuple. Row membership and the target depend only on
    constants above — never on anything in the EDITABLE region — so the held-out
    set is identical across every experiment.
    """
    df = pd.read_csv(path)

    # Target: price_cents -> euros, keep only plausibly-real prices.
    df["price_cents"] = pd.to_numeric(df["price_cents"], errors="coerce")
    df[TARGET] = df["price_cents"] / 100.0
    df = df[df[TARGET].between(MIN_PRICE, MAX_PRICE)].copy()

    # Build year is required (no year -> no age), and must be in a sane range.
    df["build_year"] = pd.to_numeric(df["build_year"], errors="coerce")
    df = df[df["build_year"].between(MIN_YEAR, REF_YEAR + 1)].copy()

    # Mileage: keep the column but null out junk sentinels. Imputation is a
    # *feature* choice and lives in the EDITABLE region — we do NOT filter on it,
    # so handling mileage differently can never change which rows are scored.
    df["mileage_km"] = pd.to_numeric(df["mileage_km"], errors="coerce")
    df.loc[~df["mileage_km"].between(1, MILEAGE_JUNK - 1), "mileage_km"] = np.nan

    # Categoricals: blanks / NaN become an explicit "Unknown" class.
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").replace("", "Unknown").astype(str)

    df = df.reset_index(drop=True)

    # Deterministic split.
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(df))
    n_val = max(1, int(round(len(df) * VAL_FRAC)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]
    return df, train_idx, val_idx


@torch.no_grad()
def evaluate(model, pipeline, df, idx, device):
    """THE METRIC (do not change). Returns (mae, rmse, mape, preds) in euros on the
    rows `idx`. The model predicts log1p(price); we invert and compare to truth.
    """
    model.eval()
    x_cat, x_num = pipeline.transform(df.iloc[idx])
    log_pred = model(x_cat.to(device), x_num.to(device)).cpu().numpy()
    pred = np.clip(np.expm1(log_pred), 0, None)         # invert log1p target
    actual = df.iloc[idx][TARGET].to_numpy(dtype=np.float64)
    err = pred - actual
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = float(np.mean(np.abs(err) / actual) * 100)
    return mae, rmse, mape, pred


# =============================================================================
# EDITABLE — THE AGENT ITERATES HERE
# -----------------------------------------------------------------------------
# Everything below is fair game: features, architecture, optimizer, loss, the
# training loop, hyperparameters. The analogue of autoresearch/train.py. The only
# rules: it must run without crashing, finish within TIME_BUDGET, and only ever
# fit statistics (scalers/imputers) on the TRAIN split (no validation leakage).
# =============================================================================

# --- Hyperparameters (edit these directly) ------------------------------------
HIDDEN = (128, 64)        # MLP hidden layer widths
DROPOUT = 0.2             # dropout in each hidden block
BATCH_SIZE = 32           # training mini-batch size
LR = 1e-2                 # Adam learning rate (peak, reached after warmup)
LR_WARMUP_FRAC = 0.05     # fraction of the budget spent warming LR up from 0
LR_FINAL_FRAC = 0.0       # final LR as a fraction of peak (cosine-decay target)
WEIGHT_DECAY = 1e-4       # Adam weight decay
LOG_EVERY = 2.0           # seconds between progress prints


def lr_multiplier(progress):
    """Linear warmup to peak, then cosine decay to LR_FINAL_FRAC over the budget."""
    if progress < LR_WARMUP_FRAC:
        return progress / LR_WARMUP_FRAC if LR_WARMUP_FRAC > 0 else 1.0
    t = (progress - LR_WARMUP_FRAC) / (1.0 - LR_WARMUP_FRAC)
    return LR_FINAL_FRAC + (1.0 - LR_FINAL_FRAC) * 0.5 * (1.0 + float(np.cos(np.pi * t)))


def embedding_dim(cardinality):
    """fast.ai-style heuristic for embedding width, clamped to [2, 50]."""
    return int(min(50, max(2, round(1.6 * cardinality ** 0.56))))


class FeaturePipeline:
    """Turns cleaned rows into (categorical-index, numeric) tensors.

    All statistics (category vocab, mileage median, numeric mean/std) are fit on
    the TRAIN rows only and then reused at transform time — this is the editable
    place to invent features, but the no-leakage rule is non-negotiable.
    """

    NUMERIC = ["age", "mileage_km"]   # numeric feature columns (derived below)

    def fit(self, train_df):
        # Category value -> index; index 0 is reserved for unseen/unknown values
        # so the model degrades gracefully on cars it never saw in training.
        self.maps = {}
        for col in CATEGORICAL:
            cats = sorted(train_df[col].unique())
            self.maps[col] = {c: i + 1 for i, c in enumerate(cats)}
        self.cardinalities = [len(self.maps[c]) + 1 for c in CATEGORICAL]

        # Mileage imputation value (train median), then numeric standardization.
        self.mileage_median = float(train_df["mileage_km"].median())
        num = self._raw_numeric(train_df)
        self.num_mean = num.mean(axis=0)
        self.num_std = num.std(axis=0) + 1e-6
        return self

    def _raw_numeric(self, df):
        age = (REF_YEAR - df["build_year"].to_numpy(dtype=np.float32))
        mileage = df["mileage_km"].fillna(self.mileage_median).to_numpy(dtype=np.float32)
        return np.stack([age, mileage], axis=1).astype(np.float32)

    def transform(self, df):
        cat = np.stack(
            [df[col].map(lambda v: self.maps[col].get(v, 0)).to_numpy() for col in CATEGORICAL],
            axis=1,
        ).astype(np.int64)
        num = (self._raw_numeric(df) - self.num_mean) / self.num_std
        return torch.tensor(cat), torch.tensor(num, dtype=torch.float32)


class CarPriceNet(nn.Module):
    """Categorical embeddings + numeric features -> MLP -> log-price."""

    def __init__(self, cat_cardinalities, n_numeric, hidden=HIDDEN, dropout=DROPOUT):
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
                nn.LayerNorm(h),     # LayerNorm works at any batch size (incl. 1)
                nn.Dropout(dropout),
            ]
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x_cat, x_num):
        embs = [emb(x_cat[:, i]) for i, emb in enumerate(self.embeddings)]
        x = torch.cat(embs + [x_num], dim=1)
        return self.mlp(x).squeeze(1)


def train(df, train_idx, val_idx, time_budget, device, show):
    """Fit a CarPriceNet within `time_budget` seconds, keeping best-val weights."""
    # --- Build features (fit on TRAIN only) ----------------------------------
    pipeline = FeaturePipeline().fit(df.iloc[train_idx])
    xc_tr, xn_tr = pipeline.transform(df.iloc[train_idx])
    yl_tr = torch.tensor(
        np.log1p(df.iloc[train_idx][TARGET].to_numpy(dtype=np.float32))
    )

    train_loader = DataLoader(
        TensorDataset(xc_tr, xn_tr, yl_tr),
        batch_size=min(BATCH_SIZE, len(train_idx)),
        shuffle=True,
        drop_last=False,
    )

    # --- Model / optimizer / loss --------------------------------------------
    model = CarPriceNet(pipeline.cardinalities, n_numeric=len(FeaturePipeline.NUMERIC)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.SmoothL1Loss()   # robust to the occasional pricing outlier

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device} | rows: {len(df)} (train {len(train_idx)} / val {len(val_idx)})")
    print(f"Categorical cardinalities: {dict(zip(CATEGORICAL, pipeline.cardinalities))}")
    print(f"Model params: {num_params:,} | time budget: {time_budget:.0f}s\n")

    # --- Training loop (fixed wall-clock budget, best-val checkpoint) ---------
    best_mae, best_state, best_epoch = float("inf"), None, 0
    train_seconds, epoch, last_log = 0.0, 0, 0.0

    while train_seconds < time_budget:
        epoch += 1
        model.train()
        # LR schedule: warmup then cosine decay, keyed to fraction of budget used.
        lrm = lr_multiplier(min(train_seconds / time_budget, 1.0))
        for g in optimizer.param_groups:
            g["lr"] = LR * lrm
        t0 = time.time()
        for xc, xn, yl in train_loader:
            xc, xn, yl = xc.to(device), xn.to(device), yl.to(device)
            optimizer.zero_grad()
            loss = loss_fn(model(xc, xn), yl)
            loss.backward()
            optimizer.step()
        train_seconds += time.time() - t0   # only training counts toward the budget

        # Score on val (cheap; not counted against the budget) and keep the best.
        mae, rmse, mape, _ = evaluate(model, pipeline, df, val_idx, device)
        if mae < best_mae:
            best_mae, best_epoch = mae, epoch
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if train_seconds - last_log >= LOG_EVERY:
            last_log = train_seconds
            print(f"epoch {epoch:5d} | {train_seconds:5.1f}s/{time_budget:.0f}s | lr {LR*lrm:.1e} | "
                  f"val MAE EUR {mae:,.0f} | RMSE EUR {rmse:,.0f} | MAPE {mape:.1f}% "
                  f"(best MAE EUR {best_mae:,.0f} @ epoch {best_epoch})")

    # Restore best-val weights before final scoring / prediction.
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, pipeline, num_params, epoch, best_epoch


def predict_price(model, pipeline, car, device):
    """Predict the euro price for a single car given as a plain dict."""
    row = pd.DataFrame([{
        "build_year": float(car["build_year"]),
        "mileage_km": float(car.get("mileage_km", np.nan)),
        **{c: str(car.get(c, "Unknown")) for c in CATEGORICAL},
    }])
    x_cat, x_num = pipeline.transform(row)
    model.eval()
    with torch.no_grad():
        log_pred = model(x_cat.to(device), x_num.to(device)).item()
    return float(np.expm1(log_pred))


# =============================================================================
# MAIN — orchestration & the autoresearch summary block (do not modify)
# =============================================================================

def main(args):
    t_start = time.time()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    df, train_idx, val_idx = load_and_split(args.data)
    if len(df) < 10:
        raise SystemExit(f"Only {len(df)} usable rows in {args.data} — too few to train.")

    t_train_start = time.time()
    model, pipeline, num_params, n_epochs, best_epoch = train(
        df, train_idx, val_idx, args.time_budget, device, args.show
    )
    train_wall = time.time() - t_train_start

    # --- Final report on the best-val model ----------------------------------
    mae, rmse, mape, pred = evaluate(model, pipeline, df, val_idx, device)
    print("\n=== Validation performance (best model) ===")
    print(f"MAE  : EUR {mae:,.0f}")
    print(f"RMSE : EUR {rmse:,.0f}")
    print(f"MAPE : {mape:.1f}%")

    print("\nExample validation predictions:")
    print(f"  {'predicted':>12} {'actual':>12}   car")
    for i, row_idx in enumerate(val_idx[: args.show]):
        row = df.iloc[row_idx]
        print(f"  EUR {pred[i]:>8,.0f} EUR {row[TARGET]:>8,.0f}   "
              f"{row['build_year']:.0f} {row['brand']} {row['model']} {row['fuel']} "
              f"{row['mileage_km'] if pd.notna(row['mileage_km']) else float('nan'):,.0f} km")

    # --- Demo: predict an arbitrary car --------------------------------------
    example = {
        "brand": "Mazda", "model": "CX-5", "fuel": "Benzine",
        "build_year": 2018, "mileage_km": 90000,
        "transmission": "Automaat", "body_type": "SUV of Terreinwagen",
    }
    price = predict_price(model, pipeline, example, device)
    print(f"\nPredicted price for {example['build_year']} {example['model']} "
          f"({example['mileage_km']:,} km): EUR {price:,.0f}")

    if args.save:
        torch.save({
            "state_dict": model.state_dict(),
            "maps": pipeline.maps, "cardinalities": pipeline.cardinalities,
            "mileage_median": pipeline.mileage_median,
            "num_mean": pipeline.num_mean, "num_std": pipeline.num_std,
            "ref_year": REF_YEAR,
        }, args.save)
        print(f"\nSaved model bundle to {args.save}")

    # --- Autoresearch summary block (parse with: grep "^val_mae:" run.log) ----
    total_seconds = time.time() - t_start
    print("---")
    print(f"val_mae:          {mae:.2f}")
    print(f"val_rmse:         {rmse:.2f}")
    print(f"val_mape:         {mape:.2f}")
    print(f"train_seconds:    {train_wall:.1f}")
    print(f"total_seconds:    {total_seconds:.1f}")
    print(f"num_params:       {num_params}")
    print(f"num_rows:         {len(df)}")
    print(f"n_train:          {len(train_idx)}")
    print(f"n_val:            {len(val_idx)}")
    print(f"n_epochs:         {n_epochs}")
    print(f"best_epoch:       {best_epoch}")
    return model


def parse_args():
    p = argparse.ArgumentParser(description="Train a car-price NN under the autoresearch loop.")
    p.add_argument("--data", default="merged_cars.csv", type=Path, help="Input CSV (scraper output)")
    p.add_argument("--time-budget", type=float, default=TIME_BUDGET,
                   help="Wall-clock training budget in seconds (hold constant across an experiment series)")
    p.add_argument("--show", type=int, default=8, help="How many example predictions to print")
    p.add_argument("--save", type=Path, default=None, help="Optional path to save the trained model bundle")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
