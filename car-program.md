# car-program — autoresearch loop for the used-car price model

This is the autonomous-research "skill" for `autoresearch-car-model.py`, adapted
from `autoresearch/program.md`. Point an agent here and let it improve the model
on its own: tweak the code, train under a fixed budget, keep the change if a
single metric improved, otherwise revert — repeat until a human stops it.

## The one-file split (read this first)

`autoresearch-car-model.py` has two clearly-banner-marked regions:

- **FIXED HARNESS — DO NOT MODIFY.** Data cleaning, the fixed-seed train/val
  split, the `evaluate()` metric, and the constants `SEED` / `VAL_FRAC` /
  `TIME_BUDGET` / `REF_YEAR` / the row-validity filters. This is the ground truth
  that makes experiments comparable. Touching it invalidates all prior results.
- **EDITABLE — THE AGENT ITERATES HERE.** Feature engineering (`FeaturePipeline`),
  the model (`CarPriceNet`), optimizer, loss, hyperparameters, the training loop.
  Everything here is fair game.

## The metric

The north-star metric is **`val_mae`** — mean absolute error in euros on the
held-out validation set, **lower is better**. `val_rmse` and `val_mape` are
reported too, but `val_mae` is what you optimize and what decides keep/discard.

## The budget

Training runs for a fixed wall-clock `TIME_BUDGET` (default 60s), with best-val
weights checkpointed throughout. Because the budget and the eval set are fixed,
*any* change — model size, batch size, optimizer, features — is fairly comparable.
**Do not change `TIME_BUDGET` during an experiment series**; it breaks comparability.

## Setup

To set up a new run, work with the user to:

1. **Agree on a run tag** based on today's date (e.g. `jun15`).
2. **Make discards reversible.** This repo is not yet under git.
   - *Recommended:* `git init && git add -A && git commit -m "baseline"`, then
     `git checkout -b autoresearch/<tag>`. You then keep/discard with commits and
     `git reset` (see the loop below).
   - *No-git fallback:* keep a backup of the current best file:
     `cp autoresearch-car-model.py .best-car-model.py`. "Keep" = overwrite the
     backup; "discard" = restore from it. The loop section spells this out.
3. **Read the in-scope files** for full context: `README.md`,
   `autoresearch-car-model.py` (both regions), and skim `merge_data.py`.
4. **Verify data exists.** `merged_cars.csv` is a *generated* file (it can be
   wiped by `make clean`). If it's missing or stale, rebuild it from the per-brand
   scrapes in `data/`: `uv run merge_data.py`. Do this once, before the loop —
   never mid-series, since it changes the val set.
5. **Initialize `results.tsv`** with the header and the baseline row (already
   seeded — see "Logging results"). Leave `results.tsv` untracked by git.
6. **Confirm and go.** Once confirmed, kick off the loop and don't stop.

## What you CAN and CANNOT do

**CAN:** edit anything in the EDITABLE region — features, architecture, optimizer,
loss, hyperparameters, training loop, model size, batch size, LR schedule, etc.

**CANNOT:**
- Modify the FIXED HARNESS region (cleaning, split, `evaluate()`, the constants).
- Change `TIME_BUDGET`, `SEED`, `--data`, or the metric mid-series.
- Re-merge / swap the dataset mid-series (changes which cars are scored).
- Leak the validation set: scalers, imputers, and category vocabularies must be
  fit on the **train split only** (`FeaturePipeline.fit` already enforces this).

**Goal:** the lowest `val_mae`. Training time is fixed, so don't worry about it —
just get the metric down without crashing or blowing past the budget.

**Simplicity criterion:** all else equal, simpler is better. A tiny `val_mae` gain
that adds ugly complexity is not worth it. Removing code for equal-or-better score
is a clean win. Weigh complexity cost against improvement magnitude.

**The first run** is always the baseline: run the file unmodified to establish the
reference score (already recorded — `val_mae` ≈ 3432.91).

## Output format

When the script finishes it prints a parseable summary block:

```
---
val_mae:          3432.91
val_rmse:         8191.09
val_mape:         18.44
train_seconds:    60.9
total_seconds:    61.0
num_params:       23311
num_rows:         12195
n_train:          9756
n_val:            2439
n_epochs:         80
best_epoch:       38
```

Extract the key metric from the log with:

```
grep "^val_mae:" run.log
```

## Logging results

Log every experiment to `results.tsv` (**tab-separated**, NOT comma-separated —
commas break in descriptions). Leave the file untracked by git. The header and 5
columns:

```
commit	val_mae	val_mape	status	description
```

1. short git commit hash (7 chars), or `baseline` / a short tag if not using git
2. `val_mae` achieved (e.g. 3432.91) — use `0.00` for crashes
3. `val_mape` achieved (e.g. 18.44) — use `0.00` for crashes
4. status: `keep`, `discard`, or `crash`
5. short text description of what the experiment tried (no tabs)

Example:

```
commit	val_mae	val_mape	status	description
baseline	3432.91	18.44	keep	baseline (default config, 60s budget, merged_cars.csv 7 brands)
a1b2c3d	3301.50	17.90	keep	log-transform mileage feature
b2c3d4e	3450.10	18.60	discard	widen MLP to (256,128) — overfits
c3d4e5f	0.00	0.00	crash	add batchnorm (NaN loss)
```

## The experiment loop

Run on a dedicated branch (e.g. `autoresearch/<tag>`) if using git.

**LOOP FOREVER:**

1. Note the current state (git commit / the `.best-car-model.py` backup).
2. Edit ONLY the EDITABLE region with one experimental idea.
3. Snapshot it: `git commit -am "<idea>"` (or just save the file in no-git mode).
4. Run it: `uv run autoresearch-car-model.py > run.log 2>&1`
   (redirect everything — do NOT use tee or let output flood your context).
5. Read the result: `grep "^val_mae:\|^val_mape:" run.log`
6. If the grep is empty, it crashed: `tail -n 50 run.log` to read the traceback
   and try to fix it. If you can't after a few attempts, give up on the idea.
7. Record the run in `results.tsv` (do NOT commit `results.tsv`).
8. If `val_mae` improved (lower) → **keep**: advance (keep the commit, or
   `cp autoresearch-car-model.py .best-car-model.py`).
9. If `val_mae` is equal or worse → **discard**: revert
   (`git reset --hard HEAD~1`, or `cp .best-car-model.py autoresearch-car-model.py`).

**Timeout:** each experiment should take ~`TIME_BUDGET` + a few seconds. If a run
exceeds 3× the budget, kill it and treat it as a failure (discard + revert).

**Crashes:** if it's something dumb (typo, missing import, OOM), fix and re-run.
If the idea is fundamentally broken, log `crash` and move on.

**NEVER STOP:** once the loop has begun, do NOT pause to ask "should I keep going?"
The human may be away and expects you to work indefinitely until manually stopped.
If you run out of ideas, think harder — re-read the in-scope files, combine
previous near-misses, try more radical changes. The loop runs until interrupted.

## Ideas to seed the search (not exhaustive)

- **Features:** log-transform `mileage_km`; `age²` or `mileage/age` interaction;
  per-brand/model mean-price target encoding (train-only!); clip/winsorize age.
- **Architecture:** deeper/wider MLP; residual blocks; different embedding sizes;
  GELU vs ReLU; dropout sweep; BatchNorm vs LayerNorm.
- **Optimization:** LR warmup + cosine decay; AdamW vs Adam; weight-decay sweep;
  batch-size sweep; gradient clipping; an EMA of weights.
- **Objective:** MAE vs SmoothL1 vs MSE on log-price; Huber delta; quantile loss.
- **Regularization / stability:** the baseline's per-epoch val MAE is noisy — a LR
  schedule or early-stop-on-best is a natural first target.
