# Slump pipeline — how to run

Two files do the work. Run them in order. Total time on this Mac: **~30–40 min** for the first full run.

## Files written for you

| File | Purpose |
|---|---|
| `slump_modeling_full.ipynb` | Full Phase 0–6 pipeline. Produces `slump_full_phase_results.xlsx` (a phase-by-phase log, ~17 sheets). |
| `build_slump_results.py` | Reads the phase log, re-runs each model under both CV regimes, writes a polished `Slump_ML_Results_v2.xlsx` (15 sheets, formatted, mirrors the CS v3 report). |
| `civil_utils.py` | Shared helpers — already in this folder. No changes needed. |

## Step 1 — run the notebook

You have two options.

### Option A: Headless (recommended, walks away to grab coffee)

In Terminal:

```bash
cd /Users/fakhirhassan/Desktop/CivilFYP
/Users/fakhirhassan/opt/anaconda3/bin/jupyter nbconvert --to notebook --execute slump_modeling_full.ipynb --output slump_modeling_full.ipynb --ExecutePreprocessor.timeout=3600
```

This executes every cell and writes the outputs (plots, tables) back into the notebook. Takes ~25–35 min.

### Option B: Interactive (run cells one by one)

```bash
cd /Users/fakhirhassan/Desktop/CivilFYP
/Users/fakhirhassan/opt/anaconda3/bin/jupyter notebook slump_modeling_full.ipynb
```

Then run cells with Shift+Enter, or Cell → Run All from the menu.

**At the end of step 1**, you'll have `slump_full_phase_results.xlsx` in the folder.

## Step 2 — build the polished report

```bash
cd /Users/fakhirhassan/Desktop/CivilFYP
/Users/fakhirhassan/opt/anaconda3/bin/python build_slump_results.py
```

This takes ~3–5 min. It re-evaluates each model under both random KFold and GroupKFold, picks a representative fold's predictions for the per-model sheets, and writes `Slump_ML_Results_v2.xlsx`.

## What you should see

```
Loading slump dataset and tuned params …
  >> Linear Regression
  >> Random Forest
  >> Gradient Boosting
  ...
  >> Stacked Ensemble
Writing workbook …
Saved -> /Users/fakhirhassan/Desktop/CivilFYP/Slump_ML_Results_v2.xlsx
```

## Output files

| File | Contents |
|---|---|
| `slump_full_phase_results.xlsx` | Phase-by-phase working log: VIF tables, permutation importance, all tuning trials, every CV regime, best hyperparameters per model. |
| `Slump_ML_Results_v2.xlsx` | The polished report. 15 sheets: Dataset Overview, Model Performance, one sheet per model, Methodology. Same format as `CompressiveStrength_ML_Results_v3.xlsx`. |

## Common issues

**`ModuleNotFoundError: smogn`** — quick fix:
```bash
/Users/fakhirhassan/opt/anaconda3/bin/pip install smogn
```

**`Phase4_leaderboard` not found when running step 2** — means step 1 didn't finish. Re-run step 1.

**Want to re-tune from scratch** — delete `slump_full_phase_results.xlsx` first, then re-run step 1. Otherwise step 1 will overwrite it anyway.

**Want to skip Phase 6 SMOGN to save time** — open the notebook, find the cell starting with `# SMOGN on the diverse bases` and skip / comment it out. Cuts ~5 min.

## What to expect from the numbers

The original `Slump_ML_Results_new.xlsx` reported:
- Best test R² ≈ 0.97 (Gradient Boosting / Random Forest), train R² ≈ 0.99, gap ≈ 0.02 — **but this was a single random split, with mix-design leakage just like CS.**

Under honest CV, expect:
- **Random KFold 5×5 test R²:** likely 0.80–0.92 for the top boosting models (still high — slump is more predictable than strength).
- **GroupKFold strict test R²:** likely 0.45–0.65 (the honest number, dropped because near-duplicate mixes can no longer leak).
- **Train/test gap:** likely 0.10–0.25 random, 0.30–0.45 group.

If your numbers are far from these, something's off — re-check that the CS pipeline was finished and `civil_utils.py` was the one that already worked for CS.
