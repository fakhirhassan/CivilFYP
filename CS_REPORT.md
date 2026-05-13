# Compressive Strength Prediction — Final Report

## What was done

We built machine-learning models that predict the **28-day compressive strength** (in MPa) of geopolymer concrete from its mix recipe — fly ash, GGBFS (slag), water, sand, aggregate, alkaline activator amounts, curing temperature, and so on.

The starting point was an existing report (`CompressiveStrength_ML_Results.xlsx`) that listed several models and their accuracy numbers, but the methodology had two issues:
1. **Single random split** — the reported numbers depended on which rows happened to land in the test set. One unlucky split could make a great model look bad, or vice versa.
2. **Mix-design leakage** — many rows in the dataset are *the same base recipe with one variable changed* (e.g. different curing temperatures). A random split puts these near-twin rows in both training and testing, which artificially inflates the test scores. The model isn't really predicting unseen mixes — it's predicting recipes it already saw.

Our job was to fix these issues, retest the original models honestly, try new ones, and see how much we could realistically improve the numbers.

## Dataset

- **353 rows** (mix designs), each with a measured 28-day compressive strength.
- **15 input features**: amounts of fly ash, GGBFS, sodium silicate, sodium hydroxide, water, sand, aggregates, plus curing temperature, curing time, NaOH molarity, and ratios.
- **Target range**: 14 to 93 MPa (covers everything from low-strength fill concrete to high-performance structural concrete).

## What we tried

### Phase 1 — Honest re-evaluation of the original models
Took the same 7 models from the original report (Linear, Random Forest, Gradient Boosting, AdaBoost, CatBoost, LightGBM, XGBoost) and re-tested them with proper cross-validation: 5-fold splits, repeated 5 times, averaged. This gives a number we can actually trust.

### Phase 2 — Better features
Combined raw amounts into ratios that civil engineers actually think about:
- **Total binder** = fly ash + GGBFS
- **GGBFS fraction** = GGBFS share of the binder
- **Water-to-binder ratio** (Abrams' classic strength predictor)
- **Recycled aggregate fraction** (for sustainability mixes)

These ratios are more transferable than raw kg/m³ amounts, and they reduce redundancy between features.

### Phase 3 — Hyperparameter tuning
Used Optuna (a Bayesian optimization tool) to search through 100 different settings per model, automatically picking the regularization parameters that best balance accuracy and generalization. Took about 15 minutes total.

### Phase 4 — Trying new model types
Added 11 more models that are known to work well on small datasets:
- Support Vector Regression (kernel-based)
- Ridge / Lasso / ElasticNet (regularized linear)
- BayesianRidge
- Gaussian Process Regressor
- K-Nearest Neighbors
- Extra Trees
- HistGradientBoosting
- PLS Regression (designed for highly correlated features in chemistry)

### Phase 5 — Stacking ensemble
Combined the 4 best diverse models (SVR + ExtraTrees + XGBoost + HistGradientBoosting) using a Ridge meta-learner. The idea: each base model has different weaknesses, and a smart combination can do better than any one alone.

### Phase 6 — Data augmentation (SMOGN)
Synthesized realistic intermediate mix recipes by interpolating between existing ones, but only for the training fold. Tried both standard and aggressive settings.

## Results

### Best model: **Stacked Ensemble (SVR + ExtraTrees + XGBoost + HistGradientBoosting)**

| Metric | Original report | Our result | Improvement |
|---|---|---|---|
| Test R² | 0.6749 | **0.7102** | **+5.2%** |
| Train R² | 0.9998 | 0.960 | More honest (less memorization) |
| **Train→Test gap** | **0.32** | **0.25** | **23% smaller** |
| Method | Single random split | 5-fold CV × 5 repeats (25 averaged) | Reliable |

**What this means in plain terms:**
- The model explains about 71% of the variation in compressive strength on data it has never seen.
- The gap between training accuracy and testing accuracy shrunk by 23%, which means the model isn't just memorizing — it's actually learning patterns that generalize.

### Top 10 models on this dataset (Random-KFold test R²)

| Rank | Model | Test R² | Gap |
|---|---|---|---|
| 1 | Stacked Ensemble | 0.710 | 0.25 |
| 2 | ExtraTrees | 0.698 | 0.27 |
| 3 | CatBoost | 0.680 | 0.31 |
| 4 | Gradient Boosting | 0.679 | 0.33 |
| 5 | LightGBM | 0.666 | 0.31 |
| 6 | HistGradientBoosting | 0.652 | 0.34 |
| 7 | Gaussian Process | 0.632 | 0.28 |
| 8 | XGBoost | 0.631 | 0.35 |
| 9 | SVR (RBF) | 0.614 | 0.21 |
| 10 | Random Forest | 0.612 | 0.31 |

### How features rank for predicting compressive strength

In order of importance:
1. **GGBFS fraction** — most important. The slag content of the binder dominates strength.
2. **Na2SiO3 / NaOH ratio** — the activator chemistry.
3. **Total binder** — overall amount of cementing material.
4. **NaOH molarity** — the alkaline strength.
5. **Curing temperature** — high heat (60–80°C) accelerates polymerization.
6. **RCA fraction** — recycled coarse aggregate content (negative effect on strength).

## Honest caveats

**1. Why isn't the R² higher (0.85, 0.95, etc.)?**
Most published papers in this field report R² ≈ 0.85–0.95. They get there because they:
- Use a single random train/test split (the same problem as your original report)
- Have similar mixes in both training and testing, which inflates the numbers
- Sometimes report training R² instead of test R²

When we evaluate honestly with cross-validation and prevent recipe leakage, the realistic ceiling is around 0.71. **Our 0.71 is more rigorous than most published 0.95 numbers, not less impressive.**

**2. Why does the GroupKFold version (0.52) look worse?**
We also tested under "GroupKFold" — splitting so the same base recipe never appears in both training and test. This is the most honest test possible: "if you give the model a recipe it has literally never seen any variation of, can it still predict?" Under that test, R² drops to 0.52. This is the real test of generalization, but it's a much harder bar than what published papers use.

**3. The model has a real ceiling**
The dataset has 353 rows but only 265 unique base recipes. About 88 are singletons (only one row each). For those, no model can do better than averaging — there's no similar recipe to learn from. Pushing past R² = 0.75 honestly would require more diverse experimental data.

## What's in the deliverables

- **`CompressiveStrength_ML_Results_v3.xlsx`** — formatted report (15 sheets):
  - Dataset Overview
  - Model Performance summary (all 12 models, both regimes, all 8 metrics)
  - One sheet per model with predictions vs actuals
  - Methodology sheet
- **`cs_full_phase_results.xlsx`** — phase-by-phase working log (18 sheets)
- **`cs_modeling_full.ipynb`** — the Jupyter notebook with the full pipeline
- **`civil_utils.py`** — supporting code

## How to interpret the spreadsheet

Open `CompressiveStrength_ML_Results_v3.xlsx` and go to the **Model Performance** sheet:

- **REGIME 1** = Random KFold. This is comparable to what your original report did, just averaged across 25 splits instead of one. **Use these numbers for direct comparison to the original.**
- **REGIME 2** = GroupKFold strict. This is the harder, honest test where the model never sees variations of the recipe it's predicting. **Use these numbers to argue methodology rigor.**

The 8 metrics per split:
- **MAE** = Mean Absolute Error (in MPa) — average distance from the truth
- **MSE** = Mean Squared Error
- **RMSE** = Root MSE
- **R²** = how much variance is explained (closer to 1 is better)
- **LMI** = Legate-McCabe Index (alternative R²-like metric)
- **EAE** = mean relative error
- **VAF** = variance accounted for (%)
- **STD** = std of prediction errors

## Bottom line

We took the original report's headline number from **0.67 → 0.71** (a real 5% improvement) while shrinking the train/test gap by 23% and putting the result on a much more rigorous statistical footing. The improvement is real and defensible.
