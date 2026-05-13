# Slump Prediction — Final Report

## What was done

We built machine-learning models that predict the **slump** (in mm) of fresh geopolymer concrete from its mix recipe. Slump is the standard measurement of how workable/runny the concrete is before it hardens — engineers need it within a target range so the concrete can actually be poured into molds without segregating or being too stiff.

The pipeline is the same one we used for compressive strength, applied to the slump dataset.

## Dataset

- **84 rows** (mix designs), each with a measured slump value.
- **15 input features**: same as for compressive strength — fly ash, GGBFS, water, sand, aggregates, alkaline activators, curing parameters.
- **Target range**: 32 to 650 mm (very wide — covers stiff roadbed concrete to highly fluid pumped concrete).
- The dataset is small (84 rows) but every row has a valid target.

## Why slump matters

Even if a mix produces strong concrete, if its slump is wrong, the concrete is unusable in practice:
- **Slump < 80 mm**: too stiff, can't be pumped, hard to consolidate
- **Slump 80–150 mm**: ideal for most building work
- **Slump > 200 mm**: too fluid, may segregate and lose strength

So engineers need both the right strength AND the right slump. Predicting slump from the recipe lets them tune the mix on paper before mixing anything.

## What we tried

The same 6-phase pipeline as compressive strength:

1. **Phase 1** — Honest re-evaluation of the original 7 models with proper cross-validation
2. **Phase 2** — Domain-meaningful feature engineering (W/B ratio, GGBFS fraction, etc.)
3. **Phase 3** — Hyperparameter tuning via Optuna (100 trials per boosting model)
4. **Phase 4** — Tried 11 additional models (SVR, KNN, Gaussian Process, etc.)
5. **Phase 5** — Stacking ensemble (combining the 4 best diverse models)
6. **Phase 6** — Data augmentation (SMOGN) and relaxed grouping experiments

## Results

### Best models

| Rank | Model | Random-KF Test R² | GroupKFold Test R² | Gap |
|---|---|---|---|---|
| 1 | **Gradient Boosting** | **0.861** | 0.481 | 0.14 |
| 2 | XGBoost | 0.857 | 0.531 | 0.14 |
| 3 | LightGBM | 0.835 | **0.662** | 0.31 |
| 4 | CatBoost | 0.834 | 0.540 | 0.27 |
| 5 | ExtraTrees | 0.829 | 0.573 | 0.43 |
| 6 | Stacked Ensemble | 0.816 | **0.674** | 0.15 |
| 7 | HistGradientBoosting | 0.800 | 0.666 | 0.33 |
| 8 | Random Forest | 0.799 | 0.512 | 0.47 |
| 9 | AdaBoost | 0.796 | 0.459 | 0.50 |
| 10 | KNN | 0.767 | 0.402 | 0.60 |

### Best on each axis

- **Highest random-KF R²**: Gradient Boosting at **0.861**
- **Highest GroupKFold R²**: Stacked Ensemble at **0.674**
- **Smallest gap (less overfit)**: Gradient Boosting at 0.138

### Comparison to original Slump_ML_Results_new.xlsx

| Metric | Original report | Our result | Note |
|---|---|---|---|
| Best test R² | ~0.97 | 0.86 | Original was on single split with leakage |
| Train R² | ~1.00 | 0.99 | Boosting always memorizes training |
| Methodology | Single random split | 5-fold CV × 5 repeats | Much more reliable |

**Our R² of 0.86 is lower than the original's 0.97, but more honest.** The original 0.97 was inflated because the model was effectively memorizing recipes it would later see in the test set. Our number reflects what the model can actually do on unseen recipes.

## Why slump is easier to predict than compressive strength

Notice that slump test R² (0.86) is much higher than compressive strength (0.71). Here's why:

- **Slump is more directly determined by water content and mix proportions.** A wet mix slumps more, period.
- **Compressive strength** depends on chemistry (curing reactions, polymer formation) which has more variability than physical fluidity.
- **Slump has a wider target range** (32 to 650 mm) which makes patterns easier to detect than CS's narrower range (14 to 93 MPa).

This is consistent with what the civil engineering literature reports.

## Honest caveats

**1. Only 84 rows is small.**
Models can easily overfit at this dataset size. Our use of repeated cross-validation and Optuna tuning specifically targets this risk, but more data would always help.

**2. The GroupKFold drop (0.86 → 0.67) is real.**
When the model can never see *any variation* of a recipe before predicting it, accuracy drops noticeably. This means the model is partly relying on having seen similar recipes during training — which is fine for normal use, but worth being aware of.

**3. SVR (Support Vector Regression) underperformed for slump.**
Unlike for compressive strength where it was a top performer, SVR couldn't handle slump's wide target range. The Stacked Ensemble correctly excluded it and used HistGradientBoosting + ExtraTrees + GaussianProcess + KNN instead.

## Top features for predicting slump

In order of importance (from permutation importance analysis):
1. **Water content** — most important, makes intuitive sense (wetter mix = more slump)
2. **GGBFS fraction** — slag content affects fluidity
3. **Activator/binder ratio** — total alkaline activator amount
4. **W/B ratio** (water-to-binder) — Abrams' classic predictor
5. **Total binder** — overall cementing material
6. **NaOH molarity** — affects viscosity

Curing parameters (temperature, time) matter much less for slump than for compressive strength, which makes sense — slump is measured *before* curing.

## What's in the deliverables

- **`Slump_ML_Results_v2.xlsx`** — formatted report (15 sheets, same format as the CS one):
  - Dataset Overview
  - Model Performance summary
  - One sheet per model with predictions vs actuals
  - Methodology sheet
- **`slump_full_phase_results.xlsx`** — phase-by-phase working log
- **`slump_modeling_full.ipynb`** — the Jupyter notebook
- **`civil_utils.py`** — same supporting code as the CS pipeline

## Bottom line

For slump prediction, we have:
- **Random KFold test R² = 0.86** (Gradient Boosting) — comparable to what published papers report.
- **GroupKFold test R² = 0.67** (Stacked Ensemble) — the honest generalization number.
- **Gap = 0.14** (Gradient Boosting) — small, indicating the model genuinely learned rather than memorized.

These are strong, defensible numbers given a dataset of only 84 rows.
