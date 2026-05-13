"""Shared helpers for the CS / Slump modeling notebooks.

Functions here handle loading the messy source xlsx files, cleaning column
names, target imputation/dropping policy, group-id detection (so near-duplicate
mix designs don't leak across folds), and a unified cross-validated evaluation
loop reporting Train/Val/Test R^2 mean +/- std for each model.
"""
from __future__ import annotations

import re
import warnings
from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold

DATA_DIR = "/Users/fakhirhassan/Desktop/CivilFYP"

CS_FILE = f"{DATA_DIR}/compressive strength 28 days complete.xlsx"
CS_FULL_FILE = f"{DATA_DIR}/compressive strength 28 days done complete .xlsx"
SLUMP_FILE = f"{DATA_DIR}/slump data complete s.xlsx"

CS_TARGET = "compressive_strength_mpa"
SLUMP_TARGET = "slump_mm"


def _clean_col(c: str) -> str:
    c = str(c).replace("\n", " ").strip()
    c = re.sub(r"\s+", " ", c)
    c = c.lower()
    c = re.sub(r"[°/()]", " ", c)
    c = re.sub(r"[^a-z0-9]+", "_", c)
    c = re.sub(r"_+", "_", c).strip("_")
    return c


_RENAME = {
    "flyash_kg_m3": "flyash",
    "ggbfs_kg_m3": "ggbfs",
    "curing_temperature_c": "curing_temp_c",
    "curing_time_hr": "curing_time_hr",
    "na2sio3_kg_m3": "na2sio3",
    "naoh_kg_m3": "naoh",
    "na2sio3_naoh": "na2sio3_naoh_ratio",
    "molarity_of_naoh": "naoh_molarity",
    "alkaline_activator_binder_ratio": "activator_binder_ratio",
    "water": "water",
    "sp": "sp",
    "nca_kg_m3": "nca",
    "rca_kg_m3": "rca",
    "sand_kg_m3": "sand",
    "r_sand_kg_m3": "r_sand",
    "compressive_strength_mpa": "compressive_strength_mpa",
    "slump": "slump_mm",
}


def load_clean(file: str, target_col_name: str) -> pd.DataFrame:
    df = pd.read_excel(file)
    df.columns = [_clean_col(c) for c in df.columns]
    df = df.rename(columns=_RENAME)
    target = "compressive_strength_mpa" if target_col_name == CS_TARGET else "slump_mm"
    df[target] = pd.to_numeric(df[target], errors="coerce")
    df = df.dropna(axis=0, how="all")
    return df


def cs_dataset(drop_na_target: bool = True) -> pd.DataFrame:
    df = load_clean(CS_FILE, CS_TARGET)
    if drop_na_target:
        df = df.dropna(subset=[CS_TARGET]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != CS_TARGET]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def cs_full_dataset(drop_na_target: bool = True) -> pd.DataFrame:
    """Full 353-row CS dataset (the file the original report used)."""
    df = load_clean(CS_FULL_FILE, CS_TARGET)
    if drop_na_target:
        df = df.dropna(subset=[CS_TARGET]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != CS_TARGET]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def slump_dataset() -> pd.DataFrame:
    df = load_clean(SLUMP_FILE, SLUMP_TARGET)
    df = df.dropna(subset=[SLUMP_TARGET]).reset_index(drop=True)
    feature_cols = [c for c in df.columns if c != SLUMP_TARGET]
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def feature_target_split(df: pd.DataFrame, target: str):
    X = df.drop(columns=[target]).copy()
    y = df[target].astype(float).values
    return X, y


def assign_group_ids(
    X: pd.DataFrame, key_cols: Iterable[str] | None = None
) -> np.ndarray:
    """Group rows that share the same base mix design.

    Default keys = the binder + activator columns (flyash, ggbfs, na2sio3,
    naoh, naoh_molarity, activator_binder_ratio). Rows that match on these
    are treated as the same 'mix' even if curing/aggregate vary.
    """
    if key_cols is None:
        key_cols = [
            "flyash",
            "ggbfs",
            "na2sio3",
            "naoh",
            "naoh_molarity",
            "activator_binder_ratio",
        ]
        key_cols = [c for c in key_cols if c in X.columns]
    keys = X[list(key_cols)].astype(float).round(3).astype(str).agg("|".join, axis=1)
    return pd.factorize(keys)[0]


@dataclass
class FoldScores:
    train_r2: list
    val_r2: list
    test_r2: list
    train_mae: list
    val_mae: list
    test_mae: list
    train_rmse: list
    val_rmse: list
    test_rmse: list

    def summary_row(self, name: str) -> dict:
        def ms(vals):
            arr = np.asarray(vals, dtype=float)
            return arr.mean(), arr.std()

        out = {"model": name}
        for split in ("train", "val", "test"):
            for metric in ("r2", "mae", "rmse"):
                m, s = ms(getattr(self, f"{split}_{metric}"))
                out[f"{split}_{metric}_mean"] = m
                out[f"{split}_{metric}_std"] = s
        out["overfit_gap"] = (
            np.mean(self.train_r2) - np.mean(self.test_r2)
        )
        return out


def cv_evaluate(
    model_factory,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    val_frac_of_train: float = 0.2,
    random_state: int = 0,
    scaler_factory=None,
) -> FoldScores:
    """Repeated K-Fold CV that produces Train/Val/Test R^2 per fold.

    For each outer fold:
      * fold rows = TEST
      * remaining rows are split (val_frac_of_train) into TRAIN and VAL
      * model is fit on TRAIN, scored on TRAIN/VAL/TEST

    If `groups` is provided, the outer split uses GroupKFold (no repeats —
    GroupKFold is deterministic). The TRAIN/VAL split inside also respects
    groups.
    """
    rng = np.random.default_rng(random_state)
    fs = FoldScores(*[[] for _ in range(9)])

    if groups is not None:
        outer = GroupKFold(n_splits=n_splits)
        outer_splits = list(outer.split(X, y, groups))
        repeats = 1
    else:
        repeats = n_repeats

    for repeat in range(repeats):
        if groups is None:
            outer = KFold(
                n_splits=n_splits, shuffle=True, random_state=random_state + repeat
            )
            outer_splits = list(outer.split(X, y))

        for trainval_idx, test_idx in outer_splits:
            if groups is not None:
                inner_groups = groups[trainval_idx]
                # split unique groups into train/val
                unique = np.unique(inner_groups)
                rng.shuffle(unique)
                n_val_groups = max(1, int(round(len(unique) * val_frac_of_train)))
                val_groups = set(unique[:n_val_groups].tolist())
                val_mask = np.isin(inner_groups, list(val_groups))
                val_idx = trainval_idx[val_mask]
                train_idx = trainval_idx[~val_mask]
            else:
                shuffled = rng.permutation(trainval_idx)
                n_val = max(1, int(round(len(shuffled) * val_frac_of_train)))
                val_idx = shuffled[:n_val]
                train_idx = shuffled[n_val:]

            X_tr = X.iloc[train_idx]
            X_va = X.iloc[val_idx]
            X_te = X.iloc[test_idx]
            y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]

            if scaler_factory is not None:
                scaler = scaler_factory()
                X_tr_arr = scaler.fit_transform(X_tr)
                X_va_arr = scaler.transform(X_va)
                X_te_arr = scaler.transform(X_te)
            else:
                X_tr_arr, X_va_arr, X_te_arr = X_tr.values, X_va.values, X_te.values

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = model_factory()
                model.fit(X_tr_arr, y_tr)
                p_tr = model.predict(X_tr_arr)
                p_va = model.predict(X_va_arr)
                p_te = model.predict(X_te_arr)

            fs.train_r2.append(r2_score(y_tr, p_tr))
            fs.val_r2.append(r2_score(y_va, p_va))
            fs.test_r2.append(r2_score(y_te, p_te))
            fs.train_mae.append(mean_absolute_error(y_tr, p_tr))
            fs.val_mae.append(mean_absolute_error(y_va, p_va))
            fs.test_mae.append(mean_absolute_error(y_te, p_te))
            fs.train_rmse.append(np.sqrt(mean_squared_error(y_tr, p_tr)))
            fs.val_rmse.append(np.sqrt(mean_squared_error(y_va, p_va)))
            fs.test_rmse.append(np.sqrt(mean_squared_error(y_te, p_te)))

    return fs


def default_model_zoo() -> dict:
    """Return the 7 baselines from the existing results spreadsheet, with the
    same defaults so numbers are comparable."""
    from sklearn.linear_model import LinearRegression
    from sklearn.ensemble import (
        AdaBoostRegressor,
        GradientBoostingRegressor,
        RandomForestRegressor,
    )

    zoo = {
        "Linear Regression": lambda: LinearRegression(),
        "Random Forest": lambda: RandomForestRegressor(random_state=42),
        "Gradient Boosting": lambda: GradientBoostingRegressor(random_state=42),
        "AdaBoost": lambda: AdaBoostRegressor(random_state=42),
    }

    try:
        from catboost import CatBoostRegressor

        zoo["CatBoost"] = lambda: CatBoostRegressor(verbose=0, random_seed=42)
    except Exception:
        pass

    try:
        from lightgbm import LGBMRegressor

        zoo["LightGBM"] = lambda: LGBMRegressor(random_state=42, verbosity=-1)
    except Exception:
        pass

    try:
        from xgboost import XGBRegressor

        zoo["XGBoost"] = lambda: XGBRegressor(
            random_state=42, verbosity=0, n_jobs=1
        )
    except Exception:
        pass

    return zoo


RAW_FEATURE_COLS = [
    "flyash",
    "ggbfs",
    "curing_temp_c",
    "curing_time_hr",
    "na2sio3",
    "naoh",
    "na2sio3_naoh_ratio",
    "naoh_molarity",
    "activator_binder_ratio",
    "water",
    "sp",
    "nca",
    "rca",
    "sand",
    "r_sand",
]


def engineer_features(
    df: pd.DataFrame,
    mode: str = "engineered",
    target: str | None = None,
    eps: float = 1e-9,
) -> pd.DataFrame:
    """Return a feature matrix with one of three encodings.

    mode='raw'           -> original 15 columns unchanged
    mode='engineered'    -> drop colinear raw amounts; add ratios/totals
    mode='both'          -> raw + engineered side by side
    """
    df = df.copy()

    fly = df["flyash"].astype(float)
    ggb = df["ggbfs"].astype(float)
    na2 = df["na2sio3"].astype(float)
    naoh = df["naoh"].astype(float)
    water = df["water"].astype(float)
    nca = df["nca"].astype(float)
    rca = df["rca"].astype(float)
    sand = df["sand"].astype(float)
    rsand = df["r_sand"].astype(float)

    df["total_binder"] = fly + ggb
    df["ggbfs_fraction"] = ggb / (df["total_binder"] + eps)
    df["total_activator"] = na2 + naoh
    df["w_b_ratio"] = water / (df["total_binder"] + eps)
    df["total_aggregate"] = nca + rca + sand + rsand
    df["rca_fraction"] = rca / (df["total_aggregate"] + eps)
    df["r_sand_fraction"] = rsand / (df["total_aggregate"] + eps)
    # nca and sand fractions stand in for the natural-aggregate share
    df["nca_fraction"] = nca / (df["total_aggregate"] + eps)
    df["sand_fraction"] = sand / (df["total_aggregate"] + eps)

    engineered_cols = [
        "total_binder",
        "ggbfs_fraction",
        "total_activator",
        "na2sio3_naoh_ratio",
        "naoh_molarity",
        "activator_binder_ratio",
        "w_b_ratio",
        "rca_fraction",
        "r_sand_fraction",
        "curing_temp_c",
        "curing_time_hr",
        "sp",
    ]

    raw_cols = [c for c in RAW_FEATURE_COLS if c in df.columns]

    if mode == "raw":
        keep = raw_cols
    elif mode == "engineered":
        keep = engineered_cols
    elif mode == "both":
        keep = list(dict.fromkeys(raw_cols + engineered_cols))
    else:
        raise ValueError(f"unknown mode {mode!r}")

    if target is not None and target in df.columns:
        keep = [c for c in keep if c != target]
        out = df[keep + [target]]
    else:
        out = df[keep]
    return out.copy()


def vif_table(X: pd.DataFrame) -> pd.DataFrame:
    """Variance inflation factor per feature. >10 = serious multicollinearity."""
    from sklearn.linear_model import LinearRegression

    rows = []
    cols = list(X.columns)
    Xv = X.values.astype(float)
    for i, col in enumerate(cols):
        y_i = Xv[:, i]
        X_other = np.delete(Xv, i, axis=1)
        model = LinearRegression().fit(X_other, y_i)
        r2 = model.score(X_other, y_i)
        vif = float("inf") if r2 >= 1 - 1e-12 else 1.0 / (1.0 - r2)
        rows.append({"feature": col, "VIF": vif, "R2_other": r2})
    return pd.DataFrame(rows).sort_values("VIF", ascending=False).reset_index(drop=True)


def _gkfold_val_r2(factory, X: pd.DataFrame, y: np.ndarray,
                   groups: np.ndarray, n_splits: int = 5,
                   val_frac_of_train: float = 0.2,
                   random_state: int = 0) -> float:
    """Mean val R^2 over GroupKFold outer folds with an inner group-aware
    train/val split. Used as the Optuna objective.
    """
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(random_state)
    val_scores = []
    gkf = GroupKFold(n_splits=n_splits)
    for trainval_idx, _test_idx in gkf.split(X, y, groups):
        inner_groups = groups[trainval_idx]
        unique = np.unique(inner_groups)
        rng.shuffle(unique)
        n_val_groups = max(1, int(round(len(unique) * val_frac_of_train)))
        val_groups = set(unique[:n_val_groups].tolist())
        val_mask = np.isin(inner_groups, list(val_groups))
        val_idx = trainval_idx[val_mask]
        train_idx = trainval_idx[~val_mask]
        if len(train_idx) < 5 or len(val_idx) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = factory()
            model.fit(X.iloc[train_idx].values, y[train_idx])
            p_va = model.predict(X.iloc[val_idx].values)
        val_scores.append(r2_score(y[val_idx], p_va))
    return float(np.mean(val_scores))


def _suggest_rf(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 12),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def _suggest_gb(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 800),
        "max_depth": trial.suggest_int("max_depth", 2, 6),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
    }


def _suggest_ada(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.05, 1.5, log=True),
        "loss": trial.suggest_categorical("loss", ["linear", "square", "exponential"]),
    }


def _suggest_cat(trial):
    return {
        "iterations": trial.suggest_int("iterations", 200, 1500),
        "depth": trial.suggest_int("depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 1, 12),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "rsm": trial.suggest_float("rsm", 0.5, 1.0),
    }


def _suggest_lgbm(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 4, 64),
        "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 2, 12),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
    }


def _suggest_xgb(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1500),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
    }


def _factory_rf(p):
    from sklearn.ensemble import RandomForestRegressor
    return RandomForestRegressor(random_state=42, n_jobs=1, **p)


def _factory_gb(p):
    from sklearn.ensemble import GradientBoostingRegressor
    return GradientBoostingRegressor(random_state=42, **p)


def _factory_ada(p):
    from sklearn.ensemble import AdaBoostRegressor
    return AdaBoostRegressor(random_state=42, **p)


def _factory_cat(p):
    from catboost import CatBoostRegressor
    return CatBoostRegressor(verbose=0, random_seed=42, allow_writing_files=False, **p)


def _factory_lgbm(p):
    from lightgbm import LGBMRegressor
    return LGBMRegressor(random_state=42, verbosity=-1, n_jobs=1, **p)


def _factory_xgb(p):
    from xgboost import XGBRegressor
    return XGBRegressor(random_state=42, verbosity=0, n_jobs=1,
                        tree_method="hist", **p)


TUNE_SPECS = {
    "Random Forest": (_suggest_rf, _factory_rf),
    "Gradient Boosting": (_suggest_gb, _factory_gb),
    "AdaBoost": (_suggest_ada, _factory_ada),
    "CatBoost": (_suggest_cat, _factory_cat),
    "LightGBM": (_suggest_lgbm, _factory_lgbm),
    "XGBoost": (_suggest_xgb, _factory_xgb),
}


def _wrap_scaled(estimator):
    """Wrap an estimator in StandardScaler -> estimator pipeline."""
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    return Pipeline([("scaler", StandardScaler()), ("est", estimator)])


def _suggest_ridge(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-3, 100.0, log=True)}


def _factory_ridge(p):
    from sklearn.linear_model import Ridge
    return _wrap_scaled(Ridge(random_state=42, **p))


def _suggest_lasso(trial):
    return {"alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True)}


def _factory_lasso(p):
    from sklearn.linear_model import Lasso
    return _wrap_scaled(Lasso(random_state=42, max_iter=20000, **p))


def _suggest_enet(trial):
    return {
        "alpha": trial.suggest_float("alpha", 1e-3, 10.0, log=True),
        "l1_ratio": trial.suggest_float("l1_ratio", 0.05, 0.95),
    }


def _factory_enet(p):
    from sklearn.linear_model import ElasticNet
    return _wrap_scaled(ElasticNet(random_state=42, max_iter=20000, **p))


def _suggest_ridge_poly(trial):
    return {
        "degree": trial.suggest_int("degree", 2, 3),
        "alpha": trial.suggest_float("alpha", 1e-2, 100.0, log=True),
        "interaction_only": trial.suggest_categorical("interaction_only", [False, True]),
    }


def _factory_ridge_poly(p):
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler

    return Pipeline([
        ("scaler", StandardScaler()),
        ("poly", PolynomialFeatures(
            degree=p["degree"],
            interaction_only=p["interaction_only"],
            include_bias=False,
        )),
        ("est", Ridge(alpha=p["alpha"], random_state=42)),
    ])


def _suggest_bayesian(trial):
    return {
        "alpha_1": trial.suggest_float("alpha_1", 1e-7, 1e-3, log=True),
        "alpha_2": trial.suggest_float("alpha_2", 1e-7, 1e-3, log=True),
        "lambda_1": trial.suggest_float("lambda_1", 1e-7, 1e-3, log=True),
        "lambda_2": trial.suggest_float("lambda_2", 1e-7, 1e-3, log=True),
    }


def _factory_bayesian(p):
    from sklearn.linear_model import BayesianRidge
    return _wrap_scaled(BayesianRidge(**p))


def _suggest_gp(trial):
    return {
        "length_scale": trial.suggest_float("length_scale", 0.1, 20.0, log=True),
        "noise_level": trial.suggest_float("noise_level", 1e-3, 5.0, log=True),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
    }


def _factory_gp(p):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel as C

    kernel = C(1.0, (1e-3, 1e3)) * RBF(
        length_scale=p["length_scale"], length_scale_bounds=(1e-2, 1e3)
    ) + WhiteKernel(noise_level=p["noise_level"], noise_level_bounds=(1e-5, 1e1))
    gp = GaussianProcessRegressor(
        kernel=kernel,
        alpha=p["alpha"],
        normalize_y=True,
        random_state=42,
        n_restarts_optimizer=0,
    )
    return _wrap_scaled(gp)


def _suggest_svr(trial):
    return {
        "C": trial.suggest_float("C", 0.1, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 1e-3, 1.0, log=True),
        "epsilon": trial.suggest_float("epsilon", 1e-3, 1.0, log=True),
    }


def _factory_svr(p):
    from sklearn.svm import SVR
    return _wrap_scaled(SVR(kernel="rbf", **p))


def _suggest_knn(trial):
    return {
        "n_neighbors": trial.suggest_int("n_neighbors", 1, 15),
        "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
        "p": trial.suggest_categorical("p", [1, 2]),
    }


def _factory_knn(p):
    from sklearn.neighbors import KNeighborsRegressor
    return _wrap_scaled(KNeighborsRegressor(**p))


def _suggest_extra(trial):
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 2, 14),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 12),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
    }


def _factory_extra(p):
    from sklearn.ensemble import ExtraTreesRegressor
    return ExtraTreesRegressor(random_state=42, n_jobs=1, **p)


def _suggest_hgb(trial):
    return {
        "max_iter": trial.suggest_int("max_iter", 100, 400),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
        "max_depth": trial.suggest_int("max_depth", 2, 8),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 25),
        "l2_regularization": trial.suggest_float("l2_regularization", 1e-4, 10.0, log=True),
        "max_leaf_nodes": trial.suggest_int("max_leaf_nodes", 8, 31),
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 20,
    }


def _factory_hgb(p):
    from sklearn.ensemble import HistGradientBoostingRegressor
    return HistGradientBoostingRegressor(random_state=42, **p)


def _suggest_pls(trial):
    return {"n_components": trial.suggest_int("n_components", 1, 8)}


def _factory_pls(p):
    from sklearn.cross_decomposition import PLSRegression

    class _PLSWrap:
        """PLSRegression returns 2D predictions for 1D y; flatten."""
        def __init__(self, **kw):
            self.kw = kw
            self.m = PLSRegression(**kw)
        def fit(self, X, y):
            self.m.fit(X, y); return self
        def predict(self, X):
            p = self.m.predict(X)
            return p.ravel() if p.ndim > 1 else p
        def get_params(self, deep=True):
            return dict(self.kw)
        def set_params(self, **kw):
            self.kw.update(kw); self.m = PLSRegression(**self.kw); return self

    return _wrap_scaled(_PLSWrap(**p))


def _suggest_mlp(trial):
    n_layers = trial.suggest_int("n_layers", 1, 3)
    layers = []
    for i in range(n_layers):
        layers.append(trial.suggest_int(f"units_l{i}", 8, 128, log=True))
    return {
        "hidden_layer_sizes": tuple(layers),
        "activation": trial.suggest_categorical("activation", ["relu", "tanh"]),
        "alpha": trial.suggest_float("alpha", 1e-6, 1e-1, log=True),
        "learning_rate_init": trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True),
        "batch_size": trial.suggest_categorical("batch_size", [8, 16, 32]),
        "solver": "adam",
        "max_iter": 1500,
        "early_stopping": True,
        "validation_fraction": 0.15,
        "n_iter_no_change": 25,
    }


def _factory_mlp(p):
    from sklearn.neural_network import MLPRegressor
    params = dict(p)
    # If we got raw Optuna trial.params (with n_layers + units_lN), reconstruct
    # hidden_layer_sizes from them. Otherwise trust the caller-supplied tuple.
    if "hidden_layer_sizes" not in params:
        n_layers = params.pop("n_layers", 1)
        layers = tuple(params.pop(f"units_l{i}") for i in range(n_layers))
        params["hidden_layer_sizes"] = layers
    else:
        params.pop("n_layers", None)
        for k in list(params.keys()):
            if k.startswith("units_l"):
                params.pop(k)
    return _wrap_scaled(MLPRegressor(random_state=42, **params))


class _SeedEnsembleRegressor:
    """Average predictions across N MLPs trained with different seeds.
    Helps push small-data R^2 up by reducing initialization variance."""
    def __init__(self, params, n_seeds=10):
        self.params = dict(params)
        self.n_seeds = n_seeds
        self.models_ = []

    def fit(self, X, y):
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        # Reconstruct hidden_layer_sizes from raw Optuna trial params
        params = dict(self.params)
        if "hidden_layer_sizes" not in params:
            n_layers = params.pop("n_layers", 1)
            layers = tuple(params.pop(f"units_l{i}") for i in range(n_layers))
            params["hidden_layer_sizes"] = layers
        else:
            params.pop("n_layers", None)
            for k in list(params.keys()):
                if k.startswith("units_l"):
                    params.pop(k)
        self.models_ = []
        for s in range(self.n_seeds):
            m = Pipeline([
                ("scaler", StandardScaler()),
                ("est", MLPRegressor(random_state=s, **params)),
            ])
            m.fit(X, y)
            self.models_.append(m)
        return self

    def predict(self, X):
        preds = np.stack([m.predict(X) for m in self.models_], axis=0)
        return preds.mean(axis=0)


def make_seed_ensemble_factory(params, n_seeds=10):
    def _f():
        return _SeedEnsembleRegressor(params, n_seeds=n_seeds)
    return _f


SMALL_DATA_SPECS = {
    "Ridge": (_suggest_ridge, _factory_ridge),
    "Lasso": (_suggest_lasso, _factory_lasso),
    "ElasticNet": (_suggest_enet, _factory_enet),
    "Ridge+Poly": (_suggest_ridge_poly, _factory_ridge_poly),
    "BayesianRidge": (_suggest_bayesian, _factory_bayesian),
    "GaussianProcess": (_suggest_gp, _factory_gp),
    "SVR_RBF": (_suggest_svr, _factory_svr),
    "KNN": (_suggest_knn, _factory_knn),
    "ExtraTrees": (_suggest_extra, _factory_extra),
    "HistGradientBoosting": (_suggest_hgb, _factory_hgb),
    "PLS": (_suggest_pls, _factory_pls),
    "MLP": (_suggest_mlp, _factory_mlp),
}

# Register them so tune_model() can look them up alongside the GBM specs.
TUNE_SPECS.update(SMALL_DATA_SPECS)


def stacking_cv_evaluate(
    base_factories: dict,
    meta_factory,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    val_frac_of_train: float = 0.2,
    inner_splits: int = 5,
    random_state: int = 0,
) -> "FoldScores":
    """Out-of-fold stacking evaluator.

    For each outer fold:
      * fold rows = TEST
      * remaining rows split into TRAIN+VAL (group-aware if groups given)
      * generate out-of-fold base predictions on TRAIN via inner KFold/GroupKFold
      * fit each base model on full TRAIN, predict VAL and TEST
      * fit meta_factory on the OOF base predictions and TRAIN target
      * score meta on TRAIN (in-sample), VAL, TEST

    Returns FoldScores compatible with summary_row().
    """
    rng = np.random.default_rng(random_state)
    fs = FoldScores(*[[] for _ in range(9)])

    if groups is not None:
        outer = GroupKFold(n_splits=n_splits)
        outer_splits = list(outer.split(X, y, groups))
        repeats = 1
    else:
        repeats = n_repeats

    for repeat in range(repeats):
        if groups is None:
            outer = KFold(
                n_splits=n_splits, shuffle=True, random_state=random_state + repeat
            )
            outer_splits = list(outer.split(X, y))

        for trainval_idx, test_idx in outer_splits:
            if groups is not None:
                inner_groups = groups[trainval_idx]
                unique = np.unique(inner_groups)
                rng.shuffle(unique)
                n_val_groups = max(1, int(round(len(unique) * val_frac_of_train)))
                val_groups = set(unique[:n_val_groups].tolist())
                val_mask = np.isin(inner_groups, list(val_groups))
                val_idx = trainval_idx[val_mask]
                train_idx = trainval_idx[~val_mask]
            else:
                shuffled = rng.permutation(trainval_idx)
                n_val = max(1, int(round(len(shuffled) * val_frac_of_train)))
                val_idx = shuffled[:n_val]
                train_idx = shuffled[n_val:]

            X_tr = X.iloc[train_idx].values
            X_va = X.iloc[val_idx].values
            X_te = X.iloc[test_idx].values
            y_tr, y_va, y_te = y[train_idx], y[val_idx], y[test_idx]

            # OOF predictions on TRAIN per base model
            n_tr = len(train_idx)
            base_names = list(base_factories.keys())
            oof_preds = np.zeros((n_tr, len(base_names)))
            val_preds = np.zeros((len(val_idx), len(base_names)))
            test_preds = np.zeros((len(test_idx), len(base_names)))

            if groups is not None:
                tr_groups = groups[train_idx]
                inner_kf = GroupKFold(
                    n_splits=min(inner_splits, len(np.unique(tr_groups)))
                )
                inner_iter = list(inner_kf.split(X_tr, y_tr, tr_groups))
            else:
                inner_kf = KFold(
                    n_splits=inner_splits, shuffle=True, random_state=random_state
                )
                inner_iter = list(inner_kf.split(X_tr, y_tr))

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                for j, name in enumerate(base_names):
                    factory = base_factories[name]
                    # OOF on train
                    for inner_tr, inner_te in inner_iter:
                        m = factory()
                        m.fit(X_tr[inner_tr], y_tr[inner_tr])
                        oof_preds[inner_te, j] = m.predict(X_tr[inner_te])
                    # Refit on full train, predict val/test
                    m_full = factory()
                    m_full.fit(X_tr, y_tr)
                    val_preds[:, j] = m_full.predict(X_va)
                    test_preds[:, j] = m_full.predict(X_te)

                # Train meta on OOF predictions
                meta = meta_factory()
                meta.fit(oof_preds, y_tr)
                # In-sample meta prediction on TRAIN uses *refit-on-full-train*
                # base predictions to mirror serving time, then meta.predict
                full_train_preds = np.zeros((n_tr, len(base_names)))
                for j, name in enumerate(base_names):
                    m_full = base_factories[name]()
                    m_full.fit(X_tr, y_tr)
                    full_train_preds[:, j] = m_full.predict(X_tr)
                p_tr = meta.predict(full_train_preds)
                p_va = meta.predict(val_preds)
                p_te = meta.predict(test_preds)

            fs.train_r2.append(r2_score(y_tr, p_tr))
            fs.val_r2.append(r2_score(y_va, p_va))
            fs.test_r2.append(r2_score(y_te, p_te))
            fs.train_mae.append(mean_absolute_error(y_tr, p_tr))
            fs.val_mae.append(mean_absolute_error(y_va, p_va))
            fs.test_mae.append(mean_absolute_error(y_te, p_te))
            fs.train_rmse.append(np.sqrt(mean_squared_error(y_tr, p_tr)))
            fs.val_rmse.append(np.sqrt(mean_squared_error(y_va, p_va)))
            fs.test_rmse.append(np.sqrt(mean_squared_error(y_te, p_te)))

    return fs


def make_tuned_factory(model_name: str, params: dict):
    """Convenience: turn a (name, best_params) into a zero-arg factory."""
    _, factory_fn = TUNE_SPECS[model_name]
    return lambda p=params, fn=factory_fn: fn(p)


def assign_group_ids_relaxed(X: pd.DataFrame) -> np.ndarray:
    """Looser grouping: only key on flyash, ggbfs, and na2sio3_naoh_ratio.

    Rationale: a row with the same binder + activator chemistry but different
    curing temperature is arguably a different physical experiment, so we
    don't lump them together. Should produce more groups -> lower variance
    on GroupKFold.
    """
    keys = []
    for col in ("flyash", "ggbfs", "na2sio3_naoh_ratio"):
        if col in X.columns:
            keys.append(col)
    if not keys:
        return np.arange(len(X))
    s = X[keys].astype(float).round(2).astype(str).agg("|".join, axis=1)
    return pd.factorize(s)[0]


def cv_evaluate_with_smogn(
    factory_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    n_repeats: int = 5,
    val_frac_of_train: float = 0.2,
    random_state: int = 0,
    smogn_kwargs: dict | None = None,
) -> "FoldScores":
    """Same as cv_evaluate, but augments the TRAIN fold with SMOGN samples
    before fitting. Test/val are never augmented.

    Falls back gracefully if smogn isn't installed or the fold is too small.
    """
    try:
        import smogn  # noqa: F401
    except Exception:
        raise RuntimeError("smogn is not installed; pip install smogn")

    rng = np.random.default_rng(random_state)
    fs = FoldScores(*[[] for _ in range(9)])
    smogn_kwargs = smogn_kwargs or {}

    if groups is not None:
        outer = GroupKFold(n_splits=n_splits)
        outer_splits = list(outer.split(X, y, groups))
        repeats = 1
    else:
        repeats = n_repeats

    for repeat in range(repeats):
        if groups is None:
            outer = KFold(
                n_splits=n_splits, shuffle=True, random_state=random_state + repeat
            )
            outer_splits = list(outer.split(X, y))

        for trainval_idx, test_idx in outer_splits:
            if groups is not None:
                inner_groups = groups[trainval_idx]
                unique = np.unique(inner_groups)
                rng.shuffle(unique)
                n_val_groups = max(1, int(round(len(unique) * val_frac_of_train)))
                val_groups = set(unique[:n_val_groups].tolist())
                val_mask = np.isin(inner_groups, list(val_groups))
                val_idx = trainval_idx[val_mask]
                train_idx = trainval_idx[~val_mask]
            else:
                shuffled = rng.permutation(trainval_idx)
                n_val = max(1, int(round(len(shuffled) * val_frac_of_train)))
                val_idx = shuffled[:n_val]
                train_idx = shuffled[n_val:]

            target_col = "__y__"
            train_df = X.iloc[train_idx].copy()
            train_df[target_col] = y[train_idx]

            try:
                aug = _smogn_safe(train_df, target_col, smogn_kwargs)
            except Exception:
                aug = train_df  # fall back to original if SMOGN fails

            X_tr_aug = aug.drop(columns=[target_col]).values
            y_tr_aug = aug[target_col].values
            X_va = X.iloc[val_idx].values
            X_te = X.iloc[test_idx].values
            y_va, y_te = y[val_idx], y[test_idx]

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                model = factory_fn()
                model.fit(X_tr_aug, y_tr_aug)
                p_tr_real = model.predict(X.iloc[train_idx].values)
                p_va = model.predict(X_va)
                p_te = model.predict(X_te)

            fs.train_r2.append(r2_score(y[train_idx], p_tr_real))
            fs.val_r2.append(r2_score(y_va, p_va))
            fs.test_r2.append(r2_score(y_te, p_te))
            fs.train_mae.append(mean_absolute_error(y[train_idx], p_tr_real))
            fs.val_mae.append(mean_absolute_error(y_va, p_va))
            fs.test_mae.append(mean_absolute_error(y_te, p_te))
            fs.train_rmse.append(np.sqrt(mean_squared_error(y[train_idx], p_tr_real)))
            fs.val_rmse.append(np.sqrt(mean_squared_error(y_va, p_va)))
            fs.test_rmse.append(np.sqrt(mean_squared_error(y_te, p_te)))

    return fs


def _smogn_safe(df: pd.DataFrame, target: str, kwargs: dict) -> pd.DataFrame:
    """Wrapper that retries SMOGN with safer settings on failure."""
    import smogn

    defaults = dict(
        y=target,
        k=5,
        pert=0.02,
        samp_method="balance",
        rel_thres=0.5,
        rel_method="auto",
        rel_xtrm_type="both",
        rel_coef=1.5,
    )
    defaults.update(kwargs or {})
    try:
        out = smogn.smoter(data=df.copy(), **defaults)
        if out is None or len(out) == 0:
            return df
        return out
    except Exception:
        return df


def tune_model(name: str, X: pd.DataFrame, y: np.ndarray,
               groups: np.ndarray, n_trials: int = 100,
               n_splits: int = 5, random_state: int = 0):
    """Run Optuna to maximise GroupKFold val R^2; return (best_params,
    best_value, study)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    suggest, factory_fn = TUNE_SPECS[name]

    def objective(trial):
        params = suggest(trial)
        return _gkfold_val_r2(lambda: factory_fn(params), X, y, groups,
                              n_splits=n_splits, random_state=random_state)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(Exception,))
    return study.best_params, study.best_value, study


def _kfold_val_r2(factory, X: pd.DataFrame, y: np.ndarray,
                  n_splits: int = 5, n_repeats: int = 3,
                  random_state: int = 0) -> float:
    """Mean test R^2 over repeated random KFold. Used as the Optuna
    objective when tuning for the *report* champion (no group leakage
    guard)."""
    from sklearn.model_selection import KFold
    scores = []
    for rep in range(n_repeats):
        kf = KFold(n_splits=n_splits, shuffle=True,
                   random_state=random_state + rep)
        for tr, te in kf.split(X):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = factory()
                m.fit(X.iloc[tr].values, y[tr])
                p = m.predict(X.iloc[te].values)
            scores.append(r2_score(y[te], p))
    return float(np.mean(scores))


def tune_model_random(name: str, X: pd.DataFrame, y: np.ndarray,
                      n_trials: int = 100, n_splits: int = 5,
                      n_repeats: int = 3, random_state: int = 0):
    """Optuna tuning where the objective is repeated random-KFold test R^2.
    Use this for the report champion (where group leakage is not a concern)."""
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    suggest, factory_fn = TUNE_SPECS[name]

    def objective(trial):
        params = suggest(trial)
        return _kfold_val_r2(lambda: factory_fn(params), X, y,
                             n_splits=n_splits, n_repeats=n_repeats,
                             random_state=random_state)

    sampler = optuna.samplers.TPESampler(seed=random_state)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False,
                   catch=(Exception,))
    return study.best_params, study.best_value, study


def existing_single_split_table(target: str = "cs") -> pd.DataFrame:
    """Pull the previously-reported single-split metrics from the existing
    results xlsx so we can compare honest CV against them."""
    if target == "cs":
        path = f"{DATA_DIR}/CompressiveStrength_ML_Results.xlsx"
    else:
        path = f"{DATA_DIR}/Slump_ML_Results_new.xlsx"
    raw = pd.read_excel(path, sheet_name="Model Performance", header=None)
    # row 0 = section labels, row 1 = metric names, rows 2..8 = models
    cols_metric = raw.iloc[1].tolist()
    rows = raw.iloc[2:9].copy()
    rows.columns = ["model"] + cols_metric[1:]
    keep = ["model"]
    for split, idx_offset in (("train", 0), ("val", 8), ("test", 16)):
        for i, m in enumerate(("MAE", "MSE", "RMSE", "R2")):
            keep.append((split, m, idx_offset + i + 1))
    out = pd.DataFrame({"model": rows["model"].values})
    metric_cols = list(rows.columns)
    for split, _ in (("train", 0), ("val", 8), ("test", 16)):
        pass
    base = list(rows.columns)
    out["train_r2_old"] = pd.to_numeric(rows.iloc[:, 4].values, errors="coerce")
    out["val_r2_old"] = pd.to_numeric(rows.iloc[:, 12].values, errors="coerce")
    out["test_r2_old"] = pd.to_numeric(rows.iloc[:, 20].values, errors="coerce")
    out["overfit_gap_old"] = out["train_r2_old"] - out["test_r2_old"]
    return out
