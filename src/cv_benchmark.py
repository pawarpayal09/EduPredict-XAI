"""Rigorous model benchmark: repeated K-fold CV + paired statistical tests.

Upgrade over the original single 80/20 hold-out comparison. Reports mean +/- std
for MAE / RMSE / R2 across repeated folds and a Nadeau-Bengio corrected resampled
t-test against the best model. TabPFN is optional (needs its weights/token):
    python -m src.cv_benchmark --with-tabpfn
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (CATEGORICAL, FEATURES, NUMERIC, RANDOM_STATE, TARGET,
                        UPGRADE_DIR, encode_for_trees, ensure_dirs, load_data)

warnings.filterwarnings("ignore")


def build_models(with_tabpfn: bool = False) -> dict:
    from catboost import CatBoostRegressor
    from lightgbm import LGBMRegressor
    from xgboost import XGBRegressor

    linear_prep = ColumnTransformer([
        ("num", StandardScaler(), NUMERIC),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), CATEGORICAL),
    ])
    models = {
        "Mean baseline": ("raw", DummyRegressor()),
        "Ridge (linear)": ("raw", make_pipeline(linear_prep, RidgeCV(alphas=np.logspace(-3, 3, 25)))),
        "Random Forest": ("enc", RandomForestRegressor(n_estimators=300, min_samples_leaf=3,
                                                       random_state=RANDOM_STATE, n_jobs=-1)),
        "Extra Trees": ("enc", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=3,
                                                   random_state=RANDOM_STATE, n_jobs=-1)),
        "CatBoost": ("enc", CatBoostRegressor(iterations=500, depth=4, learning_rate=0.03,
                                              random_seed=RANDOM_STATE, verbose=0,
                                              allow_writing_files=False)),
        "XGBoost": ("enc", XGBRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                        subsample=0.8, colsample_bytree=0.8,
                                        random_state=RANDOM_STATE, n_jobs=-1)),
        "LightGBM": ("enc", LGBMRegressor(n_estimators=300, max_depth=3, learning_rate=0.03,
                                          num_leaves=8, min_child_samples=20, subsample=0.8,
                                          subsample_freq=1, colsample_bytree=0.8,
                                          random_state=RANDOM_STATE, verbose=-1, n_jobs=1)),
    }
    if with_tabpfn:
        from tabpfn import TabPFNRegressor
        models["TabPFN"] = ("raw_cat", TabPFNRegressor(device="cpu", random_state=RANDOM_STATE))
    return models


def _prep(kind, X_tr, X_te, ref):
    if kind == "enc":
        return encode_for_trees(X_tr, ref), encode_for_trees(X_te, ref)
    if kind == "raw_cat":
        a, b = X_tr.copy(), X_te.copy()
        for c in CATEGORICAL:
            cats = sorted(ref[c].astype(str).unique())
            a[c] = pd.Categorical(a[c].astype(str), categories=cats)
            b[c] = pd.Categorical(b[c].astype(str), categories=cats)
        return a, b
    return X_tr, X_te


def corrected_resampled_ttest(diff: np.ndarray, n_train: int, n_test: int):
    """Nadeau & Bengio (2003) corrected t-test for repeated CV differences."""
    k = len(diff)
    var = np.var(diff, ddof=1)
    if var == 0:
        return 0.0, 1.0
    t = np.mean(diff) / np.sqrt((1 / k + n_test / n_train) * var)
    p = 2 * stats.t.sf(abs(t), df=k - 1)
    return float(t), float(p)


def run(n_splits=5, n_repeats=3, with_tabpfn=False):
    ensure_dirs()
    df = load_data()
    X, y = df[FEATURES], df[TARGET].values
    models = build_models(with_tabpfn)
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=RANDOM_STATE)

    fold_rows = []
    for fold_id, (tr, te) in enumerate(rkf.split(X)):
        for name, (kind, model) in models.items():
            from sklearn.base import clone
            m = clone(model)
            Xa, Xb = _prep(kind, X.iloc[tr], X.iloc[te], X)
            m.fit(Xa, y[tr])
            pred = np.clip(np.asarray(m.predict(Xb)), 0, 100)
            fold_rows.append({
                "Fold": fold_id, "Model": name,
                "MAE": mean_absolute_error(y[te], pred),
                "RMSE": float(np.sqrt(mean_squared_error(y[te], pred))),
                "R2": r2_score(y[te], pred),
            })
        print(f"fold {fold_id + 1}/{n_splits * n_repeats} done")

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(UPGRADE_DIR / "cv_fold_results.csv", index=False)

    summary = folds.groupby("Model").agg(
        MAE_mean=("MAE", "mean"), MAE_std=("MAE", "std"),
        RMSE_mean=("RMSE", "mean"), RMSE_std=("RMSE", "std"),
        R2_mean=("R2", "mean"), R2_std=("R2", "std")).sort_values("RMSE_mean")
    best = summary.index[0]
    n_te = len(X) // n_splits
    n_tr = len(X) - n_te
    ps, ts = [], []
    best_rmse = folds[folds.Model == best].sort_values("Fold")["RMSE"].values
    for name in summary.index:
        other = folds[folds.Model == name].sort_values("Fold")["RMSE"].values
        t, p = corrected_resampled_ttest(other - best_rmse, n_tr, n_te) if name != best else (np.nan, np.nan)
        ts.append(t); ps.append(p)
    summary["t_vs_best"] = ts
    summary["p_vs_best"] = ps
    summary.insert(0, "Rank", range(1, len(summary) + 1))
    summary.round(4).to_csv(UPGRADE_DIR / "cv_model_summary.csv")
    print(f"\nBest model by CV RMSE: {best}")
    print(summary.round(3).to_string())
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-tabpfn", action="store_true")
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--repeats", type=int, default=3)
    a = ap.parse_args()
    run(a.splits, a.repeats, a.with_tabpfn)
