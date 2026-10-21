"""Upgrade modules 2 & 3:
  (a) split-conformal prediction intervals (distribution-free, model-agnostic)
  (b) risk-category evaluation: original absolute thresholds vs cohort-relative thresholds,
      plus an early-warning ("bottom 20%") detector evaluation.
Uses the saved hold-out predictions, so it also covers the stored TabPFN predictions.
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split

from src.config import (CATEGORY_ORDER, FEATURES, FIG_DIR, MODELS_DIR, RANDOM_STATE, TARGET,
                        TEST_SIZE, UPGRADE_DIR, ensure_dirs, get_category, load_data)

ALPHAS = [0.20, 0.10, 0.05]          # 80 %, 90 %, 95 % intervals
PRIMARY = "TabPFN (stored predictions)"


# ------------------------------------------------------------------ conformal
def conformal_quantile(abs_residuals: np.ndarray, alpha: float) -> float:
    n = len(abs_residuals)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(abs_residuals, level, method="higher"))


def empirical_coverage(y, p, alpha, reps=1000, seed=0):
    """Randomly halve the hold-out set: calibrate on one half, test coverage on the other."""
    rng = np.random.default_rng(seed)
    cov, wid = [], []
    for _ in range(reps):
        idx = rng.permutation(len(y))
        cal, ev = idx[: len(y) // 2], idx[len(y) // 2:]
        q = conformal_quantile(np.abs(y[cal] - p[cal]), alpha)
        lo, hi = np.clip(p[ev] - q, 0, 100), np.clip(p[ev] + q, 0, 100)
        cov.append(np.mean((y[ev] >= lo) & (y[ev] <= hi)))
        wid.append(np.mean(hi - lo))
    return float(np.mean(cov)), float(np.std(cov)), float(np.mean(wid))


def run_conformal(preds: pd.DataFrame):
    y = preds["Actual"].values
    rows, quantiles, all_q = [], {}, {}
    for model in [c for c in preds.columns if c not in ("Actual", "Mean baseline")]:
        p = preds[model].values
        for a in ALPHAS:
            cov, sd, width = empirical_coverage(y, p, a)
            rows.append({"Model": model, "Nominal": f"{int((1 - a) * 100)}%",
                         "Empirical_coverage": round(cov, 4), "Coverage_std": round(sd, 4),
                         "Mean_interval_width": round(width, 2)})
        q_all = {f"{int((1 - a) * 100)}": conformal_quantile(np.abs(y - p), a) for a in ALPHAS}
        if model == PRIMARY:
            quantiles = q_all
            all_q["TabPFN"] = q_all
        elif model == "CatBoost":
            all_q["CatBoost (fallback)"] = q_all
    out = pd.DataFrame(rows)
    out.to_csv(UPGRADE_DIR / "conformal_coverage.csv", index=False)
    # saved for the app: +/- half-width for TabPFN at 80/90/95 %
    (MODELS_DIR / "conformal_quantiles.json").write_text(json.dumps(
        {"source": "split-conformal on the 200 hold-out residuals of each model",
         "n_calibration": int(len(y)), "half_width": all_q}, indent=2))
    print(out[out.Model == PRIMARY].to_string(index=False))
    print("Saved half-widths:", quantiles)

    # figure
    p = preds[PRIMARY].values
    q = quantiles["90"]
    order = np.argsort(p)
    plt.figure(figsize=(7, 5))
    plt.fill_between(np.arange(len(p)), np.clip(p[order] - q, 0, 100), np.clip(p[order] + q, 0, 100),
                     alpha=0.25, label="90% conformal interval")
    plt.plot(p[order], label="Predicted", lw=1.5)
    plt.scatter(np.arange(len(p)), y[order], s=8, c="k", label="Actual")
    plt.xlabel("Hold-out students (sorted by prediction)"); plt.ylabel("Final exam score")
    plt.title("TabPFN predictions with 90% conformal intervals"); plt.legend(); plt.tight_layout()
    plt.savefig(FIG_DIR / "conformal_intervals_tabpfn.png", dpi=200); plt.close()


# ----------------------------------------------------------------------- risk
def relative_bands(train_scores: np.ndarray):
    """Cohort-relative categories: percentile cut-offs fitted on TRAINING scores only."""
    c = np.percentile(train_scores, [10, 30, 60, 85])
    def f(s):
        return ("At Risk" if s < c[0] else "Needs Improvement" if s < c[1] else
                "Average" if s < c[2] else "Good" if s < c[3] else "Excellent")
    return f, c


def run_risk(preds: pd.DataFrame):
    df = load_data()
    Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df[TARGET], test_size=TEST_SIZE,
                                          random_state=RANDOM_STATE)
    y = preds["Actual"].values
    rel, cuts = relative_bands(ytr.values)
    (MODELS_DIR / "relative_risk_cutoffs.json").write_text(json.dumps(
        {"percentiles": [10, 30, 60, 85], "score_cutoffs": [float(c) for c in cuts],
         "fitted_on": "training split (n=800)"}, indent=2))

    rows, low_cut = [], float(np.percentile(ytr.values, 20))
    for model in [PRIMARY, "Ridge (linear)", "CatBoost"]:
        p = preds[model].values
        for scheme, fn in [("Absolute (original 75/60/50/35)", get_category), ("Cohort-relative (percentile)", rel)]:
            true_c, pred_c = [fn(v) for v in y], [fn(v) for v in p]
            rows.append({"Model": model, "Scheme": scheme,
                         "Accuracy": accuracy_score(true_c, pred_c),
                         "Macro_F1": f1_score(true_c, pred_c, average="macro", labels=CATEGORY_ORDER, zero_division=0),
                         "Within_1_category": np.mean(np.abs(
                             np.array([CATEGORY_ORDER.index(c) for c in true_c]) -
                             np.array([CATEGORY_ORDER.index(c) for c in pred_c])) <= 1)})
    cat = pd.DataFrame(rows).round(4)
    cat.to_csv(UPGRADE_DIR / "risk_category_metrics.csv", index=False)
    print(cat.to_string(index=False))

    # class balance of both schemes on the full data
    dist = pd.DataFrame({
        "Absolute": df[TARGET].apply(get_category).value_counts(),
        "Cohort-relative": df[TARGET].apply(rel).value_counts()}).reindex(CATEGORY_ORDER).fillna(0).astype(int)
    dist.to_csv(UPGRADE_DIR / "category_distribution.csv")
    print("\nCategory distribution on all 1000 students:\n", dist)

    # early-warning detector: actual score in bottom 20 % (training-based cut-off)
    flag = (y < low_cut).astype(int)
    rows = []
    for model in [PRIMARY, "Ridge (linear)", "CatBoost"]:
        p = preds[model].values
        for thr in [low_cut, low_cut + 3, low_cut + 6]:
            pred_flag = (p < thr).astype(int)
            rows.append({"Model": model, "True_low_cutoff": round(low_cut, 2),
                         "Flag_if_pred_below": round(thr, 2),
                         "Precision": precision_score(flag, pred_flag, zero_division=0),
                         "Recall": recall_score(flag, pred_flag, zero_division=0),
                         "F1": f1_score(flag, pred_flag, zero_division=0),
                         "AUC": roc_auc_score(flag, -p)})
    ew = pd.DataFrame(rows).round(4)
    ew.to_csv(UPGRADE_DIR / "early_warning_metrics.csv", index=False)
    print("\nEarly-warning (bottom 20%) detection:\n", ew.to_string(index=False))

    # confusion matrix figure (primary, cohort-relative)
    cm = confusion_matrix([rel(v) for v in y], [rel(v) for v in preds[PRIMARY]], labels=CATEGORY_ORDER)
    plt.figure(figsize=(6, 5)); plt.imshow(cm, cmap="Blues")
    plt.xticks(range(5), CATEGORY_ORDER, rotation=35, ha="right"); plt.yticks(range(5), CATEGORY_ORDER)
    for i in range(5):
        for j in range(5):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.xlabel("Predicted category"); plt.ylabel("Actual category")
    plt.title("TabPFN - cohort-relative performance categories"); plt.tight_layout()
    plt.savefig(FIG_DIR / "confusion_matrix_relative_categories.png", dpi=200); plt.close()


def run():
    ensure_dirs()
    preds = pd.read_csv(UPGRADE_DIR / "holdout_predictions_all_models.csv")
    run_conformal(preds)
    print()
    run_risk(preds)


if __name__ == "__main__":
    run()
