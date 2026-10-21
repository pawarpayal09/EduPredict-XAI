"""Hold-out comparison on the ORIGINAL 80/20 split (random_state=42), now including
the stored TabPFN predictions, a linear baseline, bootstrap 95% CIs and paired tests.
"""
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.config import (FEATURES, RANDOM_STATE, RESULTS_DIR, TARGET, TEST_SIZE,
                        UPGRADE_DIR, ensure_dirs, load_data)
from src.cv_benchmark import _prep, build_models


def bootstrap_ci(y, p, metric, n=2000, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y))
    vals = [metric(y[b], p[b]) for b in (rng.choice(idx, len(idx)) for _ in range(n))]
    return np.percentile(vals, [2.5, 97.5])


def rmse(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def run():
    ensure_dirs()
    df = load_data()
    X, y = df[FEATURES], df[TARGET]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    yt = yte.values
    preds = {}

    for name, (kind, model) in build_models(False).items():
        m = clone(model)
        a, b = _prep(kind, Xtr, Xte, X)
        m.fit(a, ytr.values)
        preds[name] = np.clip(np.asarray(m.predict(b)), 0, 100)

    stored = pd.read_csv(RESULTS_DIR / "tabpfn_predictions_with_error.csv")
    assert np.allclose(stored["Actual"].values, yt), "Stored TabPFN predictions do not align with split"
    preds["TabPFN (stored predictions)"] = np.clip(stored["Predicted"].values, 0, 100)

    rows = []
    for name, p in preds.items():
        lo, hi = bootstrap_ci(yt, p, rmse)
        rows.append({"Model": name, "MAE": mean_absolute_error(yt, p), "RMSE": rmse(yt, p),
                     "RMSE_CI95_low": lo, "RMSE_CI95_high": hi, "R2": r2_score(yt, p)})
    res = pd.DataFrame(rows).sort_values("RMSE").reset_index(drop=True)
    res.insert(0, "Rank", res.index + 1)

    # paired Wilcoxon on absolute errors, TabPFN vs each model
    ref = np.abs(yt - preds["TabPFN (stored predictions)"])
    pv = []
    for name in res["Model"]:
        if name.startswith("TabPFN"):
            pv.append(np.nan)
        else:
            pv.append(stats.wilcoxon(ref, np.abs(yt - preds[name])).pvalue)
    res["Wilcoxon_p_vs_TabPFN"] = pv
    res.round(4).to_csv(UPGRADE_DIR / "holdout_comparison.csv", index=False)
    pd.DataFrame(preds).assign(Actual=yt).to_csv(UPGRADE_DIR / "holdout_predictions_all_models.csv", index=False)
    print(res.round(3).to_string(index=False))
    return preds, yt


if __name__ == "__main__":
    run()
