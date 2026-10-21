"""Upgrade module 5: subgroup error audit + counterfactual gender-flip test.

Gender, parental education etc. are protected / non-actionable attributes, so a responsible
educational-prediction paper must show the model does not err systematically for a group.
Uses (i) stored TabPFN hold-out predictions (n=200) and (ii) 5-fold out-of-fold CatBoost
predictions on all 1000 students for more statistical power.
"""
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from scipy import stats
from sklearn.model_selection import KFold, train_test_split

from src.config import (FEATURES, RANDOM_STATE, TARGET, TEST_SIZE, UPGRADE_DIR, encode_for_trees,
                        ensure_dirs, load_data)

warnings.filterwarnings("ignore")
GROUPS = ["gender", "parental_education", "internet_access", "part_time_job", "extracurricular_activities"]


def _cb():
    return CatBoostRegressor(iterations=400, depth=4, learning_rate=0.03, random_seed=RANDOM_STATE,
                             verbose=0, allow_writing_files=False)


def audit(frame: pd.DataFrame, pred_col: str, label: str) -> pd.DataFrame:
    rows = []
    frame = frame.assign(err=frame[pred_col] - frame[TARGET])
    for g in GROUPS:
        for val, sub in frame.groupby(g):
            rows.append({"Predictor": label, "Attribute": g, "Group": val, "n": len(sub),
                         "MAE": sub.err.abs().mean(), "Bias_pred_minus_actual": sub.err.mean(),
                         "RMSE": float(np.sqrt((sub.err ** 2).mean()))})
    return pd.DataFrame(rows)


def gap_tests(frame: pd.DataFrame, pred_col: str, label: str) -> pd.DataFrame:
    rows = []
    ae = (frame[pred_col] - frame[TARGET]).abs()
    for g in GROUPS:
        levels = sorted(frame[g].unique())
        if len(levels) != 2:
            h = stats.kruskal(*[ae[frame[g] == l] for l in levels])
            rows.append({"Predictor": label, "Attribute": g, "Comparison": "all groups (Kruskal-Wallis)",
                         "MAE_gap": ae.groupby(frame[g]).mean().max() - ae.groupby(frame[g]).mean().min(),
                         "p_value": h.pvalue})
        else:
            a, b = ae[frame[g] == levels[0]], ae[frame[g] == levels[1]]
            rows.append({"Predictor": label, "Attribute": g, "Comparison": f"{levels[0]} vs {levels[1]}",
                         "MAE_gap": a.mean() - b.mean(), "p_value": stats.mannwhitneyu(a, b).pvalue})
    return pd.DataFrame(rows)


def run():
    ensure_dirs()
    df = load_data()

    # (ii) out-of-fold CatBoost on all students
    X, y = encode_for_trees(df[FEATURES], df), df[TARGET].values
    oof = np.zeros(len(df))
    for tr, te in KFold(5, shuffle=True, random_state=RANDOM_STATE).split(X):
        oof[te] = np.clip(_cb().fit(X.iloc[tr], y[tr]).predict(X.iloc[te]), 0, 100)
    full = df.assign(pred_oof=oof)

    # (i) stored TabPFN hold-out predictions
    _, Xte, _, _ = train_test_split(df[FEATURES], df[TARGET], test_size=TEST_SIZE, random_state=RANDOM_STATE)
    preds = pd.read_csv(UPGRADE_DIR / "holdout_predictions_all_models.csv")
    hold = df.loc[Xte.index].reset_index(drop=True).assign(pred_tabpfn=preds["TabPFN (stored predictions)"].values)

    tables = pd.concat([audit(hold, "pred_tabpfn", "TabPFN (hold-out, n=200)"),
                        audit(full, "pred_oof", "CatBoost (5-fold OOF, n=1000)")])
    tests = pd.concat([gap_tests(hold, "pred_tabpfn", "TabPFN (hold-out, n=200)"),
                       gap_tests(full, "pred_oof", "CatBoost (5-fold OOF, n=1000)")])
    tables.round(3).to_csv(UPGRADE_DIR / "fairness_subgroup_errors.csv", index=False)
    tests.round(4).to_csv(UPGRADE_DIR / "fairness_gap_tests.csv", index=False)
    print(tables.round(2).to_string(index=False))
    print("\n", tests.round(4).to_string(index=False))

    # counterfactual gender flip: does changing ONLY gender change the prediction?
    m = _cb().fit(X, y)
    flipped = df[FEATURES].copy()
    flipped["gender"] = flipped["gender"].map({"Male": "Female", "Female": "Male"})
    delta = m.predict(encode_for_trees(flipped, df)) - m.predict(X)
    pd.Series({"mean_abs_change": np.abs(delta).mean(), "max_abs_change": np.abs(delta).max(),
               "pct_students_changed_over_0.5_points": 100 * np.mean(np.abs(delta) > 0.5)}).round(4).to_csv(
        UPGRADE_DIR / "fairness_gender_flip.csv", header=["value"])
    print("\nGender-flip test: mean |change| = %.3f, max = %.3f points" %
          (np.abs(delta).mean(), np.abs(delta).max()))


if __name__ == "__main__":
    run()
