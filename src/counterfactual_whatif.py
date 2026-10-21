"""Upgrade module 6: actionable "what-if" counterfactual recommendations.

Extends the project's Prediction -> Explanation -> Intervention pipeline: instead of only
saying "improve attendance", the system searches realistic changes to ACTIONABLE features and
reports the smallest-effort plan that raises the predicted score by `target_gain` points.

IMPORTANT: counterfactuals are model-based (associational), not causal effect estimates.
`suggest_plans` takes any `predict_fn(DataFrame)->array`, so the app can pass TabPFN directly.
"""
import itertools
import warnings

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from sklearn.model_selection import train_test_split

from src.config import (FEATURES, RANDOM_STATE, TARGET, TEST_SIZE, UPGRADE_DIR, encode_for_trees,
                        ensure_dirs, load_data)

warnings.filterwarnings("ignore")

# (feature, candidate deltas, effort weight per unit). Bounds keep plans realistic.
STUDY_STEPS = [0, 0.5, 1.0, 1.5, 2.0]          # extra hours / day
ATTEND_STEPS = [0, 5, 10, 15]                   # extra percentage points
SLEEP_TARGET = 7.5                              # move toward a healthy 7-8 h window
EFFORT = {"study_time_hours": 1.0, "attendance_percent": 0.1, "sleep_hours": 0.8,
          "part_time_job": 2.5}                 # relative effort per unit (hours / pct-points / hours / switch)


def _candidates(student: pd.Series):
    study = [min(student.study_time_hours + d, 12.0) for d in STUDY_STEPS]
    attend = [min(student.attendance_percent + d, 100.0) for d in ATTEND_STEPS]
    sleep = [student.sleep_hours]
    if student.sleep_hours < SLEEP_TARGET - 0.25:
        sleep += [min(student.sleep_hours + d, SLEEP_TARGET) for d in (0.5, 1.0, 1.5)]
    job = [student.part_time_job] + (["No"] if student.part_time_job == "Yes" else [])
    return study, attend, sleep, job


def suggest_plans(student: pd.Series, predict_fn, target_gain: float = 5.0, top_n: int = 3):
    """Return (baseline_prediction, DataFrame of cheapest plans reaching +target_gain)."""
    study, attend, sleep, job = _candidates(student)
    combos = list(itertools.product(study, attend, sleep, job))
    frame = pd.DataFrame([student.to_dict()] * len(combos))
    frame["study_time_hours"], frame["attendance_percent"], frame["sleep_hours"], frame["part_time_job"] = zip(*combos)
    base = float(predict_fn(student.to_frame().T[FEATURES])[0])
    pred = np.clip(np.asarray(predict_fn(frame[FEATURES])), 0, 100)
    effort = (
        (frame.study_time_hours - student.study_time_hours) * EFFORT["study_time_hours"]
        + (frame.attendance_percent - student.attendance_percent) * EFFORT["attendance_percent"]
        + (frame.sleep_hours - student.sleep_hours) * EFFORT["sleep_hours"]
        + (frame.part_time_job != student.part_time_job) * EFFORT["part_time_job"]
    )
    out = frame.assign(Predicted_score=pred, Gain=pred - base, Effort=effort)
    ok = out[out.Gain >= target_gain].sort_values(["Effort", "Gain"], ascending=[True, False])
    cols = ["study_time_hours", "attendance_percent", "sleep_hours", "part_time_job",
            "Predicted_score", "Gain", "Effort"]
    return base, ok[cols].head(top_n).reset_index(drop=True)


def describe_plan(student: pd.Series, plan: pd.Series) -> str:
    steps = []
    d = plan.study_time_hours - student.study_time_hours
    if d > 1e-9: steps.append(f"study {d:+.1f} h/day more")
    d = plan.attendance_percent - student.attendance_percent
    if d > 1e-9: steps.append(f"raise attendance by {d:.0f} points")
    d = plan.sleep_hours - student.sleep_hours
    if d > 1e-9: steps.append(f"sleep {d:.1f} h more")
    if plan.part_time_job != student.part_time_job: steps.append("reduce part-time job commitments")
    return "; ".join(steps) if steps else "no change needed"


def run(target_gain=5.0):
    ensure_dirs()
    df = load_data()
    Xtr_r, Xte_r, ytr, yte = train_test_split(df[FEATURES], df[TARGET], test_size=TEST_SIZE,
                                              random_state=RANDOM_STATE)
    model = CatBoostRegressor(iterations=400, depth=4, learning_rate=0.03, random_seed=RANDOM_STATE,
                              verbose=0, allow_writing_files=False).fit(encode_for_trees(Xtr_r, df), ytr)
    fn = lambda X: model.predict(encode_for_trees(X, df))

    cut = np.percentile(ytr, 20)
    base_pred = fn(Xte_r)
    at_risk = Xte_r[base_pred < cut + 3]          # predicted low performers
    rows = []
    for idx, s in at_risk.iterrows():
        base, plans = suggest_plans(s, fn, target_gain)
        rows.append({"student_index": idx, "baseline_pred": round(base, 2),
                     "feasible": len(plans) > 0,
                     "best_gain": round(plans.Gain.iloc[0], 2) if len(plans) else np.nan,
                     "best_plan": describe_plan(s, plans.iloc[0]) if len(plans) else "none within bounds",
                     "effort": round(plans.Effort.iloc[0], 2) if len(plans) else np.nan})
    res = pd.DataFrame(rows)
    res.to_csv(UPGRADE_DIR / "counterfactual_plans_at_risk_students.csv", index=False)
    summary = pd.Series({"students_evaluated": len(res), "target_gain_points": target_gain,
                         "pct_with_feasible_plan": 100 * res.feasible.mean(),
                         "median_effort_units": res.effort.median(),
                         "mean_best_gain": res.best_gain.mean()}).round(3)
    summary.to_csv(UPGRADE_DIR / "counterfactual_summary.csv", header=["value"])
    print(summary.to_string())
    print(res.head(8).to_string(index=False))


if __name__ == "__main__":
    run()
