"""Data-driven action plan (replaces threshold-only rules).

For every ACTIONABLE factor we propose a realistic target, then ask the model how many points that single
change would add (what-if). Actions are ranked by estimated gain. Non-actionable drivers (previous grade,
parental education) become support suggestions. Gender is never used for a recommendation.
Estimates are model-based associations, not guaranteed causal effects.
"""
from __future__ import annotations

import pandas as pd

from src.config import FEATURE_LABELS


def _targets(student: pd.Series, df: pd.DataFrame) -> list[tuple[str, object, str]]:
    t = []
    study, att, sleep = float(student.study_time_hours), float(student.attendance_percent), float(student.sleep_hours)
    p75 = float(df.study_time_hours.quantile(0.75))
    if study < p75:
        new = round(min(max(study + 1.0, min(p75, study + 2.0)), 12.0) * 2) / 2
        t.append(("study_time_hours", new, f"Raise daily study time from {study:.1f} h to about {new:.1f} h with a fixed weekly schedule."))
    if att < 95:
        new = min(100.0, max(att + 10.0, 85.0))
        t.append(("attendance_percent", new, f"Lift attendance from {att:.0f}% to about {new:.0f}% - prioritise the classes missed most."))
    if sleep < 7 or sleep > 9:
        new = 7.5 if sleep < 7 else 8.5
        t.append(("sleep_hours", new, f"Move sleep from {sleep:.1f} h toward {new:.1f} h for steadier concentration."))
    if student.part_time_job == "Yes":
        t.append(("part_time_job", "No", "Reduce part-time work hours (or move shifts away from exam weeks) to free study time."))
    if student.internet_access == "No":
        t.append(("internet_access", "Yes", "Arrange reliable access to online material (campus lab, library hours, offline downloads)."))
    return t


def build_action_plan(student: pd.Series, shap_df: pd.DataFrame, score: float, service) -> dict:
    base = float(service.predict(student.to_frame().T)[0])
    actions = []
    for feat, new, msg in _targets(student, service.df):
        trial = student.copy()
        trial[feat] = new
        gain = float(service.predict(trial.to_frame().T)[0]) - base
        if gain >= 0.5:
            sev = "High" if gain >= 3 else "Medium" if gain >= 1.5 else "Low"
            actions.append({"Factor": FEATURE_LABELS[feat], "Feature": feat, "Current": student[feat], "Target": new,
                            "Action": msg, "Est_gain": round(gain, 1), "Priority": sev})
    actions.sort(key=lambda a: -a["Est_gain"])

    support = []
    neg = shap_df[shap_df.SHAP_Value < -0.5]
    if "previous_grade" in neg.Feature.values:
        support.append("Previous results are pulling the forecast down: schedule targeted revision or tutoring on the weakest topics.")
    if "parental_education" in neg.Feature.values:
        support.append("Offer mentoring or study-skills support; this factor reflects home academic support, not ability.")
    if score < service.early_warning_cut:
        support.append("Early-warning flag: arrange a short check-in with the class teacher or counsellor this month.")

    strengths = [f"{FEATURE_LABELS[r.Feature]} (+{r.SHAP_Value:.1f})" for r in shap_df[shap_df.SHAP_Value > 0.5].head(3).itertuples()
                 if r.Feature != "gender"]
    return {"actions": actions[:4], "support": support, "strengths": strengths}
