"""EduPredictService - single backend entry point used by the Streamlit app, the REST API and tests.

Responsibilities: validation, prediction, conformal intervals, categories, SHAP explanations
(exact TreeSHAP when a tree model is active, permutation SHAP for TabPFN), what-if search,
batch scoring and plain-language explanations.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
import shap

from src.config import (ALLOWED, CATEGORICAL, FEATURE_LABELS, FEATURES, MODELS_DIR, NUMERIC, RANGES, TARGET,
                        encode_for_trees, get_category, load_data)
from src.counterfactual_whatif import describe_plan, suggest_plans
from src.model_loader import EncodedCatBoost, load_primary_model

warnings.filterwarnings("ignore")
BAND_LABELS = ["At Risk", "Needs Improvement", "Average", "Good", "Excellent"]


class ValidationError(ValueError):
    def __init__(self, problems: list[str]):
        super().__init__("; ".join(problems))
        self.problems = problems


class EduPredictService:
    def __init__(self):
        self.model, self.model_name, self.q_key, self.note = load_primary_model()
        self.df = load_data()
        self.scores = self.df[TARGET].values
        self.cohort_mean = float(self.scores.mean())
        self.early_warning_cut = float(np.percentile(self.scores, 20)) + 3.0   # see early_warning_metrics.csv
        self.half_width = self._read_json("conformal_quantiles.json", "half_width", {}).get(self.q_key, {})
        self.rel_cuts = self._read_json("relative_risk_cutoffs.json", "score_cutoffs", None)
        self.is_tree = isinstance(self.model, EncodedCatBoost)
        self._bg = self.df[FEATURES].sample(40, random_state=42)

    # ----------------------------------------------------------------- helpers
    @staticmethod
    def _read_json(name, key, default):
        p = MODELS_DIR / name
        return json.loads(p.read_text()).get(key, default) if p.exists() else default

    @staticmethod
    def validate(records: pd.DataFrame | dict) -> pd.DataFrame:
        df = pd.DataFrame([records]) if isinstance(records, dict) else records.copy()
        problems = [f"missing column: {c}" for c in FEATURES if c not in df.columns]
        if problems:
            raise ValidationError(problems)
        df = df[FEATURES].copy()
        for c in NUMERIC:
            df[c] = pd.to_numeric(df[c], errors="coerce")
            lo, hi = RANGES[c]
            bad = df[c].isna() | (df[c] < lo) | (df[c] > hi)
            if bad.any():
                problems.append(f"{c}: {int(bad.sum())} value(s) missing or outside {lo}-{hi} (first at row {int(np.argmax(bad.values)) + 1})")
        for c in CATEGORICAL:
            df[c] = df[c].astype(str).str.strip()
            bad = ~df[c].isin(ALLOWED[c])
            if bad.any():
                problems.append(f"{c}: {int(bad.sum())} invalid value(s); allowed {ALLOWED[c]}")
        if problems:
            raise ValidationError(problems)
        return df.reset_index(drop=True)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.clip(np.asarray(self.model.predict(X[FEATURES])).reshape(-1), 0, 100)

    def percentile(self, score: float) -> float:
        return float((self.scores < score).mean() * 100)

    def cohort_band(self, score: float) -> str:
        if not self.rel_cuts:
            return get_category(score)
        return BAND_LABELS[int(sum(score >= c for c in self.rel_cuts))]

    def interval(self, score: float, level: str = "90"):
        hw = self.half_width.get(level, 10.0)
        return max(0.0, score - hw), min(100.0, score + hw)

    # --------------------------------------------------------------- single student
    def predict_one(self, student: dict) -> dict:
        row = self.validate(student)
        score = float(self.predict(row)[0])
        out = {"predicted_score": round(score, 2), "category": get_category(score),
               "cohort_band": self.cohort_band(score), "cohort_percentile": round(self.percentile(score), 1),
               "early_warning": bool(score < self.early_warning_cut), "model": self.model_name,
               "cohort_mean": round(self.cohort_mean, 2)}
        for lvl in ("80", "90", "95"):
            lo, hi = self.interval(score, lvl)
            out[f"interval_{lvl}"] = [round(lo, 2), round(hi, 2)]
        return out

    def explain(self, row: pd.DataFrame) -> tuple[float, pd.DataFrame]:
        """Return (base_value, DataFrame[Feature, Value, SHAP_Value, Absolute_SHAP]) for one student row."""
        row = row[FEATURES].reset_index(drop=True)
        if self.is_tree:
            enc = encode_for_trees(row, self.model.reference)
            ex = shap.TreeExplainer(self.model.model)
            sv = np.asarray(ex.shap_values(enc))[0]
            base = float(np.ravel(ex.expected_value)[0])
        else:
            sv, base = self._permutation_shap(row)
        out = pd.DataFrame({"Feature": FEATURES, "Value": [row.iloc[0][f] for f in FEATURES], "SHAP_Value": sv})
        out["Absolute_SHAP"] = out.SHAP_Value.abs()
        return base, out.sort_values("Absolute_SHAP", ascending=False).reset_index(drop=True)

    def _permutation_shap(self, row: pd.DataFrame):
        """Model-agnostic SHAP (used for TabPFN). Same approach as the original app, cleaned up."""
        ref = self.df[FEATURES]
        maps = {c: {k: float(i) for i, k in enumerate(sorted(ref[c].astype(str).unique()))} for c in CATEGORICAL}
        rev = {c: {v: k for k, v in m.items()} for c, m in maps.items()}

        def enc(frame):
            f = frame.copy()
            for c in CATEGORICAL:
                f[c] = f[c].astype(str).map(maps[c])
            return f.astype(float)

        def fn(arr):
            f = pd.DataFrame(arr, columns=FEATURES)
            for c in CATEGORICAL:
                f[c] = f[c].round().map(rev[c])
            return self.predict(f)

        ex = shap.Explainer(fn, enc(self._bg), algorithm="permutation")
        res = ex(enc(row), max_evals=2 * len(FEATURES) + 1)
        vals = np.asarray(res.values)
        vals = vals[0] if vals.ndim == 2 else vals[0, :, 0]
        return np.asarray(vals, dtype=float).reshape(-1), float(np.ravel(res.base_values)[0])

    def explanation_text(self, score: float, base: float, shap_df: pd.DataFrame) -> str:
        d = score - self.cohort_mean
        pos = shap_df[shap_df.SHAP_Value > 0.3].head(2)
        neg = shap_df[shap_df.SHAP_Value < -0.3].head(2)
        fmt = lambda r: f"{FEATURE_LABELS[r.Feature].lower()} ({r.SHAP_Value:+.1f})"
        txt = (f"The predicted score is {score:.1f}, {abs(d):.1f} points {'above' if d >= 0 else 'below'} "
               f"the cohort average of {self.cohort_mean:.1f}. ")
        if len(pos):
            txt += "Strongest boosts: " + ", ".join(fmt(r) for r in pos.itertuples()) + ". "
        if len(neg):
            txt += "Biggest drags: " + ", ".join(fmt(r) for r in neg.itertuples()) + "."
        else:
            txt += "No factor is pulling the prediction down noticeably."
        return txt

    # ------------------------------------------------------------------ what-if
    def whatif(self, student: pd.Series, target_gain: float = 5.0, top_n: int = 3):
        base, plans = suggest_plans(student, self.predict, target_gain, top_n)
        texts = [describe_plan(student, p) for _, p in plans.iterrows()]
        return base, plans, texts

    def simulate(self, student: dict, **changes) -> float:
        row = self.validate({**student, **changes})
        return float(self.predict(row)[0])

    # -------------------------------------------------------------------- batch
    def batch(self, records: pd.DataFrame) -> pd.DataFrame:
        X = self.validate(records)
        pred = self.predict(X)
        out = X.copy()
        out.insert(0, "Predicted_Score", pred.round(2))
        out["Lower_90"] = np.clip(pred - self.half_width.get("90", 10), 0, 100).round(2)
        out["Upper_90"] = np.clip(pred + self.half_width.get("90", 10), 0, 100).round(2)
        out["Category"] = [get_category(p) for p in pred]
        out["Cohort_Band"] = [self.cohort_band(p) for p in pred]
        out["Percentile"] = [round(self.percentile(p), 1) for p in pred]
        out["Early_Warning"] = pred < self.early_warning_cut
        if self.is_tree:                       # exact TreeSHAP is instant; skip for slow TabPFN
            sv = np.asarray(shap.TreeExplainer(self.model.model).shap_values(
                encode_for_trees(X, self.model.reference)))
            out["Top_Positive_Driver"] = [FEATURE_LABELS[FEATURES[i]] for i in sv.argmax(axis=1)]
            out["Top_Negative_Driver"] = [FEATURE_LABELS[FEATURES[i]] for i in sv.argmin(axis=1)]
        return out
