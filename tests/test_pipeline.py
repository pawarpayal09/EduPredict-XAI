import numpy as np
import pandas as pd
import pytest

from src.config import CATEGORICAL, FEATURES, TARGET, encode_for_trees, get_category, load_data
from src.counterfactual_whatif import suggest_plans
from src.cv_benchmark import corrected_resampled_ttest
from src.uncertainty_and_risk import conformal_quantile


def test_dataset_schema():
    df = load_data()
    assert set(FEATURES + [TARGET]) <= set(df.columns)
    assert df[FEATURES + [TARGET]].isna().sum().sum() == 0
    assert df[TARGET].between(0, 100).all()


def test_category_thresholds():
    assert get_category(80) == "Excellent" and get_category(60) == "Good"
    assert get_category(50) == "Average" and get_category(40) == "Needs Improvement"
    assert get_category(10) == "At Risk"


def test_encoding_is_stable_and_numeric():
    df = load_data()
    enc = encode_for_trees(df[FEATURES].head(5), df)
    assert enc.shape == (5, len(FEATURES)) and enc.dtypes.eq(float).all()
    assert enc["gender"].isin([0.0, 1.0]).all()


def test_conformal_quantile_coverage():
    rng = np.random.default_rng(0)
    cal, test = np.abs(rng.normal(size=5000)), np.abs(rng.normal(size=50000))
    q = conformal_quantile(cal, 0.1)
    assert abs((test <= q).mean() - 0.9) < 0.015


def test_corrected_ttest_identical_models():
    t, p = corrected_resampled_ttest(np.zeros(15), 800, 200)
    assert p == 1.0


def test_whatif_gain_is_reported_consistently():
    student = load_data()[FEATURES].iloc[0].copy()
    fn = lambda X: 0.5 * X["study_time_hours"].values * 5 + 0.1 * X["attendance_percent"].values
    base, plans = suggest_plans(student, fn, target_gain=2.0)
    assert (plans.Gain >= 2.0 - 1e-9).all()
    assert plans.Effort.is_monotonic_increasing
