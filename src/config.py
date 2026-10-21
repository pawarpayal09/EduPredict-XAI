"""Shared configuration and helpers for the EduPredict-XAI upgrade modules."""
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "student_performance_feature_engineered.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
UPGRADE_DIR = RESULTS_DIR / "upgrade"
FIG_DIR = UPGRADE_DIR / "figures"

TARGET = "final_exam_score"
FEATURES = [
    "gender", "study_time_hours", "attendance_percent", "sleep_hours",
    "parental_education", "internet_access", "extracurricular_activities",
    "part_time_job", "previous_grade",
]
NUMERIC = ["study_time_hours", "attendance_percent", "sleep_hours", "previous_grade"]
CATEGORICAL = ["gender", "parental_education", "internet_access",
               "extracurricular_activities", "part_time_job"]

# Features a student/teacher can realistically change (used for what-if analysis).
# gender, parental_education and previous_grade are NOT actionable.
ACTIONABLE = ["study_time_hours", "attendance_percent", "sleep_hours",
              "internet_access", "part_time_job", "extracurricular_activities"]

# Same split as the original notebooks (04 / 06): test_size=0.20, random_state=42
TEST_SIZE = 0.20
RANDOM_STATE = 42

# Same thresholds as the original app / intervention module
CATEGORY_BINS = [(75, "Excellent"), (60, "Good"), (50, "Average"),
                 (35, "Needs Improvement"), (-np.inf, "At Risk")]
CATEGORY_ORDER = ["At Risk", "Needs Improvement", "Average", "Good", "Excellent"]


def get_category(score: float) -> str:
    for threshold, name in CATEGORY_BINS:
        if score >= threshold:
            return name
    return "At Risk"


def load_data() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


def ensure_dirs() -> None:
    UPGRADE_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def encode_for_trees(X: pd.DataFrame, reference: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ordinal-encode categorical columns (stable alphabetical mapping).

    The mapping is built from `reference` (default: X) so train/test/app use the same codes.
    """
    ref = reference if reference is not None else X
    out = X[FEATURES].copy()
    for col in CATEGORICAL:
        cats = sorted(ref[col].astype(str).unique())
        out[col] = out[col].astype(str).map({c: float(i) for i, c in enumerate(cats)})
    return out.astype(float)


FEATURE_LABELS = {
    "gender": "Gender", "study_time_hours": "Daily study time", "attendance_percent": "Attendance",
    "sleep_hours": "Sleep", "parental_education": "Parental education", "internet_access": "Internet access",
    "extracurricular_activities": "Extracurriculars", "part_time_job": "Part-time job",
    "previous_grade": "Previous grade",
}
ALLOWED = {
    "gender": ["Female", "Male"],
    "parental_education": ["High School", "Bachelors", "Masters", "PhD"],
    "internet_access": ["Yes", "No"], "extracurricular_activities": ["Yes", "No"], "part_time_job": ["Yes", "No"],
}
RANGES = {"study_time_hours": (0, 24), "attendance_percent": (0, 100), "sleep_hours": (0, 24),
          "previous_grade": (0, 100)}
