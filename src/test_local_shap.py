import sys
from pathlib import Path

import pickle
import pandas as pd


sys.path.append(
    str(Path(__file__).parent)
)


from student_insights import (
    create_student_dataframe
)

from shap_local_explanation import (
    explain_student
)


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)


# ==========================================================
# MODEL
# ==========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tabpfn_fitted_model.pkl"
)


# ==========================================================
# DATA
# ==========================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_performance_feature_engineered.csv"
)


print("=" * 60)
print("LOCAL SHAP TEST")
print("=" * 60)


# ==========================================================
# LOAD MODEL
# ==========================================================

with open(
    MODEL_PATH,
    "rb"
) as file:

    model = pickle.load(file)


print("\nTabPFN model loaded.")


# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_csv(
    DATA_PATH
)


TARGET = "final_exam_score"


X = df.drop(
    columns=[TARGET]
).copy()


print(
    "Dataset shape:",
    X.shape
)


# ==========================================================
# BACKGROUND DATA
# ==========================================================

background_data = X.sample(
    n=min(10, len(X)),
    random_state=42
).reset_index(drop=True)


# ==========================================================
# STUDENT TO EXPLAIN
# ==========================================================

student_df = create_student_dataframe(

    gender="Female",

    study_time_hours=3,

    attendance_percent=89,

    sleep_hours=7,

    parental_education="High School",

    internet_access="Yes",

    extracurricular_activities="No",

    part_time_job="No",

    previous_grade=75
)


print("\nStudent to explain:")
print(student_df)


# ==========================================================
# SHAP
# ==========================================================

print("\nCalculating local SHAP explanation...")

explanation_df = explain_student(

    model=model,

    background_data=background_data,

    student_data=student_df,

    max_evals=25
)


# ==========================================================
# RESULT
# ==========================================================

print("\n" + "=" * 60)
print("LOCAL SHAP EXPLANATION")
print("=" * 60)

print(
    explanation_df
)


# ==========================================================
# TOP FEATURES
# ==========================================================

print("\nTop contributing features:")

for i, row in explanation_df.head(5).iterrows():

    direction = (
        "Positive"
        if row["SHAP_Value"] > 0
        else "Negative"
    )

    print(
        f"{i + 1}. "
        f"{row['Feature']} → "
        f"{direction} "
        f"({row['SHAP_Value']:.4f})"
    )


print("\nLocal SHAP analysis completed.")

# ==========================================================
# SAVE SHAP RESULT
# ==========================================================

XAI_PATH = (
    PROJECT_ROOT
    / "results"
    / "xai"
)

XAI_PATH.mkdir(
    parents=True,
    exist_ok=True
)


SHAP_OUTPUT = (
    XAI_PATH
    / "local_shap_student_example.csv"
)


explanation_df.to_csv(
    SHAP_OUTPUT,
    index=False
)


print(
    "\nSHAP explanation saved to:"
)

print(
    SHAP_OUTPUT
)