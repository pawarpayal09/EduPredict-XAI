# ============================================================
# EduPredict-XAI
# Multiple Student Local SHAP Analysis
# ============================================================

import sys
import pickle
from pathlib import Path

import pandas as pd


# ------------------------------------------------------------
# Project root
# ------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_performance_feature_engineered.csv"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tabpfn_fitted_model.pkl"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "xai"
)

RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)

# ------------------------------------------------------------
# Import SHAP function
# ------------------------------------------------------------

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

from shap_local_explanation import explain_student


print("=" * 60)
print("MULTIPLE STUDENT LOCAL SHAP ANALYSIS")
print("=" * 60)

# ------------------------------------------------------------
# Load data
# ------------------------------------------------------------

df = pd.read_csv(DATA_PATH)

TARGET = "final_exam_score"

X = df.drop(
    columns=[TARGET]
).copy()

y = df[TARGET].copy()

print("\nDataset:", df.shape)

# ------------------------------------------------------------
# Load model
# ------------------------------------------------------------

with open(MODEL_PATH, "rb") as f:
    tabpfn_model = pickle.load(f)

print("TabPFN model loaded.")

# ------------------------------------------------------------
# Background data
# ------------------------------------------------------------

background_data = X.sample(
    n=min(50, len(X)),
    random_state=42
)

# ------------------------------------------------------------
# Students to explain
# ------------------------------------------------------------

student_indices = [0, 1, 2, 3, 4]

all_results = []

# ------------------------------------------------------------
# Explain each student
# ------------------------------------------------------------

for student_index in student_indices:

    print("\n" + "=" * 60)
    print(
        f"EXPLAINING STUDENT {student_index}"
    )
    print("=" * 60)

    student = X.iloc[
        [student_index]
    ].copy()

    actual_score = y.iloc[
        student_index
    ]

    prediction = tabpfn_model.predict(
        student
    )

    predicted_score = float(
        prediction[0]
    )

    print(
        "Actual score:",
        actual_score
    )

    print(
        "Predicted score:",
        round(
            predicted_score,
            2
        )
    )

    explanation_df = explain_student(
        model=tabpfn_model,
        background_data=background_data,
        student_data=student,
        max_evals=100
    )

    explanation_df.insert(
        0,
        "Student_Index",
        student_index
    )

    explanation_df.insert(
        1,
        "Actual_Score",
        actual_score
    )

    explanation_df.insert(
        2,
        "Predicted_Score",
        predicted_score
    )

    all_results.append(
        explanation_df
    )

# ------------------------------------------------------------
# Combine results
# ------------------------------------------------------------

combined_df = pd.concat(
    all_results,
    ignore_index=True
)

# ------------------------------------------------------------
# Save
# ------------------------------------------------------------

output_path = (
    RESULTS_PATH
    / "multiple_students_local_shap.csv"
)

combined_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("MULTIPLE STUDENT SHAP ANALYSIS COMPLETED")
print("=" * 60)

print(
    "\nStudents explained:",
    len(student_indices)
)

print(
    "Rows generated:",
    len(combined_df)
)

print(
    "\nSaved to:"
)

print(output_path)