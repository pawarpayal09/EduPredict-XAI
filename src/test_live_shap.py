# ============================================================
# EduPredict-XAI
# Step 24 — Test Live SHAP
# ============================================================

import pickle
from pathlib import Path

import pandas as pd

from live_shap import explain_student


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# 2. PATHS
# ============================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tabpfn_fitted_model.pkl"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_performance_feature_engineered.csv"
)


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("=" * 60)
print("STEP 24 — LIVE SHAP TEST")
print("=" * 60)

print("\nLoading TabPFN model...")

with open(
    MODEL_PATH,
    "rb"
) as file:

    model = pickle.load(file)

print("TabPFN model loaded successfully.")


# ============================================================
# 4. LOAD DATA
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(
    DATA_PATH
)

print(
    "Dataset shape:",
    df.shape
)


# ============================================================
# 5. SEPARATE FEATURES AND TARGET
# ============================================================

TARGET = "final_exam_score"

X = df.drop(
    columns=[TARGET]
).copy()

y = df[TARGET].copy()


# ============================================================
# 6. SELECT ONE STUDENT
# ============================================================

student = X.iloc[
    [0]
].copy()

actual_score = float(
    y.iloc[0]
)

print("\nStudent selected:")

print(
    student.to_string(index=False)
)


# ============================================================
# 7. PREDICT
# ============================================================

print("\nCalculating prediction...")

prediction = model.predict(
    student
)

predicted_score = float(
    prediction[0]
)

print(
    f"Predicted final exam score: "
    f"{predicted_score:.2f}"
)

print(
    f"Actual final exam score: "
    f"{actual_score:.2f}"
)


# ============================================================
# 8. CALCULATE SHAP
# ============================================================

print("\nCalculating local SHAP explanation...")

explanation_df = explain_student(
    model=model,
    student_input=student,
    background_data=X,
    max_evals=19
)


# ============================================================
# 9. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("LOCAL SHAP EXPLANATION")
print("=" * 60)

print(
    explanation_df[
        [
            "Feature",
            "Feature_Value",
            "SHAP_Value",
            "Impact"
        ]
    ].to_string(index=False)
)


# ============================================================
# 10. TOP FEATURES
# ============================================================

print("\nTop 5 influencing features:")

top_features = explanation_df.head(5)

for rank, row in enumerate(
    top_features.itertuples(index=False),
    start=1
):

    print(
        f"{rank}. "
        f"{row.Feature}: "
        f"{row.SHAP_Value:+.4f} "
        f"({row.Impact})"
    )


print("\n" + "=" * 60)
print("LIVE SHAP TEST COMPLETED SUCCESSFULLY")
print("=" * 60)