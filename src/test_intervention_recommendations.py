# ============================================================
# EduPredict-XAI
# Step 26 — Test Personalized Recommendations
# ============================================================

from pathlib import Path

import pandas as pd

from intervention_recommendations import (
    generate_recommendations
)


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

XAI_PATH = (
    PROJECT_ROOT
    / "results"
    / "xai"
)


# ============================================================
# 2. FIND MULTIPLE STUDENT SHAP RESULTS
# ============================================================

SHAP_PATH = (
    XAI_PATH
    / "multiple_students_local_shap.csv"
)

if not SHAP_PATH.exists():

    raise FileNotFoundError(
        f"SHAP result not found:\n{SHAP_PATH}"
    )


# ============================================================
# 3. LOAD SHAP RESULTS
# ============================================================

shap_df = pd.read_csv(
    SHAP_PATH
)

print("=" * 60)
print("STEP 26 — PERSONALIZED INTERVENTION TEST")
print("=" * 60)

print("\nSHAP dataset loaded.")

print(
    "Shape:",
    shap_df.shape
)


# ============================================================
# 4. LOAD STUDENT DATA
# ============================================================

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_performance_feature_engineered.csv"
)

if not DATA_PATH.exists():

    raise FileNotFoundError(
        f"Dataset not found:\n{DATA_PATH}"
    )

df = pd.read_csv(
    DATA_PATH
)

print(
    "\nStudent dataset shape:",
    df.shape
)


# ============================================================
# 5. SELECT STUDENT
# ============================================================

student_index = 0

student = df.iloc[
    student_index
].drop(
    labels=["final_exam_score"]
)

student_shap = shap_df[
    shap_df["Student_Index"] == student_index
].copy()

if student_shap.empty:

    raise ValueError(
        f"No SHAP explanation found for student "
        f"{student_index}."
    )


# ============================================================
# 6. PREDICTION
# ============================================================

prediction_row = shap_df[
    shap_df["Student_Index"] == student_index
]

if "Predicted_Score" in prediction_row.columns:

    predicted_score = float(
        prediction_row["Predicted_Score"].iloc[0]
    )

else:

    # fallback for existing SHAP files
    predicted_score = float(
        df.iloc[
            student_index
        ]["final_exam_score"]
    )


# ============================================================
# 7. GENERATE RECOMMENDATIONS
# ============================================================

category, recommendations_df = (
    generate_recommendations(
        student=student,
        shap_df=student_shap,
        predicted_score=predicted_score,
        max_recommendations=3
    )
)


# ============================================================
# 8. DISPLAY RESULTS
# ============================================================

print("\nStudent index:")
print(student_index)

print(
    "\nPredicted score:",
    round(predicted_score, 2)
)

print(
    "\nPerformance category:",
    category
)

print("\nPersonalized recommendations:")
print("-" * 60)

for i, row in enumerate(
    recommendations_df.itertuples(
        index=False
    ),
    start=1
):

    print(
        f"\n{i}. Feature: {row.Feature}"
    )

    print(
        f"   Value: {row.Feature_Value}"
    )

    print(
        f"   SHAP contribution: "
        f"{row.SHAP_Value:.4f}"
    )

    print(
        f"   Recommendation: "
        f"{row.Recommendation}"
    )


# ============================================================
# 9. SAVE RESULT
# ============================================================

output_path = (
    XAI_PATH
    / "student_0_intervention_recommendations.csv"
)

recommendations_df.to_csv(
    output_path,
    index=False
)

print("\n" + "=" * 60)
print("STEP 26 COMPLETED SUCCESSFULLY")
print("=" * 60)

print(
    "\nRecommendation file saved:"
)

print(output_path)