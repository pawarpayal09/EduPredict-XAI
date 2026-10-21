from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# EDUPREDICT-XAI
# STEP 19 — MULTIPLE STUDENT SHAP ANALYSIS
# ============================================================

print("=" * 60)
print("EDUPREDICT-XAI — MULTIPLE STUDENT SHAP ANALYSIS")
print("=" * 60)


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "results"
XAI_PATH = RESULTS_PATH / "xai"

INPUT_FILE = (
    XAI_PATH /
    "multiple_students_local_shap.csv"
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(
        f"\nRequired SHAP result file not found:\n{INPUT_FILE}\n"
        "Please run generate_multiple_local_shap.py first."
    )

print("\nInput file found:")
print(INPUT_FILE)


# ============================================================
# 2. LOAD SHAP RESULTS
# ============================================================

df = pd.read_csv(INPUT_FILE)

print("\nSHAP result loaded successfully.")

print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. BASIC VALIDATION
# ============================================================

required_columns = [
    "Student_Index",
    "Feature",
    "SHAP_Value",
    "Feature_Value"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing required columns: "
        + str(missing_columns)
    )

print("\nRequired columns verified.")


# ============================================================
# 4. GLOBAL IMPORTANCE FROM LOCAL SHAP
# ============================================================

print("\n" + "=" * 60)
print("CALCULATING AVERAGE ABSOLUTE SHAP IMPORTANCE")
print("=" * 60)

global_shap = (
    df.groupby("Feature")["SHAP_Value"]
    .apply(lambda x: np.mean(np.abs(x)))
    .sort_values(ascending=False)
    .reset_index()
)

global_shap.columns = [
    "Feature",
    "Mean_Absolute_SHAP"
]

print("\nFeature importance:")
print(global_shap.to_string(index=False))


# ============================================================
# 5. SAVE GLOBAL SHAP SUMMARY
# ============================================================

global_output = (
    XAI_PATH /
    "multiple_student_global_shap_importance.csv"
)

global_shap.to_csv(
    global_output,
    index=False
)

print("\nGlobal SHAP importance saved:")
print(global_output)


# ============================================================
# 6. PLOT GLOBAL SHAP IMPORTANCE
# ============================================================

plt.figure(figsize=(10, 6))

plt.barh(
    global_shap["Feature"],
    global_shap["Mean_Absolute_SHAP"]
)

plt.xlabel(
    "Mean Absolute SHAP Value"
)

plt.ylabel(
    "Student Features"
)

plt.title(
    "Global Feature Importance from Multiple Student SHAP Analysis"
)

plt.gca().invert_yaxis()

plt.tight_layout()

global_plot = (
    XAI_PATH /
    "multiple_student_global_shap_importance.png"
)

plt.savefig(
    global_plot,
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("\nGlobal SHAP graph saved:")
print(global_plot)


# ============================================================
# 7. POSITIVE AND NEGATIVE CONTRIBUTIONS
# ============================================================

print("\n" + "=" * 60)
print("POSITIVE AND NEGATIVE SHAP CONTRIBUTIONS")
print("=" * 60)

positive = (
    df[df["SHAP_Value"] > 0]
    .groupby("Feature")["SHAP_Value"]
    .mean()
    .sort_values(ascending=False)
)

negative = (
    df[df["SHAP_Value"] < 0]
    .groupby("Feature")["SHAP_Value"]
    .mean()
    .sort_values()
)

print("\nAverage positive contribution:")
print(positive)

print("\nAverage negative contribution:")
print(negative)


# ============================================================
# 8. TOP POSITIVE INFLUENCES
# ============================================================

positive_output = (
    XAI_PATH /
    "positive_shap_contributions.csv"
)

positive.reset_index(
    name="Average_Positive_SHAP"
).to_csv(
    positive_output,
    index=False
)

print("\nPositive SHAP results saved:")
print(positive_output)


# ============================================================
# 9. TOP NEGATIVE INFLUENCES
# ============================================================

negative_output = (
    XAI_PATH /
    "negative_shap_contributions.csv"
)

negative.reset_index(
    name="Average_Negative_SHAP"
).to_csv(
    negative_output,
    index=False
)

print("\nNegative SHAP results saved:")
print(negative_output)


# ============================================================
# 10. STUDENT-WISE SHAP SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STUDENT-WISE SHAP SUMMARY")
print("=" * 60)

student_summary = (
    df.groupby("Student_Index")
    .agg(
        Total_Positive_Impact=(
            "SHAP_Value",
            lambda x: x[x > 0].sum()
        ),
        Total_Negative_Impact=(
            "SHAP_Value",
            lambda x: x[x < 0].sum()
        ),
        Total_Absolute_Impact=(
            "SHAP_Value",
            lambda x: np.abs(x).sum()
        )
    )
    .reset_index()
)

print("\nStudent SHAP summary:")
print(student_summary.to_string(index=False))


# ============================================================
# 11. SAVE STUDENT SUMMARY
# ============================================================

student_summary_output = (
    XAI_PATH /
    "student_wise_shap_summary.csv"
)

student_summary.to_csv(
    student_summary_output,
    index=False
)

print("\nStudent-wise SHAP summary saved:")
print(student_summary_output)


# ============================================================
# 12. MOST IMPORTANT FEATURE FOR EACH STUDENT
# ============================================================

print("\n" + "=" * 60)
print("MOST IMPORTANT FEATURE FOR EACH STUDENT")
print("=" * 60)

most_important = (
    df.assign(
        Absolute_SHAP=df["SHAP_Value"].abs()
    )
    .sort_values(
        ["Student_Index", "Absolute_SHAP"],
        ascending=[True, False]
    )
    .groupby("Student_Index")
    .first()
    .reset_index()
)

important_columns = [
    "Student_Index",
    "Feature",
    "Feature_Value",
    "SHAP_Value"
]

most_important = most_important[
    important_columns
]

print(
    most_important.to_string(
        index=False
    )
)


# ============================================================
# 13. SAVE MOST IMPORTANT FEATURES
# ============================================================

important_output = (
    XAI_PATH /
    "most_important_feature_per_student.csv"
)

most_important.to_csv(
    important_output,
    index=False
)

print("\nMost important feature results saved:")
print(important_output)


# ============================================================
# 14. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 19 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nStudents analyzed:",
      df["Student_Index"].nunique())

print("Features analyzed:",
      df["Feature"].nunique())

print("\nFiles created:")

print("✓ multiple_student_global_shap_importance.csv")
print("✓ multiple_student_global_shap_importance.png")
print("✓ positive_shap_contributions.csv")
print("✓ negative_shap_contributions.csv")
print("✓ student_wise_shap_summary.csv")
print("✓ most_important_feature_per_student.csv")

print("\nSTEP 19 completed.")