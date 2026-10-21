from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# EDUPREDICT-XAI
# STEP 20 — PERFORMANCE RISK ANALYSIS
# ============================================================

print("=" * 60)
print("EDUPREDICT-XAI — PERFORMANCE RISK ANALYSIS")
print("=" * 60)


# ============================================================
# 1. PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "results"
XAI_PATH = RESULTS_PATH / "xai"

PREDICTION_FILE = (
    RESULTS_PATH /
    "tabpfn_predictions_with_error.csv"
)

if not PREDICTION_FILE.exists():
    raise FileNotFoundError(
        f"\nPrediction file not found:\n{PREDICTION_FILE}\n\n"
        "Please make sure Step 17 was completed."
    )

print("\nPrediction file found:")
print(PREDICTION_FILE)


# ============================================================
# 2. LOAD TABPFN PREDICTIONS
# ============================================================

df = pd.read_csv(PREDICTION_FILE)

print("\nPrediction dataset loaded.")
print("Shape:", df.shape)

print("\nColumns:")
print(df.columns.tolist())


# ============================================================
# 3. IDENTIFY ACTUAL AND PREDICTED SCORE COLUMNS
# ============================================================

print("\nSearching for prediction columns...")

possible_actual = [
    "Actual",
    "Actual_Score",
    "actual",
    "final_exam_score"
]

possible_predicted = [
    "Predicted",
    "Predicted_Score",
    "predicted",
    "TabPFN_Predicted"
]

actual_column = None
predicted_column = None

for column in possible_actual:
    if column in df.columns:
        actual_column = column
        break

for column in possible_predicted:
    if column in df.columns:
        predicted_column = column
        break


if actual_column is None:
    raise ValueError(
        "Could not identify actual score column.\n"
        f"Available columns: {df.columns.tolist()}"
    )

if predicted_column is None:
    raise ValueError(
        "Could not identify predicted score column.\n"
        f"Available columns: {df.columns.tolist()}"
    )


print("\nActual score column:")
print(actual_column)

print("\nPredicted score column:")
print(predicted_column)


# ============================================================
# 4. CREATE ANALYSIS DATAFRAME
# ============================================================

analysis_df = df[
    [
        actual_column,
        predicted_column
    ]
].copy()

analysis_df.columns = [
    "Actual_Score",
    "Predicted_Score"
]


# ============================================================
# 5. PERFORMANCE CATEGORY FUNCTION
# ============================================================

def classify_performance(score):

    if score >= 85:
        return "Excellent"

    elif score >= 70:
        return "Good"

    elif score >= 50:
        return "At Risk"

    else:
        return "Critical Risk"


analysis_df["Performance_Category"] = (
    analysis_df["Predicted_Score"]
    .apply(classify_performance)
)


# ============================================================
# 6. RECOMMENDATION FUNCTION
# ============================================================

def generate_recommendation(category):

    if category == "Excellent":
        return (
            "Maintain current study habits and continue "
            "consistent academic performance."
        )

    elif category == "Good":
        return (
            "Maintain regular study and attendance while "
            "focusing on areas with lower performance."
        )

    elif category == "At Risk":
        return (
            "Increase study time, improve attendance, and "
            "provide targeted academic support."
        )

    else:
        return (
            "Immediate academic intervention is recommended, "
            "including mentoring, attendance monitoring, and "
            "a structured study plan."
        )


analysis_df["Recommendation"] = (
    analysis_df["Performance_Category"]
    .apply(generate_recommendation)
)


# ============================================================
# 7. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("STUDENT PERFORMANCE CLASSIFICATION")
print("=" * 60)

print(
    analysis_df.to_string(index=True)
)


# ============================================================
# 8. CATEGORY DISTRIBUTION
# ============================================================

print("\n" + "=" * 60)
print("PERFORMANCE CATEGORY DISTRIBUTION")
print("=" * 60)

category_counts = (
    analysis_df["Performance_Category"]
    .value_counts()
)

print(category_counts)


# ============================================================
# 9. CATEGORY PERCENTAGES
# ============================================================

category_percentages = (
    analysis_df["Performance_Category"]
    .value_counts(normalize=True)
    .mul(100)
    .round(2)
)

print("\nCategory percentages:")

for category, percentage in category_percentages.items():

    print(
        f"{category}: {percentage}%"
    )


# ============================================================
# 10. SAVE STUDENT RISK RESULTS
# ============================================================

XAI_PATH.mkdir(
    parents=True,
    exist_ok=True
)

OUTPUT_FILE = (
    XAI_PATH /
    "student_performance_risk_analysis.csv"
)

analysis_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\nPerformance analysis saved:")
print(OUTPUT_FILE)


# ============================================================
# 11. SAVE CATEGORY SUMMARY
# ============================================================

summary_df = (
    category_counts
    .rename("Student_Count")
    .reset_index()
)

summary_df.columns = [
    "Performance_Category",
    "Student_Count"
]

summary_df["Percentage"] = (
    summary_df["Student_Count"]
    / len(analysis_df)
    * 100
).round(2)


SUMMARY_FILE = (
    XAI_PATH /
    "performance_category_summary.csv"
)

summary_df.to_csv(
    SUMMARY_FILE,
    index=False
)

print("\nCategory summary saved:")
print(SUMMARY_FILE)


# ============================================================
# 12. FINAL SUMMARY
# ============================================================

print("\n" + "=" * 60)
print("STEP 20 COMPLETED SUCCESSFULLY")
print("=" * 60)

print("\nTotal students analyzed:")
print(len(analysis_df))

print("\nPerformance categories:")
print(
    analysis_df[
        "Performance_Category"
    ].value_counts()
)

print("\nFiles created:")
print("✓ student_performance_risk_analysis.csv")
print("✓ performance_category_summary.csv")

print("\nStep 20 completed.")