from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# EDUPREDICT-XAI
# STEP 21 — PERSONALIZED INTERVENTION RECOMMENDATION
# ============================================================

print("=" * 60)
print("EDUPREDICT-XAI — PERSONALIZED INTERVENTION SYSTEM")
print("=" * 60)


# ============================================================
# 1. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_PATH = PROJECT_ROOT / "results"
XAI_PATH = RESULTS_PATH / "xai"


SHAP_FILE = (
    XAI_PATH /
    "multiple_students_local_shap.csv"
)

RISK_FILE = (
    XAI_PATH /
    "student_performance_risk_analysis.csv"
)


print("\nChecking required files...")


if not SHAP_FILE.exists():
    raise FileNotFoundError(
        f"\nSHAP result file not found:\n{SHAP_FILE}\n\n"
        "Please run generate_multiple_local_shap.py first."
    )


if not RISK_FILE.exists():
    raise FileNotFoundError(
        f"\nPerformance risk file not found:\n{RISK_FILE}\n\n"
        "Please run performance_risk_analysis.py first."
    )


print("✓ SHAP results found.")
print("✓ Performance risk results found.")


# ============================================================
# 2. LOAD SHAP RESULTS
# ============================================================

print("\nLoading SHAP results...")

shap_df = pd.read_csv(
    SHAP_FILE
)

print("SHAP dataset shape:", shap_df.shape)

print("\nSHAP columns:")
print(shap_df.columns.tolist())


# ============================================================
# 3. LOAD PERFORMANCE RESULTS
# ============================================================

print("\nLoading performance categories...")

risk_df = pd.read_csv(
    RISK_FILE
)

print("Risk dataset shape:", risk_df.shape)

print("\nRisk columns:")
print(risk_df.columns.tolist())


# ============================================================
# 4. CHECK REQUIRED SHAP COLUMNS
# ============================================================

required_shap_columns = [
    "Student_Index",
    "Feature",
    "Feature_Value",
    "SHAP_Value"
]


missing_columns = [
    column
    for column in required_shap_columns
    if column not in shap_df.columns
]


if missing_columns:

    raise ValueError(
        "Required SHAP columns are missing:\n"
        + str(missing_columns)
        + "\n\nAvailable columns:\n"
        + str(shap_df.columns.tolist())
    )


print("\nRequired SHAP columns verified.")


# ============================================================
# 5. CHECK REQUIRED RISK COLUMNS
# ============================================================

required_risk_columns = [
    "Predicted_Score",
    "Performance_Category",
    "Recommendation"
]


missing_risk_columns = [
    column
    for column in required_risk_columns
    if column not in risk_df.columns
]


if missing_risk_columns:

    raise ValueError(
        "Required risk columns are missing:\n"
        + str(missing_risk_columns)
        + "\n\nAvailable columns:\n"
        + str(risk_df.columns.tolist())
    )


print("Required performance columns verified.")


# ============================================================
# 6. FEATURE-SPECIFIC INTERVENTION RULES
# ============================================================

INTERVENTION_RULES = {

    "study_time_hours": {
        "positive": (
            "Maintain the current study routine and continue "
            "consistent preparation."
        ),
        "negative": (
            "Increase structured study time and divide study "
            "sessions across the week."
        )
    },

    "attendance_percent": {
        "positive": (
            "Maintain regular attendance and continue active "
            "participation in classes."
        ),
        "negative": (
            "Improve attendance and avoid unnecessary absence "
            "from classes."
        )
    },

    "sleep_hours": {
        "positive": (
            "Maintain a consistent and healthy sleep routine."
        ),
        "negative": (
            "Review sleep habits and maintain a more consistent "
            "sleep schedule to support academic performance."
        )
    },

    "previous_grade": {
        "positive": (
            "Continue building on previous academic strengths."
        ),
        "negative": (
            "Review concepts from previous assessments and "
            "strengthen foundational knowledge."
        )
    },

    "internet_access": {
        "positive": (
            "Continue using available online educational "
            "resources effectively."
        ),
        "negative": (
            "Improve access to reliable learning resources "
            "where possible, such as institutional or library "
            "resources."
        )
    },

    "extracurricular_activities": {
        "positive": (
            "Continue balanced extracurricular participation "
            "while maintaining academic priorities."
        ),
        "negative": (
            "Review the balance between extracurricular "
            "activities and academic commitments."
        )
    },

    "part_time_job": {
        "positive": (
            "Maintain a balanced schedule between work and "
            "academic responsibilities."
        ),
        "negative": (
            "Review the balance between employment hours and "
            "study requirements."
        )
    },

    "parental_education": {
        "positive": (
            "Continue using available academic support and "
            "learning resources."
        ),
        "negative": (
            "Consider additional academic mentoring or "
            "institutional support resources."
        )
    },

    "gender": {
        "positive": (
            "No direct intervention is recommended based on "
            "gender."
        ),
        "negative": (
            "No direct intervention is recommended based on "
            "gender."
        )
    }
}


# ============================================================
# 7. GENERAL PERFORMANCE INTERVENTIONS
# ============================================================

PERFORMANCE_INTERVENTIONS = {

    "Excellent": (
        "Maintain current academic habits, continue consistent "
        "attendance and study practices, and focus on sustaining "
        "high performance."
    ),

    "Good": (
        "Maintain current academic habits while strengthening "
        "the student features that contribute positively to "
        "predicted performance."
    ),

    "At Risk": (
        "Provide targeted academic support. Prioritize "
        "attendance, structured study planning, and monitoring "
        "of the student's most influential negative factors."
    ),

    "Critical Risk": (
        "Consider immediate academic intervention, including "
        "mentoring, structured study planning, attendance "
        "monitoring, and regular progress reviews."
    )
}


# ============================================================
# 8. FUNCTION TO GENERATE STUDENT INTERVENTION
# ============================================================

def generate_student_intervention(
    student_index,
    student_shap,
    risk_row
):

    predicted_score = float(
        risk_row["Predicted_Score"]
    )

    category = str(
        risk_row["Performance_Category"]
    )

    # --------------------------------------------------------
    # Sort SHAP values
    # --------------------------------------------------------

    student_shap = student_shap.copy()

    student_shap["Absolute_SHAP"] = (
        student_shap["SHAP_Value"]
        .abs()
    )

    student_shap = student_shap.sort_values(
        by="Absolute_SHAP",
        ascending=False
    )

    # --------------------------------------------------------
    # Top positive factors
    # --------------------------------------------------------

    positive = student_shap[
        student_shap["SHAP_Value"] > 0
    ].copy()

    positive = positive.sort_values(
        by="SHAP_Value",
        ascending=False
    )

    # --------------------------------------------------------
    # Top negative factors
    # --------------------------------------------------------

    negative = student_shap[
        student_shap["SHAP_Value"] < 0
    ].copy()

    negative = negative.sort_values(
        by="SHAP_Value",
        ascending=True
    )

    # --------------------------------------------------------
    # Top 3
    # --------------------------------------------------------

    top_positive = positive.head(3)

    top_negative = negative.head(3)

    recommendations = []

    # --------------------------------------------------------
    # Performance-level recommendation
    # --------------------------------------------------------

    if category in PERFORMANCE_INTERVENTIONS:

        recommendations.append(
            PERFORMANCE_INTERVENTIONS[category]
        )

    # --------------------------------------------------------
    # Negative SHAP recommendations
    # --------------------------------------------------------

    for _, row in top_negative.iterrows():

        feature = row["Feature"]

        if feature in INTERVENTION_RULES:

            recommendation = (
                INTERVENTION_RULES[feature]["negative"]
            )

            recommendations.append(
                recommendation
            )

    # --------------------------------------------------------
    # Positive SHAP reinforcement
    # --------------------------------------------------------

    for _, row in top_positive.iterrows():

        feature = row["Feature"]

        if feature in INTERVENTION_RULES:

            recommendation = (
                INTERVENTION_RULES[feature]["positive"]
            )

            recommendations.append(
                recommendation
            )

    # --------------------------------------------------------
    # Remove duplicate recommendations
    # --------------------------------------------------------

    unique_recommendations = []

    for recommendation in recommendations:

        if recommendation not in unique_recommendations:

            unique_recommendations.append(
                recommendation
            )

    # --------------------------------------------------------
    # Convert factors to text
    # --------------------------------------------------------

    if len(top_positive) > 0:

        positive_factors = "; ".join(
            [
                f"{row['Feature']} "
                f"(SHAP={row['SHAP_Value']:.4f})"
                for _, row in top_positive.iterrows()
            ]
        )

    else:

        positive_factors = "None"


    if len(top_negative) > 0:

        negative_factors = "; ".join(
            [
                f"{row['Feature']} "
                f"(SHAP={row['SHAP_Value']:.4f})"
                for _, row in top_negative.iterrows()
            ]
        )

    else:

        negative_factors = "None"


    # --------------------------------------------------------
    # Final recommendation
    # --------------------------------------------------------

    final_recommendation = " ".join(
        unique_recommendations
    )

    return {
        "Student_Index": student_index,
        "Predicted_Score": round(
            predicted_score,
            2
        ),
        "Performance_Category": category,
        "Top_Positive_Factors": positive_factors,
        "Top_Negative_Factors": negative_factors,
        "Personalized_Intervention":
            final_recommendation
    }


# ============================================================
# 9. GENERATE RECOMMENDATIONS
# ============================================================

print("\n" + "=" * 60)
print("GENERATING PERSONALIZED INTERVENTIONS")
print("=" * 60)


recommendation_results = []


# Determine student IDs from SHAP file

student_indices = (
    shap_df["Student_Index"]
    .unique()
)


print(
    "\nStudents available for analysis:",
    len(student_indices)
)


for student_index in student_indices:

    print(
        f"\nProcessing Student {student_index}..."
    )

    student_shap = shap_df[
        shap_df["Student_Index"]
        == student_index
    ]

    # Risk file uses row order as student index
    if student_index >= len(risk_df):

        print(
            f"Warning: Student {student_index} "
            "not available in risk results."
        )

        continue


    risk_row = risk_df.iloc[
        int(student_index)
    ]


    result = generate_student_intervention(
        student_index,
        student_shap,
        risk_row
    )


    recommendation_results.append(
        result
    )


# ============================================================
# 10. CREATE FINAL DATAFRAME
# ============================================================

recommendation_df = pd.DataFrame(
    recommendation_results
)


if recommendation_df.empty:

    raise ValueError(
        "No intervention recommendations were generated."
    )


# ============================================================
# 11. DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)
print("PERSONALIZED INTERVENTION RESULTS")
print("=" * 60)


for _, row in recommendation_df.iterrows():

    print("\n" + "-" * 60)

    print(
        f"Student {row['Student_Index']}"
    )

    print(
        f"Predicted Score: "
        f"{row['Predicted_Score']}"
    )

    print(
        f"Performance Category: "
        f"{row['Performance_Category']}"
    )

    print(
        "\nTop Positive Factors:"
    )

    print(
        row["Top_Positive_Factors"]
    )

    print(
        "\nTop Negative Factors:"
    )

    print(
        row["Top_Negative_Factors"]
    )

    print(
        "\nPersonalized Intervention:"
    )

    print(
        row["Personalized_Intervention"]
    )


# ============================================================
# 12. SAVE RESULTS
# ============================================================

OUTPUT_FILE = (
    XAI_PATH /
    "personalized_intervention_recommendations.csv"
)


recommendation_df.to_csv(
    OUTPUT_FILE,
    index=False
)


print("\n" + "=" * 60)
print("STEP 21 COMPLETED SUCCESSFULLY")
print("=" * 60)


print(
    "\nStudents processed:",
    len(recommendation_df)
)


print(
    "\nFile created:"
)

print(
    OUTPUT_FILE
)


print(
    "\n✓ Prediction used"
)

print(
    "✓ Performance category used"
)

print(
    "✓ SHAP factors used"
)

print(
    "✓ Personalized interventions generated"
)

print(
    "\nStep 21 completed successfully."
)