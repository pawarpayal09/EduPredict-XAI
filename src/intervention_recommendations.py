# ============================================================
# EduPredict-XAI
# Step 26 — Personalized Intervention Recommendation Engine
# ============================================================

import pandas as pd


# ============================================================
# 1. PERFORMANCE CATEGORY
# ============================================================

def get_performance_category(score):

    score = float(score)

    if score >= 75:
        return "Excellent"

    elif score >= 60:
        return "Good"

    elif score >= 50:
        return "Average"

    elif score >= 35:
        return "Needs Improvement"

    else:
        return "At Risk"


# ============================================================
# 2. FEATURE-SPECIFIC RECOMMENDATIONS
# ============================================================

RECOMMENDATIONS = {

    "attendance_percent": {
        "low": (
            "Improve class attendance. Regular attendance can help "
            "the student understand course material and avoid missing "
            "important academic activities."
        ),
        "threshold": 75
    },

    "study_time_hours": {
        "low": (
            "Increase daily study time gradually. A consistent "
            "study schedule can provide more opportunities for "
            "revision, practice, and preparation."
        ),
        "threshold": 2
    },

    "sleep_hours": {
        "low": (
            "Maintain a healthier sleep schedule. Adequate sleep "
            "can support concentration, memory, and learning."
        ),
        "threshold": 6
    },

    "previous_grade": {
        "low": (
            "Focus on strengthening previously weak academic areas. "
            "Revision and targeted practice can help improve the "
            "foundation required for the final examination."
        ),
        "threshold": 60
    },

    "internet_access": {
        "negative": (
            "Improve access to reliable learning resources such as "
            "online lectures, educational materials, and practice "
            "resources."
        )
    },

    "extracurricular_activities": {
        "negative": (
            "Maintain a balanced extracurricular schedule so that "
            "academic preparation receives sufficient time."
        )
    },

    "part_time_job": {
        "negative": (
            "Balance part-time work with academic responsibilities. "
            "Reducing schedule conflicts may provide more time for "
            "study and examination preparation."
        )
    },

    "parental_education": {
        "negative": (
            "Consider additional academic guidance such as mentoring, "
            "peer support, tutoring, or structured study assistance."
        )
    },

    "gender": {
        "negative": (
            "No direct intervention should be based on gender. "
            "Recommendations should focus on modifiable academic "
            "and behavioral factors."
        )
    }
}


# ============================================================
# 3. GENERATE PERSONALIZED RECOMMENDATIONS
# ============================================================

def generate_recommendations(
    student,
    shap_df,
    predicted_score,
    max_recommendations=3
):

    recommendations = []

    predicted_score = float(predicted_score)

    category = get_performance_category(
        predicted_score
    )

    # --------------------------------------------------------
    # Sort by strongest negative SHAP contribution
    # --------------------------------------------------------

    negative_features = shap_df[
        shap_df["SHAP_Value"] < 0
    ].copy()

    negative_features["Absolute_SHAP"] = (
        negative_features["SHAP_Value"].abs()
    )

    negative_features = negative_features.sort_values(
        by="Absolute_SHAP",
        ascending=False
    )

    # --------------------------------------------------------
    # Examine negative features
    # --------------------------------------------------------

    for _, row in negative_features.iterrows():

        feature = row["Feature"]

        if feature not in student.index:
            continue

        value = student[feature]

        recommendation = None

        # ----------------------------------------------------
        # Attendance
        # ----------------------------------------------------

        if feature == "attendance_percent":

            if float(value) < 75:

                recommendation = (
                    RECOMMENDATIONS[feature]["low"]
                )

        # ----------------------------------------------------
        # Study time
        # ----------------------------------------------------

        elif feature == "study_time_hours":

            if float(value) < 2:

                recommendation = (
                    RECOMMENDATIONS[feature]["low"]
                )

        # ----------------------------------------------------
        # Sleep
        # ----------------------------------------------------

        elif feature == "sleep_hours":

            if float(value) < 6:

                recommendation = (
                    RECOMMENDATIONS[feature]["low"]
                )

        # ----------------------------------------------------
        # Previous grade
        # ----------------------------------------------------

        elif feature == "previous_grade":

            if float(value) < 60:

                recommendation = (
                    RECOMMENDATIONS[feature]["low"]
                )

        # ----------------------------------------------------
        # Internet access
        # ----------------------------------------------------

        elif feature == "internet_access":

            if str(value).lower() == "no":

                recommendation = (
                    RECOMMENDATIONS[feature]["negative"]
                )

        # ----------------------------------------------------
        # Extracurricular activities
        # ----------------------------------------------------

        elif feature == "extracurricular_activities":

            if str(value).lower() == "yes":

                recommendation = (
                    RECOMMENDATIONS[feature]["negative"]
                )

        # ----------------------------------------------------
        # Part-time job
        # ----------------------------------------------------

        elif feature == "part_time_job":

            if str(value).lower() == "yes":

                recommendation = (
                    RECOMMENDATIONS[feature]["negative"]
                )

        # ----------------------------------------------------
        # Parental education
        # ----------------------------------------------------

        elif feature == "parental_education":

            recommendation = (
                RECOMMENDATIONS[feature]["negative"]
            )

        # ----------------------------------------------------
        # Gender
        # ----------------------------------------------------

        elif feature == "gender":

            # Do not recommend interventions based on gender.
            recommendation = None

        # ----------------------------------------------------
        # Store recommendation
        # ----------------------------------------------------

        if recommendation is not None:

            recommendations.append({
                "Feature": feature,
                "Feature_Value": value,
                "SHAP_Value": float(row["SHAP_Value"]),
                "Recommendation": recommendation
            })

        if len(recommendations) >= max_recommendations:

            break

    # ========================================================
    # 4. FALLBACK RECOMMENDATION
    # ========================================================

    if len(recommendations) == 0:

        if predicted_score >= 75:

            recommendations.append({
                "Feature": "overall_performance",
                "Feature_Value": predicted_score,
                "SHAP_Value": 0.0,
                "Recommendation": (
                    "Maintain the current study habits and "
                    "continue consistent academic preparation."
                )
            })

        elif predicted_score >= 60:

            recommendations.append({
                "Feature": "overall_performance",
                "Feature_Value": predicted_score,
                "SHAP_Value": 0.0,
                "Recommendation": (
                    "Maintain current academic habits while "
                    "focusing on regular revision and practice."
                )
            })

        else:

            recommendations.append({
                "Feature": "overall_performance",
                "Feature_Value": predicted_score,
                "SHAP_Value": 0.0,
                "Recommendation": (
                    "The student may benefit from a structured "
                    "study plan, regular revision, and academic "
                    "guidance."
                )
            })

    # ========================================================
    # 5. CREATE RESULT DATAFRAME
    # ========================================================

    recommendations_df = pd.DataFrame(
        recommendations
    )

    return category, recommendations_df


# ============================================================
# 6. TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("STEP 26 — INTERVENTION RECOMMENDATION ENGINE")
    print("=" * 60)

    print("\nRecommendation engine loaded successfully.")

    print("\nAvailable recommendation features:")

    for feature in RECOMMENDATIONS:

        print("-", feature)

    print("\nStep 26 module test completed successfully.")