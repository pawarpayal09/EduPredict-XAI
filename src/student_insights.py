"""
EduPredict-XAI
Personalized Student Performance Insights

This module adds:
1. Performance categorization
2. Student risk level
3. SHAP-based explanation
4. What-if performance simulation
5. Personalized recommendations
"""

import numpy as np
import pandas as pd


# ==========================================================
# 1. PERFORMANCE CATEGORY
# ==========================================================

def get_performance_category(score):
    """
    Convert predicted exam score into a simple performance category.
    """

    score = float(score)

    if score >= 85:
        return "Excellent"

    elif score >= 70:
        return "Good"

    elif score >= 50:
        return "Average"

    else:
        return "Needs Improvement"


# ==========================================================
# 2. PERFORMANCE RISK LEVEL
# ==========================================================

def get_risk_level(score):
    """
    Determine student performance risk.
    """

    score = float(score)

    if score >= 75:
        return "Low Risk"

    elif score >= 50:
        return "Moderate Risk"

    else:
        return "High Risk"


# ==========================================================
# 3. SCORE INTERPRETATION
# ==========================================================

def get_score_message(score):

    score = float(score)

    if score >= 85:
        return (
            "The student is predicted to perform very well. "
            "The current academic and behavioral pattern is strong."
        )

    elif score >= 70:
        return (
            "The student is predicted to perform well, "
            "but there is still room for improvement."
        )

    elif score >= 50:
        return (
            "The student is predicted to have moderate performance. "
            "Improving important academic factors may increase the score."
        )

    else:
        return (
            "The student may be at risk of low performance. "
            "Additional academic support is recommended."
        )


# ==========================================================
# 4. CREATE USER INPUT DATAFRAME
# ==========================================================

def create_student_dataframe(
    gender,
    study_time_hours,
    attendance_percent,
    sleep_hours,
    parental_education,
    internet_access,
    extracurricular_activities,
    part_time_job,
    previous_grade
):
    """
    Convert application inputs into the same feature structure
    used during model training.
    """

    student_data = pd.DataFrame([{
        "gender": gender,
        "study_time_hours": study_time_hours,
        "attendance_percent": attendance_percent,
        "sleep_hours": sleep_hours,
        "parental_education": parental_education,
        "internet_access": internet_access,
        "extracurricular_activities": extracurricular_activities,
        "part_time_job": part_time_job,
        "previous_grade": previous_grade
    }])

    return student_data


# ==========================================================
# 5. GENERATE PREDICTION
# ==========================================================

def predict_student(model, student_df):
    """
    Generate final exam score prediction.
    """

    prediction = model.predict(student_df)

    score = float(np.asarray(prediction).reshape(-1)[0])

    # Keep prediction inside realistic exam-score range.
    score = max(0.0, min(100.0, score))

    return score


# ==========================================================
# 6. WHAT-IF SIMULATION
# ==========================================================

def simulate_what_if(
    model,
    student_df,
    changes
):
    """
    Perform what-if simulation.

    Example:
        {
            "study_time_hours": 5,
            "attendance_percent": 95
        }

    Only supplied features are changed.
    """

    simulated_student = student_df.copy()

    for feature, new_value in changes.items():

        if feature not in simulated_student.columns:
            raise ValueError(
                f"Unknown feature for what-if analysis: {feature}"
            )

        simulated_student.loc[
            simulated_student.index[0],
            feature
        ] = new_value

    predicted_score = predict_student(
        model,
        simulated_student
    )

    return predicted_score, simulated_student


# ==========================================================
# 7. COMPARE CURRENT VS WHAT-IF
# ==========================================================

def compare_predictions(
    current_score,
    what_if_score
):
    """
    Calculate improvement between current and simulated prediction.
    """

    improvement = float(
        what_if_score - current_score
    )

    return improvement


# ==========================================================
# 8. RULE-BASED PERSONALIZED RECOMMENDATIONS
# ==========================================================

def generate_recommendations(student_df):

    recommendations = []

    study_hours = float(
        student_df["study_time_hours"].iloc[0]
    )

    attendance = float(
        student_df["attendance_percent"].iloc[0]
    )

    sleep = float(
        student_df["sleep_hours"].iloc[0]
    )

    previous_grade = float(
        student_df["previous_grade"].iloc[0]
    )


    # ------------------------------------------------------
    # Study time
    # ------------------------------------------------------

    if study_hours < 2:

        recommendations.append(
            "Increase focused study time gradually. "
            "A consistent daily study routine may help improve performance."
        )

    elif study_hours < 4:

        recommendations.append(
            "Consider increasing focused study time slightly "
            "while maintaining a consistent study schedule."
        )


    # ------------------------------------------------------
    # Attendance
    # ------------------------------------------------------

    if attendance < 75:

        recommendations.append(
            "Attendance is relatively low. "
            "Improving class attendance should be a priority."
        )

    elif attendance < 85:

        recommendations.append(
            "Try to improve attendance because regular class participation "
            "may support better academic performance."
        )


    # ------------------------------------------------------
    # Sleep
    # ------------------------------------------------------

    if sleep < 6:

        recommendations.append(
            "Sleep duration is relatively low. "
            "Maintaining adequate sleep may support learning and concentration."
        )

    elif sleep > 10:

        recommendations.append(
            "Sleep duration is relatively high. "
            "Maintaining a balanced daily routine may be beneficial."
        )


    # ------------------------------------------------------
    # Previous grade
    # ------------------------------------------------------

    if previous_grade < 50:

        recommendations.append(
            "Previous academic performance is relatively low. "
            "Focus on strengthening fundamental concepts and revision."
        )

    elif previous_grade < 70:

        recommendations.append(
            "Review previous weak areas and regularly practice difficult topics."
        )


    # ------------------------------------------------------
    # Default recommendation
    # ------------------------------------------------------

    if len(recommendations) == 0:

        recommendations.append(
            "The current student profile appears balanced. "
            "Continue maintaining the existing academic routine."
        )


    return recommendations