import sys
from pathlib import Path

import pickle

sys.path.append(
    str(Path(__file__).parent)
)

from student_insights import (
    create_student_dataframe,
    predict_student,
    simulate_what_if,
    compare_predictions
)


# ==========================================================
# PROJECT ROOT
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ==========================================================
# TABPFN MODEL
# ==========================================================

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tabpfn_fitted_model.pkl"
)


print("=" * 60)
print("TABPFN WHAT-IF SIMULATION TEST")
print("=" * 60)


print("\nLoading model:")
print(MODEL_PATH)


if not MODEL_PATH.exists():

    raise FileNotFoundError(
        f"TabPFN model not found:\n{MODEL_PATH}"
    )


with open(
    MODEL_PATH,
    "rb"
) as file:

    model = pickle.load(file)


print("\nTabPFN model loaded successfully.")


# ==========================================================
# ORIGINAL STUDENT
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


print("\nOriginal Student:")
print(student_df)


# ==========================================================
# ORIGINAL PREDICTION
# ==========================================================

current_score = predict_student(
    model,
    student_df
)


print(
    "\nOriginal Predicted Score:",
    round(current_score, 2)
)


# ==========================================================
# WHAT-IF SCENARIO
# ==========================================================

changes = {

    "study_time_hours": 5,

    "attendance_percent": 95
}


what_if_score, simulated_student = simulate_what_if(

    model,

    student_df,

    changes
)


print("\nWhat-If Student:")
print(simulated_student)


print(
    "\nWhat-If Predicted Score:",
    round(what_if_score, 2)
)


# ==========================================================
# IMPROVEMENT
# ==========================================================

improvement = compare_predictions(

    current_score,

    what_if_score
)


print(
    "\nPredicted Change:",
    round(improvement, 2)
)


print("\n" + "=" * 60)
print("WHAT-IF SIMULATION COMPLETED")
print("=" * 60)