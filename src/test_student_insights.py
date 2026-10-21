from student_insights import (
    get_performance_category,
    get_risk_level,
    get_score_message,
    create_student_dataframe,
    generate_recommendations
)


print("=" * 60)
print("TESTING STUDENT INSIGHTS MODULE")
print("=" * 60)


# ----------------------------------------------------------
# Test performance category
# ----------------------------------------------------------

score = 86.29

print("\nPredicted Score:", score)

print(
    "Performance Category:",
    get_performance_category(score)
)

print(
    "Risk Level:",
    get_risk_level(score)
)

print(
    "Message:",
    get_score_message(score)
)


# ----------------------------------------------------------
# Test student dataframe
# ----------------------------------------------------------

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


print("\nStudent Data:")
print(student_df)


# ----------------------------------------------------------
# Test recommendations
# ----------------------------------------------------------

recommendations = generate_recommendations(
    student_df
)


print("\nPersonalized Recommendations:")

for i, recommendation in enumerate(
    recommendations,
    start=1
):

    print(
        f"{i}. {recommendation}"
    )


print("\n" + "=" * 60)
print("STUDENT INSIGHTS MODULE TEST COMPLETED")
print("=" * 60)