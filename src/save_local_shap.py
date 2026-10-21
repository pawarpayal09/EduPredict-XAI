# ============================================================
# EduPredict-XAI
# STEP 18 — LOCAL SHAP EXPLANATION
# ============================================================

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap


# ============================================================
# 1. PROJECT PATHS
# ============================================================

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


# ============================================================
# 2. TARGET
# ============================================================

TARGET = "final_exam_score"


# ============================================================
# 3. BASIC INFORMATION
# ============================================================

print("=" * 60)
print("EDUPREDICT-XAI — LOCAL SHAP EXPLANATION")
print("=" * 60)


# ============================================================
# 4. CHECK REQUIRED FILES
# ============================================================

print("\nChecking required files...")

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found:\n{DATA_PATH}"
    )

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"\nTabPFN model not found:\n{MODEL_PATH}"
    )

print("Dataset found.")
print("Model found.")


# ============================================================
# 5. LOAD DATASET
# ============================================================

print("\nLoading dataset...")

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)

if TARGET not in df.columns:
    raise KeyError(
        f"Target column '{TARGET}' was not found."
    )


# ============================================================
# 6. SEPARATE FEATURES AND TARGET
# ============================================================

X = df.drop(
    columns=[TARGET]
).copy()

y = df[TARGET].copy()

print("Feature count:", X.shape[1])

print("\nFeatures:")

for i, feature in enumerate(
    X.columns,
    start=1
):
    print(f"{i}. {feature}")


# ============================================================
# 7. LOAD TRAINED TABPFN MODEL
# ============================================================

print("\nLoading trained TabPFN model...")

with open(
    MODEL_PATH,
    "rb"
) as f:
    tabpfn_model = pickle.load(f)

print("TabPFN model loaded successfully.")


# ============================================================
# 8. SELECT STUDENT
# ============================================================
#
# For now we explain the first student in the dataset.
#
# Later we can change this to:
# - student ID
# - user input
# - a new student's data
#

student_index = 0

student = X.iloc[
    [student_index]
].copy()

print("\nStudent selected:")
print(student.to_string(index=False))


# ============================================================
# 9. CREATE NUMERICAL REPRESENTATION FOR SHAP
# ============================================================
#
# SHAP's permutation masker performs numerical comparisons.
# Our dataset contains categorical strings such as:
#
# Female / Male
# Yes / No
# High School / Bachelor's etc.
#
# Therefore, SHAP cannot directly operate on the original
# mixed string/numerical dataframe.
#
# We create a numerical representation ONLY for SHAP.
#
# IMPORTANT:
# The TabPFN model still receives the ORIGINAL feature values.
# We are NOT changing the trained model or project dataset.
# ============================================================

X_shap = X.copy()

category_maps = {}

categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    include=[np.number]
).columns.tolist()

print("\nNumerical features:")
print(numeric_columns)

print("\nCategorical features:")
print(categorical_columns)


# ============================================================
# 10. ENCODE CATEGORICAL FEATURES
# ============================================================

for column in categorical_columns:

    values = (
        X_shap[column]
        .astype(str)
        .fillna("Missing")
    )

    categories = sorted(
        values.unique().tolist()
    )

    category_maps[column] = {
        category: index
        for index, category in enumerate(categories)
    }

    X_shap[column] = values.map(
        category_maps[column]
    ).astype(float)


# Make sure numerical columns are numeric

for column in numeric_columns:

    X_shap[column] = pd.to_numeric(
        X_shap[column],
        errors="coerce"
    )


# Fill any numerical missing values

X_shap = X_shap.fillna(
    X_shap.median(numeric_only=True)
)

# Any remaining missing values

X_shap = X_shap.fillna(0)


# ============================================================
# 11. CREATE FUNCTION TO CONVERT SHAP DATA BACK
# ============================================================
#
# SHAP works with numerical data.
#
# Before sending data to TabPFN, we convert categorical
# numerical codes back to their original text values.
# ============================================================

def shap_to_original(
    data
):

    data = np.asarray(data)

    converted = pd.DataFrame(
        data,
        columns=X.columns
    )

    # Convert categorical features back

    for column in categorical_columns:

        reverse_map = {
            value: key
            for key, value
            in category_maps[column].items()
        }

        converted[column] = (
            converted[column]
            .round()
            .astype(int)
            .clip(
                lower=0,
                upper=len(reverse_map) - 1
            )
            .map(reverse_map)
        )

    # Convert numerical columns

    for column in numeric_columns:

        converted[column] = pd.to_numeric(
            converted[column],
            errors="coerce"
        )

    return converted


# ============================================================
# 12. CREATE TABPFN PREDICTION WRAPPER
# ============================================================
#
# SHAP calls this function.
#
# Input:
# numerical SHAP representation
#
# Output:
# TabPFN predictions
# ============================================================

def model_predict(
    data
):

    original_data = shap_to_original(
        data
    )

    predictions = tabpfn_model.predict(
        original_data
    )

    return np.asarray(
        predictions,
        dtype=float
    )


# ============================================================
# 13. CHECK PREDICTION BEFORE SHAP
# ============================================================

print("\nTesting TabPFN prediction...")

student_prediction = tabpfn_model.predict(
    student
)

student_prediction = float(
    np.asarray(
        student_prediction
    ).reshape(-1)[0]
)

print(
    f"Predicted final exam score: "
    f"{student_prediction:.2f}"
)


# ============================================================
# 14. PREPARE SHAP BACKGROUND DATA
# ============================================================
#
# We use a small background sample so that the local
# explanation does not take unnecessarily long on CPU.
# ============================================================

BACKGROUND_SIZE = min(
    30,
    len(X_shap)
)

background = X_shap.sample(
    n=BACKGROUND_SIZE,
    random_state=42
)

student_shap = X_shap.iloc[
    [student_index]
]


print("\nSHAP background size:")
print(BACKGROUND_SIZE)


# ============================================================
# 15. CREATE SHAP EXPLAINER
# ============================================================

print("\nCreating SHAP permutation explainer...")

explainer = shap.Explainer(
    model_predict,
    background,
    algorithm="permutation"
)

print("SHAP explainer created successfully.")


# ============================================================
# 16. CALCULATE LOCAL SHAP VALUES
# ============================================================
#
# There are 9 features in our current dataset.
#
# 2 * number_of_features + 1
# = 2 * 9 + 1
# = 19
#
# This keeps the calculation controlled on CPU.
# ============================================================

MAX_EVALS = (
    2 * X.shape[1]
) + 1

print("\nCalculating local SHAP explanation...")
print("Number of features:", X.shape[1])
print("Maximum evaluations:", MAX_EVALS)

shap_values = explainer(
    student_shap,
    max_evals=MAX_EVALS
)

print("\nLocal SHAP calculation completed successfully.")


# ============================================================
# 17. EXTRACT SHAP VALUES
# ============================================================

values = np.asarray(
    shap_values.values
)

if values.ndim == 2:
    values = values[0]

values = values.reshape(-1)

base_value = np.asarray(
    shap_values.base_values
).reshape(-1)[0]


# ============================================================
# 18. CREATE EXPLANATION TABLE
# ============================================================

explanation_df = pd.DataFrame({
    "Feature": X.columns,
    "Feature_Value": [
        student.iloc[0][feature]
        for feature in X.columns
    ],
    "SHAP_Value": values
})


# ============================================================
# 19. DETERMINE CONTRIBUTION DIRECTION
# ============================================================

explanation_df[
    "Contribution"
] = np.where(
    explanation_df["SHAP_Value"] > 0,
    "Increases predicted score",
    "Decreases predicted score"
)


# ============================================================
# 20. SORT BY ABSOLUTE IMPACT
# ============================================================

explanation_df[
    "Absolute_SHAP"
] = explanation_df[
    "SHAP_Value"
].abs()

explanation_df = (
    explanation_df
    .sort_values(
        by="Absolute_SHAP",
        ascending=False
    )
    .reset_index(drop=True)
)


# ============================================================
# 21. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("LOCAL SHAP EXPLANATION")
print("=" * 60)

print(
    f"\nStudent index: {student_index}"
)

print(
    f"Predicted final exam score: "
    f"{student_prediction:.2f}"
)

print(
    f"SHAP base value: "
    f"{base_value:.2f}"
)

print("\nFeature contributions:")

print(
    explanation_df[
        [
            "Feature",
            "Feature_Value",
            "SHAP_Value",
            "Contribution"
        ]
    ].to_string(index=False)
)


# ============================================================
# 22. SAVE LOCAL SHAP RESULT
# ============================================================

output_path = (
    RESULTS_PATH
    / "local_shap_explanation_student_0.csv"
)

explanation_df.to_csv(
    output_path,
    index=False
)


# ============================================================
# 23. SAVE SUMMARY TEXT
# ============================================================

summary_path = (
    RESULTS_PATH
    / "local_shap_summary_student_0.txt"
)

with open(
    summary_path,
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "EduPredict-XAI — Local SHAP Explanation\n"
    )

    f.write("=" * 60 + "\n\n")

    f.write(
        f"Student index: {student_index}\n"
    )

    f.write(
        f"Predicted final exam score: "
        f"{student_prediction:.4f}\n"
    )

    f.write(
        f"SHAP base value: "
        f"{base_value:.4f}\n\n"
    )

    f.write(
        "Feature Contributions:\n"
    )

    f.write(
        explanation_df[
            [
                "Feature",
                "Feature_Value",
                "SHAP_Value",
                "Contribution"
            ]
        ].to_string(index=False)
    )


# ============================================================
# 24. FINAL MESSAGE
# ============================================================

print("\n" + "=" * 60)
print("LOCAL SHAP EXPLANATION COMPLETED")
print("=" * 60)

print("\nFiles created:")

print(
    "✓",
    output_path
)

print(
    "✓",
    summary_path
)

print("\nStudent prediction:")
print(
    f"Final Exam Score = "
    f"{student_prediction:.2f}"
)

print("\nTop influencing features:")

for _, row in explanation_df.head(5).iterrows():

    print(
        f"- {row['Feature']}: "
        f"{row['SHAP_Value']:.4f} "
        f"({row['Contribution']})"
    )

print("\nDone.")