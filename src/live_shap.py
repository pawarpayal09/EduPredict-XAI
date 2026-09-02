# ============================================================
# EduPredict-XAI
# Step 24 — Live Local SHAP Explanation
# Mixed Numerical + Categorical Features
# ============================================================

import numpy as np
import pandas as pd
import shap


# ============================================================
# 1. FEATURE DEFINITIONS
# ============================================================

FEATURES = [
    "gender",
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job",
    "previous_grade"
]


# ============================================================
# 2. NUMERICAL FEATURES
# ============================================================

NUMERICAL_FEATURES = [
    "study_time_hours",
    "attendance_percent",
    "sleep_hours",
    "previous_grade"
]


# ============================================================
# 3. CATEGORICAL FEATURES
# ============================================================

CATEGORICAL_FEATURES = [
    "gender",
    "parental_education",
    "internet_access",
    "extracurricular_activities",
    "part_time_job"
]


# ============================================================
# 4. CREATE SAFE SHAP REPRESENTATION
# ============================================================

def prepare_shap_data(
    df,
    category_maps=None
):
    """
    Convert mixed categorical/numerical student data
    into a numerical representation suitable for SHAP.

    The original values are NOT changed in the actual
    TabPFN model input. This representation is only used
    internally by SHAP.
    """

    result = df[
        FEATURES
    ].copy()

    if category_maps is None:

        category_maps = {}

        for feature in CATEGORICAL_FEATURES:

            categories = sorted(
                result[feature]
                .astype(str)
                .unique()
                .tolist()
            )

            category_maps[feature] = {
                value: index
                for index, value in enumerate(categories)
            }


    for feature in CATEGORICAL_FEATURES:

        mapping = category_maps[feature]

        result[feature] = (
            result[feature]
            .astype(str)
            .map(mapping)
        )

    for feature in NUMERICAL_FEATURES:

        result[feature] = pd.to_numeric(
            result[feature],
            errors="raise"
        )

    return result.astype(float), category_maps


# ============================================================
# 5. CREATE MODEL PREDICTION FUNCTION
# ============================================================

def create_prediction_function(
    model,
    background_original,
    category_maps
):

    def prediction_function(shap_data):

        # Convert SHAP numerical array into DataFrame
        shap_df = pd.DataFrame(
            shap_data,
            columns=FEATURES
        )

        # Create the actual TabPFN input
        model_input = shap_df.copy()


        # Restore categorical values
        for feature in CATEGORICAL_FEATURES:

            reverse_mapping = {
                encoded: original
                for original, encoded
                in category_maps[feature].items()
            }

            model_input[feature] = (
                model_input[feature]
                .round()
                .astype(int)
                .map(reverse_mapping)
            )


        # Restore numerical columns
        for feature in NUMERICAL_FEATURES:

            model_input[feature] = pd.to_numeric(
                model_input[feature],
                errors="raise"
            )


        # Exact feature order
        model_input = model_input[
            FEATURES
        ]


        # TabPFN prediction
        predictions = model.predict(
            model_input
        )

        return np.asarray(
            predictions
        )


    return prediction_function


# ============================================================
# 6. MAIN LOCAL SHAP FUNCTION
# ============================================================

def explain_student(
    model,
    student_input,
    background_data,
    max_evals=19
):

    # --------------------------------------------------------
    # Validate features
    # --------------------------------------------------------

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in student_input.columns
    ]

    if missing_features:

        raise ValueError(
            "Missing features: "
            + ", ".join(missing_features)
        )


    # --------------------------------------------------------
    # Keep exact feature order
    # --------------------------------------------------------

    student_input = student_input[
        FEATURES
    ].copy()

    background_original = background_data[
        FEATURES
    ].copy()


    # --------------------------------------------------------
    # Small SHAP background
    # --------------------------------------------------------

    background_original = (
        background_original
        .sample(
            n=min(20, len(background_original)),
            random_state=42
        )
        .reset_index(drop=True)
    )


    # --------------------------------------------------------
    # Convert background to numerical SHAP representation
    # --------------------------------------------------------

    background_numeric, category_maps = (
        prepare_shap_data(
            background_original
        )
    )


    # --------------------------------------------------------
    # Convert student to same representation
    # --------------------------------------------------------

    student_numeric, _ = prepare_shap_data(
        student_input,
        category_maps=category_maps
    )


    # --------------------------------------------------------
    # Safety check
    # --------------------------------------------------------

    if student_numeric.isnull().any().any():

        raise ValueError(
            "Student input contains a categorical value "
            "that was not present in the SHAP background."
        )


    if background_numeric.isnull().any().any():

        raise ValueError(
            "Background data contains invalid categorical "
            "values."
        )


    # --------------------------------------------------------
    # Create SHAP prediction function
    # --------------------------------------------------------

    prediction_function = create_prediction_function(
        model=model,
        background_original=background_original,
        category_maps=category_maps
    )


    # --------------------------------------------------------
    # Create SHAP masker
    #
    # IMPORTANT:
    # We use purely numerical SHAP data here.
    # This avoids the previous string/numeric error.
    # --------------------------------------------------------

    masker = shap.maskers.Independent(
        background_numeric
    )


    # --------------------------------------------------------
    # Create permutation explainer
    # --------------------------------------------------------

    explainer = shap.Explainer(
        prediction_function,
        masker,
        algorithm="permutation"
    )


    # --------------------------------------------------------
    # Calculate SHAP
    # --------------------------------------------------------

    shap_explanation = explainer(
        student_numeric,
        max_evals=max_evals
    )


    # --------------------------------------------------------
    # Extract SHAP values
    # --------------------------------------------------------

    shap_values = np.asarray(
        shap_explanation.values
    )

    if shap_values.ndim == 2:

        shap_values = shap_values[0]


    # --------------------------------------------------------
    # Create result table
    # --------------------------------------------------------

    explanation_df = pd.DataFrame({

        "Feature": FEATURES,

        "Feature_Value": [
            student_input.iloc[0][feature]
            for feature in FEATURES
        ],

        "SHAP_Value": shap_values

    })


    # --------------------------------------------------------
    # Determine direction
    # --------------------------------------------------------

    explanation_df["Impact"] = np.where(
        explanation_df["SHAP_Value"] > 0,
        "Increases predicted score",
        "Decreases predicted score"
    )


    # --------------------------------------------------------
    # Absolute importance
    # --------------------------------------------------------

    explanation_df["Absolute_Impact"] = (
        explanation_df["SHAP_Value"]
        .abs()
    )


    # --------------------------------------------------------
    # Sort strongest influence first
    # --------------------------------------------------------

    explanation_df = (
        explanation_df
        .sort_values(
            by="Absolute_Impact",
            ascending=False
        )
        .reset_index(drop=True)
    )


    return explanation_df