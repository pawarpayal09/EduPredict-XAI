# ============================================================
# EduPredict-XAI
# Local SHAP Explanation for TabPFN
# ============================================================

import numpy as np
import pandas as pd
import shap


# ============================================================
# Create numeric representation for SHAP
# ============================================================

def create_numeric_representation(X):
    """
    Convert categorical columns to numerical codes.

    This representation is used ONLY by SHAP.
    The original TabPFN model continues to receive
    the original categorical values.
    """

    X_numeric = X.copy()
    mappings = {}

    for column in X.columns:

        if X_numeric[column].dtype == "object":

            categories = sorted(
                X_numeric[column]
                .dropna()
                .astype(str)
                .unique()
            )

            mapping = {
                category: index
                for index, category in enumerate(categories)
            }

            reverse_mapping = {
                index: category
                for category, index in mapping.items()
            }

            X_numeric[column] = (
                X_numeric[column]
                .astype(str)
                .map(mapping)
            )

            mappings[column] = {
                "forward": mapping,
                "reverse": reverse_mapping
            }

        else:

            X_numeric[column] = pd.to_numeric(
                X_numeric[column],
                errors="coerce"
            )

    return X_numeric.astype(float), mappings


# ============================================================
# Convert SHAP numeric data back to original data
# ============================================================

def convert_numeric_to_original(
    X_numeric,
    original_columns,
    mappings
):
    """
    Convert SHAP numeric representation back into the
    original categorical values expected by TabPFN.
    """

    X_original = pd.DataFrame(
        index=X_numeric.index,
        columns=original_columns
    )

    for column in original_columns:

        if column in mappings:

            reverse_mapping = mappings[column]["reverse"]

            values = []

            for value in X_numeric[column]:

                value = int(
                    np.clip(
                        round(float(value)),
                        min(reverse_mapping.keys()),
                        max(reverse_mapping.keys())
                    )
                )

                values.append(
                    reverse_mapping[value]
                )

            X_original[column] = values

        else:

            X_original[column] = X_numeric[column]

    return X_original


# ============================================================
# Local SHAP Explanation
# ============================================================

def explain_student(
    model,
    background_data,
    student_data,
    max_evals=100
):
    """
    Generate a local SHAP explanation for one student.

    Parameters
    ----------
    model:
        Trained TabPFN model.

    background_data:
        Background dataset used by SHAP.

    student_data:
        One student record to explain.

    max_evals:
        Maximum SHAP evaluations.

    Returns
    -------
    explanation_df:
        Feature-level SHAP contribution table.
    """

    print("\nPreparing data for SHAP...")

    # --------------------------------------------------------
    # Make sure inputs are DataFrames
    # --------------------------------------------------------

    if not isinstance(background_data, pd.DataFrame):
        background_data = pd.DataFrame(background_data)

    if not isinstance(student_data, pd.DataFrame):
        student_data = pd.DataFrame(student_data)

    # --------------------------------------------------------
    # Keep same column order
    # --------------------------------------------------------

    student_data = student_data[
        background_data.columns
    ].copy()

    # --------------------------------------------------------
    # Convert categorical features to numeric representation
    # --------------------------------------------------------

    background_numeric, mappings = (
        create_numeric_representation(
            background_data
        )
    )

    student_numeric = student_data.copy()

    for column in student_numeric.columns:

        if column in mappings:

            forward_mapping = mappings[column]["forward"]

            student_numeric[column] = (
                student_numeric[column]
                .astype(str)
                .map(forward_mapping)
            )

        else:

            student_numeric[column] = pd.to_numeric(
                student_numeric[column],
                errors="coerce"
            )

    student_numeric = student_numeric.astype(float)

    # --------------------------------------------------------
    # Prediction function for SHAP
    # --------------------------------------------------------

    def prediction_function(X_numeric):

        X_numeric = np.asarray(
            X_numeric,
            dtype=float
        )

        X_numeric_df = pd.DataFrame(
            X_numeric,
            columns=background_data.columns
        )

        X_original = convert_numeric_to_original(
            X_numeric_df,
            background_data.columns,
            mappings
        )

        predictions = model.predict(
            X_original
        )

        return np.asarray(
            predictions,
            dtype=float
        )

    # --------------------------------------------------------
    # Create SHAP explainer
    # --------------------------------------------------------

    print("Creating SHAP permutation explainer...")

    explainer = shap.Explainer(
        prediction_function,
        background_numeric,
        algorithm="permutation"
    )

    # --------------------------------------------------------
    # Calculate SHAP values
    # --------------------------------------------------------

    print("Calculating local SHAP values...")

    shap_result = explainer(
        student_numeric,
        max_evals=max_evals
    )

    # --------------------------------------------------------
    # Extract SHAP values
    # --------------------------------------------------------

    shap_values = np.asarray(
        shap_result.values
    )

    if shap_values.ndim == 2:
        shap_values = shap_values[0]

    # --------------------------------------------------------
    # Create explanation table
    # --------------------------------------------------------

    explanation_df = pd.DataFrame({
        "Feature": background_data.columns,
        "Feature_Value": [
            student_data.iloc[0][feature]
            for feature in background_data.columns
        ],
        "SHAP_Value": shap_values
    })

    explanation_df["Absolute_SHAP"] = (
        explanation_df["SHAP_Value"]
        .abs()
    )

    explanation_df["Impact"] = np.where(
        explanation_df["SHAP_Value"] >= 0,
        "Increases prediction",
        "Decreases prediction"
    )

    explanation_df = explanation_df.sort_values(
        by="Absolute_SHAP",
        ascending=False
    ).reset_index(drop=True)

    return explanation_df