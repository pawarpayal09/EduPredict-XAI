# ============================================================
# EduPredict-XAI
# STEP 25–30 — Explainable Student Performance Application
# ============================================================

import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import streamlit as st

# ============================================================
# PROJECT ROOT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Add project root to Python path so that
# "src" can be imported when running:
# streamlit run app/App.py

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ============================================================
# PROJECT MODULES
# ============================================================

from src.intervention_recommendations import generate_recommendations

# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="EduPredict-XAI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================
# 2. CUSTOM CSS — FRONTEND ONLY
# ============================================================

st.markdown(
    """
    <style>

    /* -------------------------------------------------------
       GLOBAL
    ------------------------------------------------------- */

    .main {
        padding-top: 1.5rem;
    }

    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* -------------------------------------------------------
       MAIN TITLE
    ------------------------------------------------------- */

    .main-title {
        text-align: center;
        font-size: 2.6rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
        letter-spacing: -0.5px;
    }

    .main-subtitle {
        text-align: center;
        font-size: 1.1rem;
        opacity: 0.75;
        margin-bottom: 1.5rem;
    }


    /* -------------------------------------------------------
       SECTION TITLES
    ------------------------------------------------------- */

    .section-title {
        font-size: 1.45rem;
        font-weight: 750;
        margin-top: 0.8rem;
        margin-bottom: 0.9rem;
    }

    .section-description {
        opacity: 0.72;
        margin-bottom: 1.2rem;
        line-height: 1.6;
    }


    /* -------------------------------------------------------
       INPUT CARD
    ------------------------------------------------------- */

    .input-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 16px;
        padding: 1.2rem 1.3rem 0.7rem 1.3rem;
        margin-bottom: 1rem;
        background: rgba(128, 128, 128, 0.035);
    }


    /* -------------------------------------------------------
       RESULT CARDS
    ------------------------------------------------------- */

    .result-card {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 16px;
        padding: 1.3rem;
        text-align: center;
        min-height: 130px;
        background: rgba(128, 128, 128, 0.035);
    }

    .result-label {
        font-size: 0.9rem;
        opacity: 0.7;
        margin-bottom: 0.5rem;
    }

    .result-value {
        font-size: 2rem;
        font-weight: 800;
    }


    /* -------------------------------------------------------
       FACTOR CARDS
    ------------------------------------------------------- */

    .factor-card {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 1rem;
        margin-bottom: 0.75rem;
        background: rgba(128, 128, 128, 0.035);
    }

    .factor-name {
        font-size: 1rem;
        font-weight: 750;
        margin-bottom: 0.3rem;
    }

    .factor-value {
        font-size: 0.9rem;
        opacity: 0.72;
        margin-bottom: 0.3rem;
    }

    .factor-effect-positive {
        font-weight: 650;
    }

    .factor-effect-negative {
        font-weight: 650;
    }


    /* -------------------------------------------------------
       RECOMMENDATION CARDS
    ------------------------------------------------------- */

    .recommendation-card {
        border: 1px solid rgba(128, 128, 128, 0.18);
        border-radius: 14px;
        padding: 1rem 1.1rem;
        margin-bottom: 0.75rem;
        background: rgba(128, 128, 128, 0.035);
    }

    .recommendation-number {
        font-weight: 800;
        margin-bottom: 0.35rem;
    }


    /* -------------------------------------------------------
       ACTION SUMMARY
    ------------------------------------------------------- */

    .action-summary {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 16px;
        padding: 1.3rem;
        margin-top: 0.5rem;
        background: rgba(128, 128, 128, 0.035);
        line-height: 1.7;
    }


    /* -------------------------------------------------------
       BUTTON
    ------------------------------------------------------- */

    div.stButton {
        display: flex;
        justify-content: center;
    }

    div.stButton > button {
        width: auto !important;
        min-width: 300px;
        max-width: 420px;
        padding: 0.65rem 1.5rem;
        border-radius: 10px;
        font-size: 1rem;
        font-weight: 700;
    }


    /* -------------------------------------------------------
       DATAFRAME
    ------------------------------------------------------- */

    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }


    /* -------------------------------------------------------
       METRIC
    ------------------------------------------------------- */

    [data-testid="stMetric"] {
        padding: 0.8rem;
    }


    /* -------------------------------------------------------
       DIVIDERS
    ------------------------------------------------------- */

    hr {
        margin-top: 1.5rem;
        margin-bottom: 1.5rem;
    }


    /* -------------------------------------------------------
       FOOTER
    ------------------------------------------------------- */

    .footer {
        text-align: center;
        opacity: 0.55;
        font-size: 0.5rem;
        margin-top: 2.5rem;
        padding-top: 1rem;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "tabpfn_fitted_model.pkl"
)

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "student_performance_feature_engineered.csv"
)


# ============================================================
# 4. APPLICATION HEADER
# ============================================================

st.markdown(
    '<div class="main-title">🎓 EduPredict-XAI</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="main-subtitle">'
    'Explainable Student Performance Prediction System'
    '</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div style="
        text-align:center;
        opacity:0.72;
        max-width:850px;
        margin:auto;
        line-height:1.7;
    ">
    Predict a student's expected final exam score using the trained
    TabPFN model and understand <b>why</b> the model made that prediction
    using SHAP-based explainability.
    </div>
    """,
    unsafe_allow_html=True
)

st.divider()


# ============================================================
# 5. LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"TabPFN model not found:\n{MODEL_PATH}"
        )

    with open(
        MODEL_PATH,
        "rb"
    ) as file:

        return pickle.load(file)


# ============================================================
# 6. LOAD DATASET
# ============================================================

@st.cache_data
def load_dataset():

    if not DATA_PATH.exists():

        raise FileNotFoundError(
            f"Dataset not found:\n{DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# 7. LOAD MODEL AND DATA
# ============================================================

try:

    tabpfn_model = load_model()
    df = load_dataset()

except Exception as error:

    st.error(
        "Unable to load required project files."
    )

    st.exception(error)

    st.stop()


# ============================================================
# 8. DEFINE FEATURES
# ============================================================

TARGET = "final_exam_score"

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
# 9. VALIDATE DATASET COLUMNS
# ============================================================

missing_features = [
    feature
    for feature in FEATURES
    if feature not in df.columns
]

if missing_features:

    st.error(
        "The processed dataset is missing required feature columns:"
    )

    st.code(
        ", ".join(missing_features)
    )

    st.stop()


# ============================================================
# 10. STUDENT INFORMATION
# ============================================================

st.markdown(
    '<div class="section-title">📝 Student Information</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Enter the student information below. The model will use these '
    'nine factors to estimate the final exam score.'
    '</div>',
    unsafe_allow_html=True
)


# ============================================================
# LEFT + RIGHT INPUT COLUMNS
# ============================================================

left_col, right_col = st.columns(
    2,
    gap="small"
)


# ============================================================
# LEFT COLUMN
#
# 1. Gender
# 2. Internet Access
# 3. Extracurricular Activities
# 4. Part-time Job
# ============================================================

with left_col:
    gender = st.selectbox(
        "👤 Gender",
        [
            "Female",
            "Male"
        ]
    )

    internet_access = st.selectbox(
        "🌐 Internet Access",
        [
            "Yes",
            "No"
        ]
    )

    extracurricular_activities = st.selectbox(
        "🎯 Extracurricular Activities",
        [
            "Yes",
            "No"
        ]
    )

    part_time_job = st.selectbox(
        "💼 Part-time Job",
        [
            "Yes",
            "No"
        ]
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# RIGHT COLUMN
#
# 1. Study Time
# 2. Attendance
# 3. Sleep Hours
# 4. Previous Grade
# ============================================================

with right_col:

    study_time_hours = st.number_input(
        "📚 Study Time (hours)",
        min_value=0.0,
        max_value=24.0,
        value=2.0,
        step=0.5
    )

    attendance_percent = st.number_input(
        "📅 Attendance (%)",
        min_value=0.0,
        max_value=100.0,
        value=75.0,
        step=1.0
    )

    sleep_hours = st.number_input(
        "😴 Sleep Hours",
        min_value=0.0,
        max_value=24.0,
        value=7.0,
        step=0.5
    )

    previous_grade = st.number_input(
        "📈 Previous Grade",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        step=1.0
    )

    st.markdown(
        '</div>',
        unsafe_allow_html=True
    )


# ============================================================
# PARENTAL EDUCATION — CENTERED
# ============================================================

st.markdown(
    """
    <div style="
        text-align:center;
        margin-top:0.1em;
        margin-bottom:0.1rem;
        font-size:1.0rem;
    ">
        🎓 Parental Education
    </div>
    """,
    unsafe_allow_html=True
)

parental_left, parental_center, parental_right = st.columns(
    [1, 2, 1]
)

with parental_center:

    parental_education = st.selectbox(
        "Parental Education",
        [
            "High School",
            "Bachelors",
            "Masters",
            "PhD"
        ],
        label_visibility="collapsed"
    )


# ============================================================
# PREDICT BUTTON — CENTERED BELOW PARENTAL EDUCATION
# ============================================================

st.markdown(
    "<div style='height: 0.8rem;'></div>",
    unsafe_allow_html=True
)

button_left, button_center, button_right = st.columns(
    [1, 2, 1]
)

with button_center:

    predict_button = st.button(
        "🔮 Predict & Explain Student Performance",
        type="primary",
        use_container_width=True
    )

# ============================================================
# 11. HELPER FUNCTIONS
# ============================================================

def get_category(score):

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


def generate_recommendation(
    feature,
    value,
    shap_value
):

    recommendations = []

    if feature == "attendance_percent":

        if shap_value < 0:

            recommendations.append(
                "Improve attendance because the current attendance "
                "level is negatively influencing the prediction."
            )

        elif value >= 90:

            recommendations.append(
                "Maintain your excellent attendance level."
            )

        else:

            recommendations.append(
                "Try to increase attendance consistently."
            )

    elif feature == "study_time_hours":

        if shap_value < 0:

            recommendations.append(
                "Increase daily study time gradually and maintain "
                "a consistent study schedule."
            )

        else:

            recommendations.append(
                "Continue maintaining your current study routine."
            )

    elif feature == "previous_grade":

        if shap_value < 0:

            recommendations.append(
                "Review weak areas from previous assessments "
                "to strengthen your academic foundation."
            )

        else:

            recommendations.append(
                "Use your previous academic performance as a "
                "foundation for further improvement."
            )

    elif feature == "sleep_hours":

        if shap_value < 0:

            recommendations.append(
                "Maintain a healthier and more consistent sleep "
                "schedule to support academic performance."
            )

        else:

            recommendations.append(
                "Continue maintaining a consistent sleep routine."
            )

    elif feature == "internet_access":

        if str(value) == "No":

            recommendations.append(
                "Access to reliable learning resources may help "
                "support your academic preparation."
            )

        else:

            recommendations.append(
                "Use internet access effectively for educational "
                "resources and study materials."
            )

    elif feature == "extracurricular_activities":

        recommendations.append(
            "Maintain a healthy balance between extracurricular "
            "activities and academic responsibilities."
        )

    elif feature == "part_time_job":

        if str(value) == "Yes":

            recommendations.append(
                "Balance work responsibilities carefully with "
                "study time and academic preparation."
            )

        else:

            recommendations.append(
                "Use the available time effectively for academic preparation."
            )

    elif feature == "parental_education":

        recommendations.append(
            "Use available academic guidance and learning resources "
            "to strengthen your preparation."
        )

    elif feature == "gender":

        recommendations.append(
            "This demographic feature should not be treated as an "
            "actionable intervention factor."
        )

    return recommendations


# ============================================================
# 12. PREDICTION + SHAP
# ============================================================

if predict_button:

    try:

        # ====================================================
        # CREATE STUDENT INPUT
        # ====================================================

        student_input = pd.DataFrame({

            "gender": [
                gender
            ],

            "study_time_hours": [
                study_time_hours
            ],

            "attendance_percent": [
                attendance_percent
            ],

            "sleep_hours": [
                sleep_hours
            ],

            "parental_education": [
                parental_education
            ],

            "internet_access": [
                internet_access
            ],

            "extracurricular_activities": [
                extracurricular_activities
            ],

            "part_time_job": [
                part_time_job
            ],

            "previous_grade": [
                previous_grade
            ]
        })


        # ====================================================
        # EXACT FEATURE ORDER
        # ====================================================

        student_input = student_input[
            FEATURES
        ]


        # ====================================================
        # TABPFN PREDICTION
        # ====================================================

        with st.spinner(
            "Generating student performance prediction..."
        ):

            prediction = tabpfn_model.predict(
                student_input
            )

        predicted_score = float(
            np.asarray(
                prediction
            ).reshape(-1)[0]
        )

        predicted_score = max(
            0.0,
            min(
                100.0,
                predicted_score
            )
        )

        category = get_category(
            predicted_score
        )


        # ====================================================
        # PREDICTION RESULT
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">📊 Prediction Result</div>',
            unsafe_allow_html=True
        )

        result_col1, result_col2 = st.columns(
            2,
            gap="large"
        )

        with result_col1:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Predicted Final Exam Score
                    </div>
                    <div class="result-value">
                        {predicted_score:.2f}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with result_col2:

            st.markdown(
                f"""
                <div class="result-card">
                    <div class="result-label">
                        Performance Category
                    </div>
                    <div class="result-value">
                        {category}
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )

        # ====================================================
        # CATEGORY MESSAGE
        # ====================================================
        st.divider()

        if category == "Excellent":

            st.success(
                "Excellent predicted performance! "
                "Continue maintaining the current academic habits."
            )

        elif category == "Good":

            st.success(
                "Good predicted performance. "
                "There is still room for further improvement."
            )

        elif category == "Average":

            st.warning(
                "Average predicted performance. "
                "Targeted academic improvement is recommended."
            )

        elif category == "Needs Improvement":

            st.warning(
                "The prediction indicates that improvement "
                "may be required in several areas."
            )

        else:

            st.error(
                "The prediction indicates a higher academic risk. "
                "Early intervention is recommended."
            )

        # ====================================================
        # SHAP EXPLANATION
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '🔍 Why did the model make this prediction?'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'SHAP explains how each student feature contributed '
            'to the predicted final exam score. Positive SHAP '
            'values push the prediction upward, while negative '
            'values push it downward.'
            '</div>',
            unsafe_allow_html=True
        )


        NUMERICAL_FEATURES = [
            "study_time_hours",
            "attendance_percent",
            "sleep_hours",
            "previous_grade"
        ]

        CATEGORICAL_FEATURES = [
            "gender",
            "parental_education",
            "internet_access",
            "extracurricular_activities",
            "part_time_job"
        ]


        # ====================================================
        # CREATE SHAP ENCODING MAPS
        # ====================================================

        shap_background = df[
            FEATURES
        ].copy()

        shap_student = student_input[
            FEATURES
        ].copy()

        categorical_maps = {}


        for feature in CATEGORICAL_FEATURES:

            categories = sorted(
                shap_background[
                    feature
                ]
                .astype(str)
                .dropna()
                .unique()
                .tolist()
            )

            category_to_number = {
                category: float(index)
                for index, category
                in enumerate(categories)
            }

            categorical_maps[
                feature
            ] = category_to_number

            shap_background[
                feature
            ] = (
                shap_background[
                    feature
                ]
                .astype(str)
                .map(
                    category_to_number
                )
            )

            shap_student[
                feature
            ] = (
                shap_student[
                    feature
                ]
                .astype(str)
                .map(
                    category_to_number
                )
            )


        # ====================================================
        # CONVERT NUMERICAL COLUMNS
        # ====================================================

        for feature in NUMERICAL_FEATURES:

            shap_background[
                feature
            ] = pd.to_numeric(
                shap_background[
                    feature
                ],
                errors="coerce"
            )

            shap_student[
                feature
            ] = pd.to_numeric(
                shap_student[
                    feature
                ],
                errors="coerce"
            )


        # ====================================================
        # CLEAN SHAP BACKGROUND
        # ====================================================

        shap_background = (
            shap_background
            .replace(
                [
                    np.inf,
                    -np.inf
                ],
                np.nan
            )
            .dropna()
        )

        if len(
            shap_background
        ) == 0:

            raise ValueError(
                "No valid rows are available for the SHAP "
                "background dataset."
            )

        background_size = min(
            30,
            len(shap_background)
        )

        shap_background = (
            shap_background
            .sample(
                n=background_size,
                random_state=42
            )
        )


        # ====================================================
        # REVERSE CATEGORY MAPS
        # ====================================================

        reverse_maps = {}

        for feature in CATEGORICAL_FEATURES:

            reverse_maps[
                feature
            ] = {

                number: category

                for category, number
                in categorical_maps[
                    feature
                ].items()

            }


        # ====================================================
        # SHAP PREDICTION FUNCTION
        # ====================================================

        def shap_model_predict(
            encoded_data
        ):

            if isinstance(
                encoded_data,
                pd.DataFrame
            ):

                encoded_df = (
                    encoded_data.copy()
                )

            else:

                encoded_df = pd.DataFrame(
                    encoded_data,
                    columns=FEATURES
                )


            # Decode categorical columns

            for feature in CATEGORICAL_FEATURES:

                numeric_values = pd.to_numeric(
                    encoded_df[
                        feature
                    ],
                    errors="coerce"
                ).round()

                encoded_df[
                    feature
                ] = numeric_values.map(
                    reverse_maps[
                        feature
                    ]
                )


            # Numerical columns

            for feature in NUMERICAL_FEATURES:

                encoded_df[
                    feature
                ] = pd.to_numeric(
                    encoded_df[
                        feature
                    ],
                    errors="coerce"
                )


            encoded_df = (
                encoded_df[
                    FEATURES
                ]
            )


            if encoded_df.isnull().any().any():

                raise ValueError(
                    "SHAP generated an invalid encoded value "
                    "that could not be mapped back to the "
                    "original feature format."
                )


            return np.asarray(
                tabpfn_model.predict(
                    encoded_df
                )
            )


        # ====================================================
        # CREATE SHAP EXPLAINER
        # ====================================================

        with st.spinner(
            "Calculating SHAP explanation..."
        ):

            explainer = shap.Explainer(
                shap_model_predict,
                shap_background,
                algorithm="permutation"
            )

            shap_explanation = explainer(
                shap_student,
                max_evals=(
                    2 * len(FEATURES) + 1
                )
            )


        st.success(
            "SHAP explanation calculated successfully."
        )


        # ====================================================
        # EXTRACT SHAP VALUES
        # ====================================================

        shap_values = np.asarray(
            shap_explanation.values
        )

        if shap_values.ndim == 2:

            shap_values = shap_values[0]

        elif shap_values.ndim == 3:

            shap_values = (
                shap_values[
                    0,
                    :,
                    0
                ]
            )

        shap_values = np.asarray(
            shap_values,
            dtype=float
        ).reshape(-1)


        if len(
            shap_values
        ) != len(FEATURES):

            raise ValueError(
                "SHAP returned an unexpected number "
                "of feature values. "
                f"Expected {len(FEATURES)}, "
                f"got {len(shap_values)}."
            )


        # ====================================================
        # CREATE SHAP RESULT TABLE
        # ====================================================

        shap_df = pd.DataFrame({

            "Feature": FEATURES,

            "Value": [
                student_input.iloc[
                    0
                ][feature]

                for feature in FEATURES
            ],

            "SHAP_Value": shap_values

        })

        shap_df[
            "Absolute_SHAP"
        ] = (
            shap_df[
                "SHAP_Value"
            ].abs()
        )

        shap_df = (
            shap_df
            .sort_values(
                "Absolute_SHAP",
                ascending=False
            )
            .reset_index(
                drop=True
            )
        )


        # ====================================================
        # FEATURE CONTRIBUTIONS
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📌 Feature Contributions'
            '</div>',
            unsafe_allow_html=True
        )

        display_shap = shap_df[
            [
                "Feature",
                "Value",
                "SHAP_Value"
            ]
        ].copy()

        display_shap[
            "Effect"
        ] = (
            display_shap[
                "SHAP_Value"
            ].apply(
                lambda x:
                "⬆️ Increases prediction"
                if x > 0
                else
                "⬇️ Decreases prediction"
            )
        )

        st.dataframe(
            display_shap,
            use_container_width=True,
            hide_index=True
        )


        # ====================================================
        # SHAP BAR CHART
        # ====================================================

        st.markdown(
            '<div class="section-title">'
            '📊 Feature Impact on Prediction'
            '</div>',
            unsafe_allow_html=True
        )

        chart_df = (
            shap_df
            .sort_values(
                "SHAP_Value"
            )
        )

        st.bar_chart(
            chart_df.set_index(
                "Feature"
            )[
                "SHAP_Value"
            ]
        )


        # ====================================================
        # MOST INFLUENTIAL FACTORS
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '⭐ Most Influential Factors'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'These are the five features that had the largest '
            'impact on this individual prediction according '
            'to SHAP.'
            '</div>',
            unsafe_allow_html=True
        )

        top_features = shap_df.head(5)


        for rank, (_, row) in enumerate(
            top_features.iterrows(),
            start=1
        ):

            feature = row[
                "Feature"
            ]

            value = row[
                "Value"
            ]

            shap_value = float(
                row[
                    "SHAP_Value"
                ]
            )


            if shap_value > 0:

                st.markdown(
                    f"""
                    <div class="factor-card">
                        <div class="factor-name">
                            #{rank} &nbsp; {feature}
                        </div>
                        <div class="factor-value">
                            Current value: <b>{value}</b>
                        </div>
                        <div class="factor-effect-positive">
                            ⬆️ Increased the prediction by
                            approximately
                            <b>{shap_value:.2f}</b> points.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            else:

                st.markdown(
                    f"""
                    <div class="factor-card">
                        <div class="factor-name">
                            #{rank} &nbsp; {feature}
                        </div>
                        <div class="factor-value">
                            Current value: <b>{value}</b>
                        </div>
                        <div class="factor-effect-negative">
                            ⬇️ Decreased the prediction by
                            approximately
                            <b>{abs(shap_value):.2f}</b> points.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


        # ====================================================
        # PERSONALIZED RECOMMENDATIONS
        # ====================================================

        st.divider()

        st.markdown(
            '<div class="section-title">'
            '💡 Personalized Recommendations'
            '</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="section-description">'
            'These recommendations are generated from the '
            'student\'s individual SHAP explanation and current '
            'input values.'
            '</div>',
            unsafe_allow_html=True
        )


        recommendations = []


        for _, row in top_features.iterrows():

            feature = row[
                "Feature"
            ]

            value = row[
                "Value"
            ]

            shap_value = float(
                row[
                    "SHAP_Value"
                ]
            )

            feature_recommendations = (
                generate_recommendation(
                    feature,
                    value,
                    shap_value
                )
            )

            recommendations.extend(
                feature_recommendations
            )


        unique_recommendations = []


        for recommendation in recommendations:

            if recommendation not in unique_recommendations:

                unique_recommendations.append(
                    recommendation
                )


        if unique_recommendations:

            for i, recommendation in enumerate(
                unique_recommendations[:5],
                start=1
            ):

                st.markdown(
                    f"""
                    <div class="recommendation-card">
                        <div class="recommendation-number">
                            💡 Recommendation {i}
                        </div>
                        {recommendation}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

        else:

            st.info(
                "Maintain the current academic habits "
                "and monitor future performance."
            )


                # ====================================================
        # PERSONALIZED INTERVENTION RECOMMENDATIONS
        # ====================================================

        st.divider()

        st.subheader(
            "🎯 Personalized Intervention Recommendations"
        )

        st.caption(
            "These recommendations focus on factors that may "
            "negatively influence the predicted performance "
            "and provide targeted intervention strategies."
        )

        (
            recommendation_category,
            recommendations_df
        ) = generate_recommendations(

            student=student_input.iloc[0],

            shap_df=shap_df,

            predicted_score=predicted_score,

            max_recommendations=3
        )

        # ----------------------------------------------------
        # INTERVENTION RECOMMENDATIONS
        # ----------------------------------------------------

        if (
            recommendations_df is not None
            and not recommendations_df.empty
        ):

            for index, row in enumerate(
                recommendations_df.itertuples(
                    index=False
                ),
                start=1
            ):

                with st.container(border=True):

                    st.markdown(
                        f"### 🎯 Intervention {index}"
                    )

                    st.write(
                        f"**Factor:** {row.Feature}"
                    )

                    st.write(
                        f"**Current Value:** "
                        f"{row.Feature_Value}"
                    )

                    st.write(
                        f"**Recommendation:** "
                        f"{row.Recommendation}"
                    )

        else:

            st.success(
                "No immediate intervention is required. "
                "Continue maintaining the current academic habits."
            )


        # ====================================================
        # STUDENT ACTION SUMMARY
        # ====================================================

        st.divider()

        st.subheader(
            "📋 Student Action Summary"
        )

        if predicted_score >= 75:

            summary_text = (
                "The student is predicted to perform well. "
                "Current positive academic behaviors should "
                "be maintained while continuing to monitor "
                "future performance."
            )

            summary_type = "success"

        elif predicted_score >= 60:

            summary_text = (
                "The student shows moderate performance potential. "
                "Targeted improvement in the identified factors "
                "may increase future academic performance."
            )

            summary_type = "warning"

        elif predicted_score >= 50:

            summary_text = (
                "The student may require additional academic support. "
                "The identified negative SHAP factors should be "
                "addressed through targeted improvement."
            )

            summary_type = "warning"

        else:

            summary_text = (
                "The student may be at academic risk. "
                "Early intervention, structured study planning, "
                "and academic support are recommended."
            )

            summary_type = "error"


        if summary_type == "success":

            st.success(
                summary_text
            )

        elif summary_type == "warning":

            st.warning(
                summary_text
            )

        else:

            st.error(
                summary_text
            )


        # ====================================================
        # STUDENT INPUT
        # ====================================================

        with st.expander(
            "📋 View Student Input"
        ):

            display_input = student_input.copy()

            display_input.columns = [
                "Gender",
                "Study Time Hours",
                "Attendance %",
                "Sleep Hours",
                "Parental Education",
                "Internet Access",
                "Extracurricular Activities",
                "Part-time Job",
                "Previous Grade"
            ]

            st.dataframe(
                display_input,
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # TECHNICAL INFORMATION
        # ====================================================

        with st.expander(
            "ℹ️ Technical Information"
        ):

            tech_col1, tech_col2 = st.columns(2)

            with tech_col1:

                st.write(
                    "**Prediction Model:** TabPFN"
                )

                st.write(
                    "**Target:** Final Exam Score"
                )

                st.write(
                    "**Explainability:** SHAP"
                )

            with tech_col2:

                st.write(
                    "**Explanation Type:** "
                    "Local Feature Attribution"
                )

                st.write(
                    f"**Number of Features:** "
                    f"{len(FEATURES)}"
                )

                st.write(
                    f"**SHAP Background Samples:** "
                    f"{background_size}"
                )


        # ====================================================
        # DOWNLOAD STUDENT REPORT
        # ====================================================

        report = []

        report.append(
            "EduPredict-XAI — Student Performance Report"
        )

        report.append(
            "=" * 55
        )

        report.append(
            f"Predicted Final Exam Score: "
            f"{predicted_score:.2f}"
        )

        report.append(
            f"Performance Category: "
            f"{recommendation_category}"
        )

        report.append("")

        report.append(
            "Top SHAP Factors:"
        )

        for _, row in shap_df.head(5).iterrows():

            direction = (
                "increases"
                if row["SHAP_Value"] > 0
                else "decreases"
            )

            report.append(
                f"- {row['Feature']}: "
                f"{row['SHAP_Value']:.4f} "
                f"({direction} prediction)"
            )


        report.append("")

        report.append(
            "Personalized Recommendations:"
        )


        if (
            recommendations_df is not None
            and not recommendations_df.empty
        ):

            for index, row in enumerate(
                recommendations_df.itertuples(
                    index=False
                ),
                start=1
            ):

                report.append(
                    f"{index}. "
                    f"{row.Recommendation}"
                )

        else:

            report.append(
                "No immediate intervention is required."
            )


        report_text = "\n".join(
            report
        )


        # ----------------------------------------------------
        # DOWNLOAD BUTTON
        # Left aligned, same primary style as prediction button
        # ----------------------------------------------------

        st.download_button(
            label="📥 Download Student Report",
            data=report_text,
            file_name="student_performance_report.txt",
            mime="text/plain",
            type="primary"
        )


    # ========================================================
    # ERROR HANDLING
    # ========================================================

    except Exception as error:

        st.error(
            "Prediction or SHAP explanation failed."
        )

        st.exception(error)

