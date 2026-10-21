import pandas as pd
import streamlit as st

from app.ui import callout, get_service, hero, kpi_row, section
from src.config import ACTIONABLE, FEATURE_LABELS, FEATURES, TARGET

svc = get_service()
df = svc.df
hero("About EduPredict-XAI",
     "An explainable decision-support system that predicts a student's final exam score, shows why, states how "
     "certain it is, and suggests what could help.", "Project overview")

kpi_row([("Students in dataset", f"{len(df):,}", "reference cohort"),
         ("Input factors", f"{len(FEATURES)}", "academic, lifestyle, background"),
         ("Prediction target", "Exam score", "0 – 100 scale"),
         ("Score range in data", f"{df[TARGET].min():.0f} – {df[TARGET].max():.0f}", f"mean {df[TARGET].mean():.1f}")])

section("The problem it solves", "Why prediction alone is not enough.")
c1, c2, c3 = st.columns(3, gap="medium")
for col, (t, d) in zip((c1, c2, c3), [
    ("🔮 Prediction", "Estimates the final exam score early enough for teachers and students to act, with a 90% range instead of a single number."),
    ("🔍 Explanation", "SHAP shows how each factor pushes the score up or down, in a waterfall chart and in plain language, so the result is not a black box."),
    ("🎯 Intervention", "Turns the explanation into a ranked action plan. Each suggestion carries an estimated point gain from re-running the model on a realistic change.")]):
    col.markdown(f'<div class="card"><div style="font-weight:800;font-size:1.05rem;margin-bottom:.3rem">{t}</div>'
                 f'<div style="color:#4A4F72;line-height:1.55">{d}</div></div>', unsafe_allow_html=True)

section("Input factors", "What the model looks at, and which factors can realistically be changed.")
desc = {"gender": "Student gender (used by the model, never used for recommendations)",
        "study_time_hours": "Average hours of study per day", "attendance_percent": "Share of classes attended",
        "sleep_hours": "Average nightly sleep", "parental_education": "Highest education level of parents",
        "internet_access": "Reliable internet access at home", "extracurricular_activities": "Takes part in extracurriculars",
        "part_time_job": "Works a part-time job", "previous_grade": "Result in the previous assessment"}
table = pd.DataFrame({"Factor": [FEATURE_LABELS[f] for f in FEATURES], "Meaning": [desc[f] for f in FEATURES],
                      "Can be changed?": ["Yes" if f in ACTIONABLE else "No (context only)" for f in FEATURES]})
st.dataframe(table, hide_index=True, width="stretch")

section("Methods used")
m = pd.DataFrame([
    ["Prediction", "TabPFN (primary), compared with Ridge, Random Forest, Extra Trees, CatBoost, XGBoost, LightGBM"],
    ["Explanation", "SHAP: exact TreeSHAP for tree models, permutation SHAP for TabPFN"],
    ["Uncertainty", "Split-conformal prediction intervals (80/90/95%) with measured coverage"],
    ["Risk bands", "Cohort-relative percentile bands (10/30/60/85) plus a bottom-20% early-warning flag"],
    ["Action plan", "Realistic single-factor targets, ranked by model-estimated point gain; combined what-if plans"],
    ["Validation", "Repeated 5×3 cross-validation, bootstrap CIs, corrected paired tests, fairness audit"]],
    columns=["Component", "Technique"])
st.dataframe(m, hide_index=True, width="stretch")

section("Quick glossary")
for term, text in [
    ("RMSE / MAE", "Typical size of the prediction error in score points. RMSE penalises large misses more. Lower is better."),
    ("R²", "Share of the differences between students that the model explains (1 = perfect, 0 = no better than the average)."),
    ("SHAP value", "How many points a factor adds to (+) or removes from (−) the prediction compared with the average student."),
    ("90% prediction interval", "A range built so that about 9 in 10 similar students land inside it. Wider means less certain."),
    ("Cohort band / percentile", "Where the predicted score sits relative to the other students, e.g. 'better than 40% of students'."),
    ("Early warning", "Predicted score is among the lowest ~20% of students (with a small safety margin), so a check-in is worthwhile."),
    ("What-if gain", "Change in the predicted score if only the stated factors change. It is a model estimate, not a guarantee.")]:
    with st.expander(term):
        st.write(text)

section("Frequently asked questions")
for q, a in [
    ("Does a low prediction mean the student will fail?", "No. It is a probability-style estimate with a range. Use it to start a conversation and offer support, not to label a student."),
    ("Why is the action plan sometimes short?", "Only factors that can realistically change and that the model estimates to add at least 0.5 points are listed."),
    ("Why does the app sometimes use a different model than TabPFN?", "If the saved TabPFN file is missing the app falls back to CatBoost and shows the active model in the sidebar. Scores are similar on this dataset."),
    ("Can I score a whole class?", "Yes, use the Class / Batch Analysis page with a CSV file. Results can be filtered and downloaded."),
    ("Where is the evidence that the models and explanations are reliable?", "On the Research Insights page: accuracy, explanation stability, interval calibration and fairness checks.")]:
    with st.expander(q):
        st.write(a)

section("Limitations and responsible use")
callout("• The reference dataset has 1,000 students, looks synthetic, and scores are capped at 100.<br>"
        "• SHAP effects and what-if gains are associations learned from data, not proven causes. Validate with real interventions.<br>"
        "• Intervals are calibrated on 200 hold-out students; small subgroups carry wider uncertainty.<br>"
        "• The tool supports teachers. It must not be the only basis for grading, ranking or excluding a student.", "warn")
st.caption(f"Active model: {svc.model_name}")
