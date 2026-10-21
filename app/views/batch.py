import io

import pandas as pd
import plotly.express as px
import streamlit as st

from app.ui import BAND_COLORS, INDIGO, RED, TEAL, callout, get_service, hero, kpi_row, section, style_fig
from src.config import CATEGORY_ORDER, FEATURES
from src.service import ValidationError

svc = get_service()
hero("Class / Batch Analysis", "Upload a CSV of students to score a whole class, flag early-warning cases "
     "and download the results.", "Teacher view")

TEMPLATE = pd.DataFrame([
    ["Female", 3.5, 85, 7.0, "Bachelors", "Yes", "Yes", "No", 70],
    ["Male", 1.5, 68, 5.5, "High School", "No", "Yes", "Yes", 55],
    ["Female", 5.5, 96, 7.5, "Masters", "Yes", "No", "No", 88],
    ["Male", 2.5, 74, 6.0, "High School", "Yes", "No", "Yes", 62],
    ["Female", 4.0, 90, 8.0, "PhD", "Yes", "Yes", "No", 79],
    ["Male", 0.8, 60, 5.0, "High School", "No", "No", "Yes", 48],
], columns=FEATURES)

c1, c2 = st.columns([2, 1], gap="large")
with c1:
    up = st.file_uploader("Upload CSV", type=["csv"], help="Required columns: " + ", ".join(FEATURES))
with c2:
    st.markdown("**Need a starting point?**")
    st.download_button("⬇ Download CSV template", TEMPLATE.to_csv(index=False), "edupredict_template.csv", "text/csv", width="stretch")
    use_sample = st.button("Try with sample class", width="stretch")

if use_sample:
    st.session_state["batch_df"] = TEMPLATE.copy()
if up is not None:
    try:
        st.session_state["batch_df"] = pd.read_csv(up)
    except Exception as e:  # noqa: BLE001
        st.error(f"Could not read the file: {e}")

raw = st.session_state.get("batch_df")
if raw is None:
    callout("Columns needed: <code>" + "</code>, <code>".join(FEATURES) + "</code>. Extra columns (e.g. student name or ID) are ignored "
            "in the model but you can keep them in your own copy.")
    st.stop()

try:
    res = svc.batch(raw)
except ValidationError as e:
    st.error("Your file has problems:")
    for p in e.problems:
        st.write("• " + p)
    st.stop()

if "student_id" in raw.columns:
    res.insert(0, "student_id", raw["student_id"].values)

n = len(res)
kpi_row([
    ("Students scored", f"{n}", "rows validated"),
    ("Average predicted", f"{res.Predicted_Score.mean():.1f}", f"cohort ref. {svc.cohort_mean:.1f}", INDIGO),
    ("Early-warning cases", f"{int(res.Early_Warning.sum())}", f"{res.Early_Warning.mean() * 100:.0f}% of class", RED),
    ("Lowest predicted", f"{res.Predicted_Score.min():.1f}", "needs attention first"),
])

section("Class overview")
a, b = st.columns(2, gap="large")
with a:
    dist = res.Cohort_Band.value_counts().reindex(CATEGORY_ORDER).fillna(0).reset_index()
    dist.columns = ["Band", "Students"]
    fig = px.bar(dist, x="Band", y="Students", color="Band", color_discrete_map=BAND_COLORS, title="Students per cohort band")
    fig.update_layout(showlegend=False)
    st.plotly_chart(style_fig(fig, 320), width="stretch")
with b:
    fig = px.histogram(res, x="Predicted_Score", nbins=20, color_discrete_sequence=[INDIGO], title="Predicted score distribution")
    fig.add_vline(x=svc.early_warning_cut, line_dash="dash", line_color=RED, annotation_text="early-warning cut")
    st.plotly_chart(style_fig(fig, 320), width="stretch")

section("Student results")
f1, f2 = st.columns(2)
bands = f1.multiselect("Filter by cohort band", CATEGORY_ORDER, default=CATEGORY_ORDER)
only_ew = f2.toggle("Show early-warning cases only", value=False)
view = res[res.Cohort_Band.isin(bands)]
if only_ew:
    view = view[view.Early_Warning]
view = view.sort_values("Predicted_Score")
st.dataframe(view, hide_index=True, width="stretch", column_config={
    "Predicted_Score": st.column_config.ProgressColumn("Predicted", min_value=0, max_value=100, format="%.1f"),
    "Early_Warning": st.column_config.CheckboxColumn("Early warning")})
st.download_button("⬇ Download full results (CSV)", res.to_csv(index=False), "edupredict_batch_results.csv", "text/csv", type="primary")
if not svc.is_tree:
    st.caption("Per-student top drivers are skipped when TabPFN is active (SHAP for TabPFN is slow). Use Predict & Explain for single students.")
