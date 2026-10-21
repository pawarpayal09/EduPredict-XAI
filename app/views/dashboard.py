import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import (BAND_COLORS, INDIGO, TEAL, callout, get_service, hero, kpi_row, read_result, section,
                    style_fig)
from src.config import CATEGORY_ORDER, TARGET, get_category

svc = get_service()
df = svc.df
hero("Student Performance Intelligence Dashboard",
     "Predict final exam scores, understand <b>why</b>, quantify uncertainty and turn insight into action.",
     "Prediction → Explanation → Intervention")

hold, cv, cov = read_result("holdout_comparison.csv"), read_result("cv_model_summary.csv"), read_result("conformal_coverage.csv")
best_rmse = f"{hold.RMSE.min():.2f}" if hold is not None else "–"
cov90 = "–"
if cov is not None:
    r = cov[(cov.Model.str.startswith("TabPFN")) & (cov.Nominal == "90%")]
    cov90 = f"{r.Empirical_coverage.iloc[0] * 100:.1f}%" if len(r) else "–"
kpi_row([
    ("Students analysed", f"{len(df):,}", "9 predictors per student"),
    ("Cohort mean score", f"{df[TARGET].mean():.1f}", f"SD {df[TARGET].std():.1f}"),
    ("Best hold-out RMSE", best_rmse, "points on a 0-100 scale", INDIGO),
    ("90% interval coverage", cov90, "verified empirically", TEAL),
    ("Early-warning share", f"{(df[TARGET] < svc.early_warning_cut).mean() * 100:.0f}%", f"scores below {svc.early_warning_cut:.0f}"),
])

section("Cohort overview", "Where students stand and which habits move the score.")
c1, c2 = st.columns(2, gap="large")
with c1:
    fig = px.histogram(df, x=TARGET, nbins=30, color_discrete_sequence=[INDIGO])
    fig.update_layout(title="Final exam score distribution", xaxis_title="Final exam score", yaxis_title="Students")
    ceil = int((df[TARGET] >= 100).sum())
    fig.add_annotation(x=100, y=0, text=f"{ceil} at ceiling (100)", showarrow=True, ay=-60, font=dict(size=11))
    st.plotly_chart(style_fig(fig), width="stretch")
with c2:
    fig = px.scatter(df, x="study_time_hours", y=TARGET, color="part_time_job", opacity=.6,
                     color_discrete_map={"Yes": "#F59E0B", "No": INDIGO},
                     labels={"study_time_hours": "Daily study time (h)", TARGET: "Final exam score"})
    k, b0 = np.polyfit(df.study_time_hours, df[TARGET], 1)
    xs = np.linspace(df.study_time_hours.min(), df.study_time_hours.max(), 20)
    fig.add_scatter(x=xs, y=k * xs + b0, mode="lines", name=f"Trend (+{k:.1f} pts per hour)", line=dict(color="#1E2140", width=3))
    fig.update_layout(title="Study time vs score (coloured by part-time job)")
    st.plotly_chart(style_fig(fig), width="stretch")

c3, c4 = st.columns(2, gap="large")
with c3:
    absolute = df[TARGET].apply(get_category).value_counts().reindex(CATEGORY_ORDER).fillna(0)
    relative = df[TARGET].apply(svc.cohort_band).value_counts().reindex(CATEGORY_ORDER).fillna(0)
    fig = go.Figure()
    fig.add_bar(name="Fixed thresholds (75/60/50/35)", x=CATEGORY_ORDER, y=absolute.values, marker_color="#CBD1EA")
    fig.add_bar(name="Cohort-relative bands", x=CATEGORY_ORDER, y=relative.values, marker_color=INDIGO)
    fig.update_layout(title="Why cohort-relative risk bands?", barmode="group")
    st.plotly_chart(style_fig(fig), width="stretch")
    callout(f"Fixed thresholds label <b>{int(absolute['Excellent'])}</b> of {len(df)} students 'Excellent' and "
            f"<b>{int(absolute['At Risk'])}</b> 'At Risk', which hides real differences. Percentile bands separate students usefully.", "warn")
with c4:
    if hold is not None:
        h = hold.sort_values("RMSE", ascending=False)
        fig = go.Figure(go.Bar(
            x=h.RMSE, y=h.Model, orientation="h", marker_color=[TEAL if "TabPFN" in m else "#B8BEE0" for m in h.Model],
            error_x=dict(type="data", symmetric=False, array=h.RMSE_CI95_high - h.RMSE, arrayminus=h.RMSE - h.RMSE_CI95_low)))
        fig.update_layout(title="Hold-out RMSE with 95% bootstrap CI (lower is better)", xaxis_title="RMSE (points)")
        st.plotly_chart(style_fig(fig), width="stretch")
        callout("TabPFN is the strongest non-linear model, but a simple linear model is statistically comparable: "
                "the signal in this dataset is mostly linear. Details in <b>Research Insights</b>.")
    else:
        st.info("Run `python run_all_upgrades.py` to populate the model leaderboard.")

section("How the system works")
steps = [("1", "Student profile", "9 validated inputs"), ("2", "Prediction", "TabPFN / CatBoost score"),
         ("3", "Uncertainty", "Conformal 90% interval"), ("4", "Explanation", "SHAP waterfall + text"),
         ("5", "Action plan", "Ranked what-if gains")]
for col, (n, t, s) in zip(st.columns(5, gap="medium"), steps):
    col.markdown(f'<div class="card step"><div class="n">{n}</div><div style="font-weight:750">{t}</div>'
                 f'<div style="color:#6B7194;font-size:.85rem">{s}</div></div>', unsafe_allow_html=True)
