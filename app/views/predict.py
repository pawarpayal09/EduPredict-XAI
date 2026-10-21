import html
import json
from datetime import date

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import (BAND_COLORS, INDIGO, PRIORITY_COLORS, RED, TEAL, badge, callout, get_service, hero, kpi,
                    section, style_fig)
from src.config import ALLOWED, FEATURE_LABELS, FEATURES, TARGET
from src.recommendations import build_action_plan

svc = get_service()
hero("Predict & Explain", "Enter a student profile to get the expected exam score, a 90% range, the reasons "
     "behind it, and a ranked plan for improvement.", "Single student")

PRESETS = {
    "Custom": None,
    "Typical student": dict(gender="Female", study_time_hours=3.5, attendance_percent=85.0, sleep_hours=7.0,
                            parental_education="Bachelors", internet_access="Yes", extracurricular_activities="Yes",
                            part_time_job="No", previous_grade=70.0),
    "Needs support": dict(gender="Male", study_time_hours=1.5, attendance_percent=68.0, sleep_hours=5.5,
                          parental_education="High School", internet_access="No", extracurricular_activities="Yes",
                          part_time_job="Yes", previous_grade=55.0),
    "High achiever": dict(gender="Female", study_time_hours=5.5, attendance_percent=96.0, sleep_hours=7.5,
                          parental_education="Masters", internet_access="Yes", extracurricular_activities="No",
                          part_time_job="No", previous_grade=88.0),
}
DEFAULTS = PRESETS["Typical student"]
for k, v in DEFAULTS.items():
    st.session_state.setdefault(f"in_{k}", v)


def apply_preset():
    p = PRESETS.get(st.session_state["preset"])
    if p:
        for k, v in p.items():
            st.session_state[f"in_{k}"] = v


left, right = st.columns([1, 2.1], gap="large")

with left:
    st.markdown("#### 📝 Student profile")
    st.selectbox("Load example", list(PRESETS), key="preset", on_change=apply_preset)
    st.slider("📚 Daily study time (h)", 0.0, 12.0, step=0.5, key="in_study_time_hours")
    st.slider("📅 Attendance (%)", 0.0, 100.0, step=1.0, key="in_attendance_percent")
    st.slider("😴 Sleep (h)", 3.0, 12.0, step=0.5, key="in_sleep_hours")
    st.slider("📈 Previous grade", 0.0, 100.0, step=1.0, key="in_previous_grade")
    a, b = st.columns(2)
    a.selectbox("Gender", ALLOWED["gender"], key="in_gender")
    b.selectbox("Parental education", ALLOWED["parental_education"], key="in_parental_education")
    a.selectbox("Internet access", ALLOWED["internet_access"], key="in_internet_access")
    b.selectbox("Part-time job", ALLOWED["part_time_job"], key="in_part_time_job")
    st.selectbox("Extracurricular activities", ALLOWED["extracurricular_activities"], key="in_extracurricular_activities")
    go_btn = st.button("🔮 Analyze student", type="primary", width="stretch")

current = {k: st.session_state[f"in_{k}"] for k in FEATURES}

if go_btn:
    with st.spinner("Predicting and computing SHAP explanation..."):
        row = svc.validate(current)
        pred = svc.predict_one(current)
        base, shap_df = svc.explain(row)
        plan = build_action_plan(row.iloc[0], shap_df, pred["predicted_score"], svc)
        wi_base, wi_plans, wi_texts = svc.whatif(row.iloc[0], 5.0)
    st.session_state["analysis"] = dict(student=current, pred=pred, base=base, shap=shap_df, plan=plan,
                                        text=svc.explanation_text(pred["predicted_score"], base, shap_df),
                                        wi=(wi_plans, wi_texts))
    st.session_state.update(wi_study=float(current["study_time_hours"]), wi_att=float(current["attendance_percent"]),
                            wi_sleep=float(current["sleep_hours"]), wi_job=current["part_time_job"],
                            wi_net=current["internet_access"])

A = st.session_state.get("analysis")
with right:
    if not A:
        st.markdown("#### 👈 Fill in the profile and press **Analyze student**")
        callout("You will get: the predicted score and 90% range · where the student sits in the cohort · a SHAP waterfall "
                "explaining the score · a ranked action plan with estimated point gains · a live what-if simulator · a downloadable report.")
        st.stop()

    P = A["pred"]
    if A["student"] != current:
        callout("Inputs changed since the last analysis. Press <b>Analyze student</b> to refresh the results.", "warn")
    score, lo, hi = P["predicted_score"], *P["interval_90"]
    band = P["cohort_band"]
    c = st.columns(4, gap="small")
    c[0].markdown(kpi("Predicted score", f"{score:.1f}", f"vs cohort mean {P['cohort_mean']:.1f}", INDIGO), unsafe_allow_html=True)
    c[1].markdown(kpi("90% range", f"{lo:.0f}–{hi:.0f}", "likely exam outcome"), unsafe_allow_html=True)
    c[2].markdown(kpi("Cohort position", band, f"better than {P['cohort_percentile']:.0f}% of students", BAND_COLORS[band]), unsafe_allow_html=True)
    c[3].markdown(kpi("Status", "⚠ Early warning" if P["early_warning"] else "✔ On track",
                      f"fixed-scale category: {P['category']}", RED if P["early_warning"] else TEAL), unsafe_allow_html=True)

    hist = px.histogram(svc.df, x=TARGET, nbins=35, color_discrete_sequence=["#D5D9F0"])
    hist.add_vrect(x0=lo, x1=hi, fillcolor=INDIGO, opacity=0.12, line_width=0)
    hist.add_vline(x=score, line_color=INDIGO, line_width=3, annotation_text=f"Predicted {score:.1f}")
    hist.update_layout(title="Where this prediction falls in the cohort (shaded = 90% range)", xaxis_title="Final exam score",
                       yaxis_title="Students", showlegend=False)
    st.plotly_chart(style_fig(hist, 260), width="stretch")

    t1, t2, t3, t4 = st.tabs(["🔍 Why this score?", "🎯 Action plan", "🔄 What-if simulator", "📄 Report"])

    with t1:
        callout(A["text"])
        sh = A["shap"].sort_values("Absolute_SHAP", ascending=True)
        labels = [f"{FEATURE_LABELS[f]} = {v}" for f, v in zip(sh.Feature, sh.Value)]
        wf = go.Figure(go.Waterfall(
            orientation="h", measure=["absolute"] + ["relative"] * len(sh) + ["total"],
            y=["Cohort baseline"] + labels + ["Prediction"], x=[A["base"]] + list(sh.SHAP_Value) + [0],
            connector=dict(line=dict(color="#CBD1EA")),
            increasing=dict(marker=dict(color="#22C55E")), decreasing=dict(marker=dict(color="#EF4444")),
            totals=dict(marker=dict(color=INDIGO)), texttemplate="%{delta:+.1f}", textposition="outside"))
        wf.update_layout(title="SHAP waterfall: how each factor moves the score", xaxis_title="Predicted score",
                         xaxis_range=[max(0, min(A["base"], score) - 14), min(105, max(A["base"], score) + 14)])
        st.plotly_chart(style_fig(wf, 430), width="stretch")
        table = A["shap"][["Feature", "Value", "SHAP_Value"]].copy()
        table["Feature"] = table.Feature.map(FEATURE_LABELS)
        table["Value"] = table.Value.astype(str)
        table["Effect"] = np.where(table.SHAP_Value > 0, "⬆ raises score", "⬇ lowers score")
        st.dataframe(table.round(2), hide_index=True, width="stretch")

    with t2:
        plan = A["plan"]
        if plan["strengths"]:
            callout("<b>Keep doing:</b> " + " · ".join(plan["strengths"]), "good")
        if plan["actions"]:
            section("Ranked actions", "Estimated gain = change in predicted score if only this factor changes (model-based).")
            for a in plan["actions"]:
                col = PRIORITY_COLORS[a["Priority"]]
                st.markdown(f'<div class="action" style="--c:{col}"><div class="t">{a["Factor"]} {badge(a["Priority"] + " priority", col)}'
                            f' &nbsp; <b style="color:{TEAL}">+{a["Est_gain"]} pts</b></div><div>{a["Action"]}</div></div>',
                            unsafe_allow_html=True)
        else:
            callout("No single change is estimated to add more than 0.5 points. The current habits are already well-aligned.", "good")
        for s in plan["support"]:
            callout("🤝 " + s, "warn")
        plans, texts = A["wi"]
        section("Fastest combined plans to gain +5 points")
        if len(plans):
            out = plans[["Predicted_score", "Gain", "Effort"]].round(1).copy()
            out.insert(0, "Plan", texts)
            st.dataframe(out, hide_index=True, width="stretch")
        else:
            st.info("No combination of realistic changes reaches +5 points; consider mentoring or tutoring support.")
        st.caption("Plans use only actionable factors and realistic bounds. They are associations learned from data, not guaranteed causal effects.")

    with t3:
        section("Try a scenario", "Move the sliders and see the predicted score update instantly.")
        s1, s2 = st.columns(2)
        study = s1.slider("Daily study time (h)", 0.0, 12.0, step=0.5, key="wi_study")
        att = s1.slider("Attendance (%)", 0.0, 100.0, step=1.0, key="wi_att")
        sleep = s2.slider("Sleep (h)", 3.0, 12.0, step=0.5, key="wi_sleep")
        job = s2.radio("Part-time job", ["Yes", "No"], horizontal=True, key="wi_job")
        net = s2.radio("Internet access", ["Yes", "No"], horizontal=True, key="wi_net")
        new = svc.simulate(A["student"], study_time_hours=study, attendance_percent=att, sleep_hours=sleep,
                           part_time_job=job, internet_access=net)
        d = new - score
        m1, m2, m3 = st.columns(3)
        m1.metric("Original prediction", f"{score:.1f}")
        m2.metric("Scenario prediction", f"{new:.1f}", f"{d:+.1f} pts")
        m3.metric("Scenario band", svc.cohort_band(new))
        fig = go.Figure()
        fig.add_bar(x=["Original", "Scenario"], y=[score, new], marker_color=["#B8BEE0", TEAL if d >= 0 else RED],
                    text=[f"{score:.1f}", f"{new:.1f}"], textposition="outside")
        fig.update_layout(yaxis_range=[max(0, min(score, new) - 15), min(105, max(score, new) + 8)], title="Scenario comparison")
        st.plotly_chart(style_fig(fig, 300), width="stretch")

    with t4:
        rows = "".join(f"<tr><td>{FEATURE_LABELS[f]}</td><td>{html.escape(str(v))}</td></tr>" for f, v in A["student"].items())
        acts = "".join(f"<li><b>{html.escape(a['Factor'])}</b> (+{a['Est_gain']} pts, {a['Priority']}): {html.escape(a['Action'])}</li>"
                       for a in A["plan"]["actions"]) or "<li>No change estimated to add more than 0.5 points.</li>"
        report = f"""<html><head><meta charset="utf-8"><title>EduPredict-XAI report</title><style>
body{{font-family:Arial;max-width:760px;margin:30px auto;color:#1E2140}}h1{{color:#4F46E5}}td{{padding:4px 12px;border-bottom:1px solid #eee}}
.k{{background:#EEF0FF;border-radius:10px;padding:12px 16px;margin:12px 0}}</style></head><body>
<h1>EduPredict-XAI - Student Performance Report</h1><p>Generated {date.today()} - model: {P['model']}</p>
<div class="k"><b>Predicted final exam score: {score:.1f}</b> (90% range {lo:.0f}-{hi:.0f})<br>
Cohort position: {band} (better than {P['cohort_percentile']:.0f}% of students) - Fixed-scale category: {P['category']}</div>
<h3>Student profile</h3><table>{rows}</table><h3>Why this score</h3><p>{html.escape(A['text'])}</p>
<h3>Recommended actions</h3><ul>{acts}</ul>
<p style="color:#6B7194;font-size:12px">Model-based estimates for decision support only; not a guarantee of results.</p></body></html>"""
        d1, d2 = st.columns(2)
        d1.download_button("⬇ Download report (HTML)", report, "edupredict_report.html", "text/html", type="primary", width="stretch")
        d2.download_button("⬇ Download data (JSON)", json.dumps({"student": A["student"], "prediction": P,
                           "shap": {r.Feature: round(float(r.SHAP_Value), 3) for r in A["shap"].itertuples()},
                           "actions": A["plan"]["actions"]}, indent=2, default=str), "edupredict_result.json", "application/json", width="stretch")
        st.caption("Open the HTML report in a browser and use Print → Save as PDF if you need a PDF.")
