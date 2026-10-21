import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.ui import (INDIGO, TEAL, callout, get_service, hero, kpi_row, missing_results_notice, read_result, section,
                    style_fig)
from src.config import ALLOWED, CATEGORICAL, FEATURE_LABELS, FEATURES, FIG_DIR

svc = get_service()
hero("Research Insights",
     "Everything that proves the system can be trusted: how accurate the models are, how reliable the explanations "
     "are, how well uncertainty is calibrated, and whether predictions are fair.",
     "Performance · Explainability · Uncertainty · Fairness")

cv, hold, preds = read_result("cv_model_summary.csv"), read_result("holdout_comparison.csv"), read_result("holdout_predictions_all_models.csv")
cov, risk, ew = read_result("conformal_coverage.csv"), read_result("risk_category_metrics.csv"), read_result("early_warning_metrics.csv")
imp, stab, faith = read_result("global_importance_all_methods.csv"), read_result("shap_bootstrap_stability.csv"), read_result("shap_faithfulness_deletion.csv")
agree, loc = read_result("explanation_agreement.csv"), read_result("shap_local_top3_stability.csv")
sub, gap, flip = read_result("fairness_subgroup_errors.csv"), read_result("fairness_gap_tests.csv"), read_result("fairness_gender_flip.csv")
if cv is None or hold is None:
    missing_results_notice()
    st.stop()

# ---- headline numbers --------------------------------------------------------------------------
tab_cov = cov[(cov.Model.str.startswith("TabPFN")) & (cov.Nominal == "90%")].Empirical_coverage.iloc[0] * 100 if cov is not None else np.nan
flip_v = flip.set_index(flip.columns[0])["value"]["mean_abs_change"] if flip is not None else np.nan
kpi_row([
    ("Best CV error (RMSE)", f"{cv.RMSE_mean.min():.2f}", f"{cv.sort_values('RMSE_mean').Model.iloc[0]}, 15 folds", INDIGO),
    ("90% interval coverage", f"{tab_cov:.1f}%", "measured on unseen students", TEAL),
    ("Explanation agreement", f"{agree.Spearman.mean():.2f}" if agree is not None else "–", "mean rank correlation, 5 methods"),
    ("Gender-flip change", f"{flip_v:.2f} pts", "same student, gender swapped"),
])
st.write("")

t1, t2, t3, t4 = st.tabs(["📈 Model performance", "🔍 Explainability", "🎯 Uncertainty & risk", "⚖️ Fairness & ethics"])

# ==================================================================================== PERFORMANCE
with t1:
    section("Which model predicts best?", "Repeated 5×3 cross-validation (15 folds): mean ± SD of RMSE, lower is better.")
    d = cv.sort_values("RMSE_mean", ascending=False)
    fig = go.Figure(go.Bar(x=d.RMSE_mean, y=d.Model, orientation="h", error_x=dict(type="data", array=d.RMSE_std),
                           marker_color=[INDIGO if r == 1 else "#B8BEE0" for r in d.Rank]))
    fig.update_layout(xaxis_title="RMSE (points)")
    c1, c2 = st.columns([1.2, 1], gap="large")
    c1.plotly_chart(style_fig(fig, 340), width="stretch")
    show = cv[["Rank", "Model", "MAE_mean", "RMSE_mean", "R2_mean", "p_vs_best"]].copy()
    show["Worse than best?"] = np.where(show.p_vs_best.isna(), "–", np.where(show.p_vs_best < 0.05, "Yes (p<0.05)", "No"))
    c2.dataframe(show.drop(columns="p_vs_best").round(3), hide_index=True, width="stretch", height=300)
    callout("A regularised linear model is the most accurate under cross-validation: study habits relate to the score almost linearly, "
            "so complex models add little. The project's primary model, TabPFN, is the strongest <i>non-linear</i> model on the "
            "original test split and significantly beats XGBoost, LightGBM and Extra Trees.")

    section("Original 80/20 hold-out test (n = 200)", "RMSE with 95% bootstrap confidence interval and paired test against TabPFN.")
    h = hold.sort_values("RMSE", ascending=False)
    fig = go.Figure(go.Bar(
        x=h.RMSE, y=h.Model, orientation="h", marker_color=[TEAL if "TabPFN" in m else "#B8BEE0" for m in h.Model],
        error_x=dict(type="data", symmetric=False, array=h.RMSE_CI95_high - h.RMSE, arrayminus=h.RMSE - h.RMSE_CI95_low)))
    fig.update_layout(xaxis_title="RMSE (points)")
    c1, c2 = st.columns([1.2, 1], gap="large")
    c1.plotly_chart(style_fig(fig, 340), width="stretch")
    c2.dataframe(hold[["Rank", "Model", "MAE", "RMSE", "R2", "Wilcoxon_p_vs_TabPFN"]].round(3), hide_index=True, width="stretch", height=300)

    section("How close are the predictions?")
    models = [c for c in preds.columns if c not in ("Actual", "Mean baseline")]
    model = st.selectbox("Model", models, index=models.index("TabPFN (stored predictions)") if "TabPFN (stored predictions)" in models else 0)
    y, p = preds.Actual, preds[model]
    c1, c2 = st.columns(2, gap="large")
    fig = px.scatter(x=y, y=p, labels={"x": "Actual score", "y": "Predicted score"}, color_discrete_sequence=[INDIGO], opacity=.7,
                     title="Predicted vs actual")
    fig.add_shape(type="line", x0=40, y0=40, x1=100, y1=100, line=dict(color="#EF4444", dash="dash"))
    c1.plotly_chart(style_fig(fig, 340), width="stretch")
    fig = px.histogram(p - y, nbins=30, color_discrete_sequence=[TEAL], title="Residuals (predicted − actual)")
    fig.update_layout(xaxis_title="Error (points)", showlegend=False)
    c2.plotly_chart(style_fig(fig, 340), width="stretch")
    st.caption(f"Mean error {np.mean(p - y):+.2f} points. Errors are largest for very high scores because the target is capped at 100.")

# ================================================================================== EXPLAINABILITY
with t2:
    section("What drives the predictions?", "Five attribution methods compared (share of total importance).")
    if imp is not None:
        imp = imp.rename(columns={imp.columns[0]: "Feature"})
        share = imp.set_index("Feature").clip(lower=0)
        share = (share / share.sum()).reset_index().melt("Feature", var_name="Method", value_name="Share")
        share["Feature"] = share.Feature.map(FEATURE_LABELS)
        fig = px.bar(share, y="Feature", x="Share", color="Method", barmode="group", orientation="h")
        fig.update_layout(yaxis=dict(autorange="reversed"))
        st.plotly_chart(style_fig(fig, 500), width="stretch")
        if agree is not None:
            callout(f"The methods agree: mean rank correlation <b>{agree.Spearman.mean():.2f}</b> (lowest {agree.Spearman.min():.2f}). "
                    "Study time and previous grade dominate; gender has almost no influence.", "good")

    c1, c2 = st.columns(2, gap="large")
    with c1:
        section("Are explanations stable?", "30 bootstrap re-trainings.")
        if stab is not None:
            s = stab.rename(columns={stab.columns[0]: "Feature"}).copy()
            s["Feature"] = s.Feature.map(FEATURE_LABELS)
            fig = go.Figure(go.Bar(x=s.Feature, y=s.Mean_abs_SHAP, error_y=dict(type="data", array=s.Std), marker_color=INDIGO))
            fig.update_layout(yaxis_title="Mean |SHAP| ± SD", xaxis_tickangle=-35)
            st.plotly_chart(style_fig(fig, 340), width="stretch")
            if loc is not None:
                j = float(loc.set_index(loc.columns[0]).loc["mean_local_top3_jaccard", "value"])
                st.caption(f"Per-student top-3 drivers overlap {j:.2f} (Jaccard) between re-trained models.")
    with c2:
        section("Do explanations reflect the model?", "Deletion test.")
        if faith is not None:
            fig = go.Figure()
            fig.add_scatter(x=faith.k_features_masked, y=faith.Mean_abs_change_top_SHAP, name="Mask top-k SHAP features", line=dict(color=INDIGO, width=3))
            fig.add_scatter(x=faith.k_features_masked, y=faith.Mean_abs_change_random, name="Mask k random features", line=dict(color="#94A3B8", dash="dash"))
            fig.update_layout(xaxis_title="k features masked", yaxis_title="Mean |change in prediction|")
            st.plotly_chart(style_fig(fig, 340), width="stretch")
            st.caption(f"Removing the most important SHAP feature changes predictions {faith.Ratio.iloc[0]:.1f}× more than a random one.")

    section("Feature effect explorer", "Thick line = average effect (PDP); thin lines = individual students (ICE).")
    feat = st.selectbox("Feature", FEATURES, format_func=FEATURE_LABELS.get, index=1)
    n = 120 if svc.is_tree else 30
    sample = svc.df[FEATURES].sample(n, random_state=1).reset_index(drop=True)
    grid = ALLOWED[feat] if feat in CATEGORICAL else list(np.round(np.linspace(svc.df[feat].quantile(.05), svc.df[feat].quantile(.95), 15), 2))
    M = np.array([svc.predict(sample.assign(**{feat: g})) for g in grid])
    fig = go.Figure()
    if feat in CATEGORICAL:
        fig.add_bar(x=[str(g) for g in grid], y=M.mean(axis=1), marker_color=TEAL)
        fig.update_layout(yaxis_range=[M.mean() - 8, M.mean() + 8])
    else:
        for j in range(min(30, M.shape[1])):
            fig.add_scatter(x=grid, y=M[:, j], mode="lines", line=dict(color="rgba(79,70,229,.12)", width=1), showlegend=False)
        fig.add_scatter(x=grid, y=M.mean(axis=1), mode="lines", line=dict(color=INDIGO, width=4), name="Average")
    fig.update_layout(xaxis_title=FEATURE_LABELS[feat], yaxis_title="Predicted score")
    st.plotly_chart(style_fig(fig, 380), width="stretch")
    img = FIG_DIR / "shap_beeswarm_catboost.png"
    if img.exists():
        with st.expander("SHAP beeswarm (every student, every feature)"):
            st.image(str(img), width=700)
            st.caption("Each dot is a student. Red = high feature value, blue = low. Horizontal position = impact on the predicted score.")

# ================================================================================ UNCERTAINTY & RISK
with t3:
    section("How trustworthy are the score ranges?", "Conformal intervals: nominal vs measured coverage.")
    if cov is not None:
        c = cov[cov.Model.str.startswith("TabPFN")].copy()
        c1, c2 = st.columns([1, 1.2], gap="large")
        c1.dataframe(c, hide_index=True, width="stretch")
        with c2:
            img = FIG_DIR / "conformal_intervals_tabpfn.png"
            if img.exists():
                st.image(str(img))
        callout("A 90% interval really covers about 90% of unseen students, so the ranges shown in <b>Predict &amp; Explain</b> can be trusted.", "good")
    section("Risk categories and early warning", "Fixed thresholds vs cohort-relative bands, and the bottom-20% detector.")
    c1, c2 = st.columns(2, gap="large")
    if risk is not None:
        c1.markdown("**Category accuracy**")
        c1.dataframe(risk.round(3), hide_index=True, width="stretch")
    if ew is not None:
        c2.markdown("**Early-warning detector**")
        c2.dataframe(ew[ew.Model.str.startswith("TabPFN")].round(3), hide_index=True, width="stretch")
    callout("Fixed thresholds (75/60/50/35) label most students 'Excellent', so their high accuracy only reflects the majority class. "
            "Cohort-relative bands are balanced and more useful for spotting students who need support.", "warn")

# =============================================================================== FAIRNESS & ETHICS
with t4:
    if sub is not None:
        section("Does the model err more for some groups?")
        c1, c2 = st.columns(2)
        pred = c1.selectbox("Predictor", sub.Predictor.unique())
        attr = c2.selectbox("Attribute", sub.Attribute.unique())
        d = sub[(sub.Predictor == pred) & (sub.Attribute == attr)].copy()
        d["label"] = d.Group + " (n=" + d.n.astype(str) + ")"
        fig = px.bar(d, x="label", y="MAE", text=d.MAE.round(2), color_discrete_sequence=[INDIGO], title="Mean absolute error by group")
        fig.update_layout(xaxis_title="", yaxis_title="MAE (points)")
        st.plotly_chart(style_fig(fig, 320), width="stretch")
        g = gap.copy()
        g["Significant (p<0.05)"] = g.p_value < 0.05
        st.dataframe(g.round(4), hide_index=True, width="stretch")
        callout("No significant error gap by gender. A small but detectable gap by parental education appears in the larger out-of-fold "
                "analysis (students with High-School-educated parents have higher error), so treat their predictions with extra care.", "warn")
    callout("<b>Responsible use:</b> the tool supports teachers; it must not be used for grading, ranking or excluding students. Gender and "
            "parental education are never used to generate recommendations. Explanations and what-if gains are associations, not causes.")
