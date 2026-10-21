"""EduPredict-XAI v3 - multi-page application.   Run:  streamlit run app/App.py"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(page_title="EduPredict-XAI", page_icon="🎓", layout="wide", initial_sidebar_state="expanded")

from app.ui import badge, get_service, inject_css  # noqa: E402

inject_css()
service = get_service()

with st.sidebar:
    st.markdown("## 🎓 EduPredict-XAI")
    st.caption("Explainable student performance prediction")
    colour = "#14B8A6" if service.model_name == "TabPFN" else "#F59E0B"
    st.markdown(badge(f"Model: {service.model_name}", colour), unsafe_allow_html=True)
    if service.note:
        st.caption("Running on the CatBoost fallback model (TabPFN file not found). See README to enable TabPFN.")
    st.divider()

pages = [
    st.Page("views/dashboard.py", title="Dashboard", icon="📊", default=True),
    st.Page("views/predict.py", title="Predict & Explain", icon="🔮"),
    st.Page("views/batch.py", title="Class / Batch Analysis", icon="📂"),
    st.Page("views/research_insights.py", title="Research Insights", icon="🧠"),
    st.Page("views/about.py", title="About", icon="ℹ️"),
]
st.navigation(pages).run()
