"""Shared UI kit: CSS, cards, colours, cached backend service."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.config import UPGRADE_DIR  # noqa: E402
from src.service import EduPredictService  # noqa: E402

INDIGO, TEAL, RED, AMBER, GREEN, SLATE = "#4F46E5", "#14B8A6", "#EF4444", "#F59E0B", "#22C55E", "#64748B"
BAND_COLORS = {"At Risk": RED, "Needs Improvement": AMBER, "Average": SLATE, "Good": TEAL, "Excellent": GREEN}
PRIORITY_COLORS = {"High": RED, "Medium": AMBER, "Low": SLATE}

CSS = """
<style>
.block-container{max-width:1280px;padding-top:1.4rem;padding-bottom:3rem}
header[data-testid="stHeader"]{background:transparent}
.hero{background:linear-gradient(120deg,#4F46E5 0%,#7C3AED 55%,#14B8A6 130%);color:#fff;border-radius:22px;
  padding:1.6rem 2rem;margin-bottom:1.3rem;box-shadow:0 10px 30px rgba(79,70,229,.25)}
.hero h1{margin:0;font-size:2rem;font-weight:800;letter-spacing:-.5px;color:#fff}
.hero p{margin:.35rem 0 0;opacity:.9;font-size:1.02rem;max-width:900px}
.hero .chip{display:inline-block;background:rgba(255,255,255,.18);border-radius:999px;padding:.15rem .8rem;
  font-size:.78rem;margin-bottom:.6rem;font-weight:600}
.card{background:#fff;border:1px solid #E3E6F3;border-radius:18px;padding:1.05rem 1.2rem;
  box-shadow:0 2px 10px rgba(30,33,64,.05);height:100%}
.kpi .lbl{font-size:.78rem;text-transform:uppercase;letter-spacing:.06em;color:#6B7194;font-weight:700}
.kpi .val{font-size:1.9rem;font-weight:800;line-height:1.15;margin:.15rem 0}
.kpi .sub{font-size:.82rem;color:#6B7194}
.badge{display:inline-block;border-radius:999px;padding:.12rem .7rem;font-size:.75rem;font-weight:700;color:#fff}
.action{background:#fff;border:1px solid #E3E6F3;border-left:6px solid var(--c);border-radius:14px;
  padding:.85rem 1.1rem;margin-bottom:.7rem}
.action .t{font-weight:750;margin-bottom:.15rem}
.action .g{font-size:.85rem;color:#6B7194}
.callout{border-radius:14px;padding:.85rem 1.1rem;margin:.4rem 0 1rem;background:#EEF0FF;border:1px solid #D9DDFB;line-height:1.55}
.callout.warn{background:#FFF7E6;border-color:#FFE2A8}
.callout.good{background:#EAFBF3;border-color:#BDEFD6}
.callout.bad{background:#FEEEEE;border-color:#F8C9C9}
.step{text-align:center}.step .n{width:34px;height:34px;border-radius:50%;background:#4F46E5;color:#fff;
  display:inline-flex;align-items:center;justify-content:center;font-weight:800;margin-bottom:.4rem}
.sec{font-size:1.25rem;font-weight:800;margin:1.2rem 0 .2rem;letter-spacing:-.2px}
.seccap{color:#6B7194;margin-bottom:.8rem;font-size:.93rem}
div.stButton>button[kind="primary"]{border-radius:12px;font-weight:700;padding:.6rem 1rem}
button[role="tab"]{font-weight:650}
[data-testid="stSidebar"]{border-right:1px solid #E3E6F3}
</style>
"""


def inject_css() -> None:
    st.markdown(CSS, unsafe_allow_html=True)


def hero(title: str, subtitle: str, chip: str = "") -> None:
    chip_html = f'<div class="chip">{chip}</div><br>' if chip else ""
    st.markdown(f'<div class="hero">{chip_html}<h1>{title}</h1><p>{subtitle}</p></div>', unsafe_allow_html=True)


def section(title: str, caption: str = "") -> None:
    st.markdown(f'<div class="sec">{title}</div>' + (f'<div class="seccap">{caption}</div>' if caption else ""),
                unsafe_allow_html=True)


def kpi(label: str, value: str, sub: str = "", color: str = "#1E2140") -> str:
    return (f'<div class="card kpi"><div class="lbl">{label}</div><div class="val" style="color:{color}">{value}</div>'
            f'<div class="sub">{sub}</div></div>')


def kpi_row(items: list[tuple]) -> None:
    for col, it in zip(st.columns(len(items), gap="medium"), items):
        col.markdown(kpi(*it), unsafe_allow_html=True)


def badge(text: str, color: str) -> str:
    return f'<span class="badge" style="background:{color}">{text}</span>'


def callout(text: str, kind: str = "info") -> None:
    st.markdown(f'<div class="callout {kind}">{text}</div>', unsafe_allow_html=True)


def style_fig(fig, height: int = 380):
    fig.update_layout(height=height, margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white",
                      paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#1E2140"),
                      legend=dict(orientation="h", y=-0.2))
    fig.update_xaxes(gridcolor="#EEF0F8")
    fig.update_yaxes(gridcolor="#EEF0F8")
    return fig


@st.cache_resource(show_spinner="Loading prediction model...")
def get_service() -> EduPredictService:
    return EduPredictService()


@st.cache_data
def read_result(name: str) -> pd.DataFrame | None:
    p = UPGRADE_DIR / name
    return pd.read_csv(p) if p.exists() else None


def missing_results_notice() -> None:
    callout("Result files not found. Run <code>python run_all_upgrades.py</code> once to generate them.", "warn")
