"""
Panel facing website for the LSTM plus XGBoost precipitation ensemble.

Run locally with:
    cd streamlit_dashboard
    pip install -r requirements.txt
    streamlit run app.py

Structure:
    app.py            entry point and navigation
    theme.py          shared CSS and header helper, matches metrics_dashboard.html
    inference.py       loads saved model artifacts and runs inference only, no training
    view_overview.py   landing page
    view_forecast.py   upload a CSV and get a forecast from the saved ensemble
    view_performance.py  live evaluation computed from the uploaded file, with a
                          separate training time reference scorecard
"""

import streamlit as st

import view_forecast
import view_overview
import view_performance
from inference import ACTIVE_MODEL_NAME
from theme import apply_theme

st.set_page_config(
    page_title="Precipitation Ensemble",
    layout="wide",
)

apply_theme()

PAGES = {
    "Overview": view_overview,
    "Forecast": view_forecast,
    "Performance": view_performance,
}

with st.sidebar:
    st.markdown("### Precipitation Ensemble")
    st.caption("LSTM plus XGBoost, inference only site")
    choice = st.radio("Section", list(PAGES.keys()), label_visibility="collapsed")
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.caption(
        f"Loaded model artifacts: **{ACTIVE_MODEL_NAME}** (from models/{ACTIVE_MODEL_NAME}/). "
        "This site loads a model that was already trained elsewhere; it does not train "
        "or tune anything itself."
    )

PAGES[choice].render()
