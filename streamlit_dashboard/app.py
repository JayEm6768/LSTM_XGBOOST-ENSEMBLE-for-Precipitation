"""
Panel-defense dashboard for the LSTM-XGBoost precipitation ensemble.

Run locally with:
    pip install -r requirements.txt
    streamlit run app.py

The dashboard itself is a self-contained HTML/CSS/JS file (metrics_dashboard.html,
the same one used as the Claude Artifact) embedded directly in the page. Editing
the numbers means editing that file's `models` / `BASELINE` objects near the
bottom of its <script> block; this app.py does not duplicate the data.
"""

from pathlib import Path

import streamlit as st

st.set_page_config(
    page_title="Precipitation Model Scorecard",
    page_icon="🌧️",
    layout="wide",
)

# Streamlit's default page padding fights the dashboard's own layout; strip it
# so the embedded page reads as the whole screen, not a page-inside-a-page.
st.markdown(
    """
    <style>
        .block-container { padding: 0 !important; max-width: 100% !important; }
        iframe { border: none; }
        header[data-testid="stHeader"] { background: transparent; }
    </style>
    """,
    unsafe_allow_html=True,
)

html_path = Path(__file__).parent / "metrics_dashboard.html"
html = html_path.read_text(encoding="utf-8")

st.components.v1.html(html, height=3050, scrolling=True)
