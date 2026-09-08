"""
Shared visual theme for the native Streamlit pages, kept visually consistent
with metrics_dashboard.html (same color roles, same type system). Import
apply_theme() once at the top of app.py.
"""

import streamlit as st

PALETTE = {
    "lstm": "#2a78d6",
    "xgb": "#eb6834",
    "ensemble": "#1baf7a",
    "good": "#0ca30c",
    "critical": "#d03b3b",
}

CSS = """
<style>
:root {
  --surface-1: #fcfcfb;
  --page-plane: #f9f9f7;
  --text-primary: #0b0b0b;
  --text-secondary: #52514e;
  --text-muted: #898781;
  --gridline: #e1e0d9;
  --border: rgba(11,11,11,0.10);
  --card-shadow: 0 1px 2px rgba(11,11,11,0.04), 0 1px 12px rgba(11,11,11,0.03);
  --lstm: #2a78d6;
  --xgb: #eb6834;
  --ensemble: #1baf7a;
  --good: #0ca30c;
  --critical: #d03b3b;
}
@media (prefers-color-scheme: dark) {
  :root {
    --surface-1: #1a1a19;
    --page-plane: #0d0d0d;
    --text-primary: #ffffff;
    --text-secondary: #c3c2b7;
    --text-muted: #898781;
    --gridline: #2c2c2a;
    --border: rgba(255,255,255,0.10);
    --card-shadow: 0 1px 2px rgba(0,0,0,0.3), 0 1px 16px rgba(0,0,0,0.25);
    --lstm: #3987e5;
    --xgb: #d95926;
    --ensemble: #199e70;
    --good: #0ca30c;
    --critical: #e66767;
  }
}

html, body, [class*="css"], .stApp {
  font-family: "IBM Plex Sans", "Segoe UI", system-ui, -apple-system, sans-serif;
}
code, .stCode, [data-testid="stMetricValue"] {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
}

.stApp {
  background: var(--page-plane);
  color: var(--text-primary);
}

section[data-testid="stSidebar"] {
  background: var(--surface-1);
  border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * {
  color: var(--text-primary);
}

.block-container {
  max-width: 1180px;
  padding-top: 2.4rem;
  padding-bottom: 4rem;
}

.eyebrow {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 11.5px;
  letter-spacing: 0.09em;
  text-transform: uppercase;
  color: var(--text-muted);
  margin: 0 0 6px 0;
}

h1, h2, h3 {
  letter-spacing: -0.01em;
  font-weight: 600;
}

.dek {
  color: var(--text-secondary);
  font-size: 15.5px;
  line-height: 1.6;
  max-width: 72ch;
}

.app-card {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 22px 24px;
  box-shadow: var(--card-shadow);
  margin-bottom: 18px;
}

.app-card h3 {
  margin-top: 0;
  margin-bottom: 8px;
  font-size: 16px;
}

.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 12px;
  padding: 4px 10px;
  border-radius: 999px;
  border: 1px solid var(--border);
  color: var(--text-secondary);
}
.badge.lstm { border-color: var(--lstm); color: var(--lstm); }
.badge.xgb { border-color: var(--xgb); color: var(--xgb); }
.badge.ensemble { border-color: var(--ensemble); color: var(--ensemble); }

.swatch-dot {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 2px;
  margin-right: 6px;
  vertical-align: middle;
}

hr.divider {
  border: none;
  border-top: 1px solid var(--border);
  margin: 28px 0;
}

.note-box {
  border: 1px solid var(--border);
  border-left: 3px solid var(--text-muted);
  background: color-mix(in oklab, var(--text-muted) 8%, transparent);
  border-radius: 8px;
  padding: 14px 16px;
  font-size: 13.5px;
  color: var(--text-secondary);
  line-height: 1.55;
}

.note-box.critical {
  border-left-color: var(--critical);
}

[data-testid="stMetric"] {
  background: var(--surface-1);
  border: 1px solid var(--border);
  border-radius: 12px;
  padding: 14px 16px;
}

.stButton > button, .stDownloadButton > button {
  border-radius: 8px;
  font-weight: 500;
}

footer, #MainMenu {
  visibility: hidden;
}
</style>
"""


def apply_theme():
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(eyebrow: str, title: str, dek: str = ""):
    parts = [f'<p class="eyebrow">{eyebrow}</p>', f"<h1>{title}</h1>"]
    if dek:
        parts.append(f'<p class="dek">{dek}</p>')
    st.markdown("".join(parts), unsafe_allow_html=True)
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
