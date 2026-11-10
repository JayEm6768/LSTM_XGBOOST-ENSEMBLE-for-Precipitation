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
  color-scheme: light;

  --code-font: "Cascadia Code", "Cascadia Mono", Consolas, ui-monospace, "SF Mono",
             "Segoe UI Mono", Menlo, monospace;

  --ink:       #0c1526;
  --title:     #0c1526;
  --ink-2:     #42506b;
  --ink-3:     #74819a;

  --page-top:    #f3f7fb;
  --page-mid:    #e9f0f8;
  --page-bottom: #eef3fa;

  --glass:          rgba(255, 255, 255, 0.66);
  --glass-strong:   rgba(255, 255, 255, 0.86);
  --glass-border:   rgba(20, 40, 70, 0.10);
  --glass-hairline: rgba(255, 255, 255, 0.95);

  --glow-lstm: rgba(64, 140, 226, 0.30);
  --glow-xgb:  rgba(235, 140, 80, 0.24);
  --glow-ens:  rgba(27, 175, 122, 0.24);
  --vignette:  rgba(10, 22, 45, 0.16);

  --lstm:     #2a78d6;
  --xgb:      #eb6834;
  --ensemble: #1baf7a;
  --good:     #0ca30c;
  --critical: #d03b3b;

  --shadow-sm: 0 1px 2px rgba(12, 21, 38, 0.06);
  --shadow-md: 0 24px 56px -28px rgba(12, 21, 38, 0.40);
  --app-radius: 16px;
}

@media (prefers-color-scheme: dark) {
  :root {
    color-scheme: dark;

    --ink:       #eef3ff;
    --title:     #ffffff;
    --ink-2:     #a9b6d0;
    --ink-3:     #6d7d99;

    --page-top:    #0a0f1e;
    --page-mid:    #0b1122;
    --page-bottom: #070b15;

    --glass:          rgba(16, 22, 40, 0.62);
    --glass-strong:   rgba(21, 29, 52, 0.88);
    --glass-border:   rgba(150, 182, 255, 0.14);
    --glass-hairline: rgba(160, 195, 255, 0.22);

    --glow-lstm: rgba(57, 135, 229, 0.26);
    --glow-xgb:  rgba(217, 89, 38, 0.18);
    --glow-ens:  rgba(25, 158, 112, 0.20);
    --vignette:  rgba(2, 5, 12, 0.36);

    --lstm:     #3987e5;
    --xgb:      #d95926;
    --ensemble: #199e70;
    --good:     #0ca30c;
    --critical: #e66767;

    --shadow-sm: 0 1px 2px rgba(0, 0, 0, 0.45);
    --shadow-md: 0 26px 60px -30px rgba(0, 0, 0, 0.85);
  }
}

/* ---- 2. Atmosphere: colour fields, grain, vignette, slow drift ------- */

html {
  background: var(--page-mid);
}

body {
  background: transparent;
}

/* The "sky": soft colour fields that drift slowly behind the whole app. */
body::before {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  will-change: background-position;
  background-image:
    radial-gradient(42rem 30rem at 14% 8%,   var(--glow-lstm), transparent 62%),
    radial-gradient(40rem 28rem at 88% 6%,   var(--glow-xgb),  transparent 62%),
    radial-gradient(46rem 36rem at 52% 118%, var(--glow-ens),  transparent 64%),
    radial-gradient(130% 96% at 50% 42%, transparent 52%, var(--vignette) 100%),
    linear-gradient(180deg, var(--page-top) 0%, var(--page-mid) 45%, var(--page-bottom) 100%);
  background-size:
    2200px 1600px,
    2100px 1500px,
    2300px 1800px,
    100% 100%,
    100% 100%;
  animation: atmosphere-drift 80s ease-in-out infinite alternate;
}

/* Fine film grain: kills gradient banding and adds texture/depth. */
body::after {
  content: "";
  position: fixed;
  inset: 0;
  z-index: -1;
  pointer-events: none;
  opacity: 0.05;
  mix-blend-mode: overlay;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='180' height='180'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.9' numOctaves='2' stitchTiles='stitch'/%3E%3C/filter%3E%3Crect width='100%25' height='100%25' filter='url(%23n)'/%3E%3C/svg%3E");
}

@keyframes atmosphere-drift {
  0%   { background-position: 6% 4%, 90% 2%, 52% 120%, center, center; }
  100% { background-position: 16% 12%, 80% 12%, 60% 110%, center, center; }
}

/* ---- 3. Base typography ----------------------------------------------- */

html, body, .stApp, [class*="css"] {
  font-family: var(--code-font);
  -webkit-font-smoothing: antialiased;
  text-rendering: optimizeLegibility;
}

.stApp,
[data-testid="stAppViewContainer"] {
  background: transparent !important;
  color: var(--ink);
}

.block-container {
  max-width: 1180px;
  padding-top: 2.8rem;
  padding-bottom: 5rem;
}

h1, h2, h3, h4, h5 {
  color: var(--ink);
  font-weight: 600;
  letter-spacing: -0.015em;
}

h1 {
  color: var(--title);
  font-weight: 700;
  font-size: clamp(1.7rem, 3vw, 2.3rem);
  line-height: 1.12;
  letter-spacing: -0.025em;
}


[data-testid="stMarkdownContainer"] p {
  color: var(--ink-2);
  line-height: 1.75;
}

[data-testid="stCaptionContainer"] {
  color: var(--ink-3);
  font-size: 12px;
  line-height: 1.6;
}
[data-testid="stCaptionContainer"] strong {
  color: var(--ink-2);
}

code {
  font-family: var(--code-font);
  font-size: 0.9em;
  background: color-mix(in srgb, var(--ink) 8%, transparent);
  padding: 2px 6px;
  border-radius: 6px;
}

::selection {
  background: color-mix(in srgb, var(--lstm) 28%, transparent);
}

:focus-visible {
  outline: 2px solid var(--lstm);
  outline-offset: 2px;
  border-radius: 4px;
}

/* ---- 4. Page furniture --------------------------------------------------- */

.eyebrow {
  font-size: 11px;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--ink-3);
  font-weight: 500;
  margin: 0 0 8px 0;
}

.dek {
  color: var(--ink-2);
  font-size: 14.5px;
  line-height: 1.75;
  max-width: 74ch;
}

hr.divider {
  border: none;
  height: 1px;
  margin: 30px 0;
  background: linear-gradient(90deg, var(--glass-border), var(--ink-3), transparent 80%);
  opacity: 0.55;
}

/* ---- 5. Glass surfaces ------------------------------------------------- */

.app-card,
[data-testid="stMetric"],
[data-testid="stExpander"],
[data-testid="stAlert"] {
  position: relative;
  background: var(--glass);
  border: 1px solid var(--glass-border);
  border-radius: var(--app-radius);
  box-shadow: var(--shadow-sm), var(--shadow-md);
  backdrop-filter: blur(16px) saturate(150%);
  -webkit-backdrop-filter: blur(16px) saturate(150%);
}

.app-card {
  padding: 22px 24px;
  margin-bottom: 18px;
  overflow: hidden;
  transition: transform 0.22s cubic-bezier(0.2, 0.7, 0.3, 1),
              border-color 0.22s ease,
              box-shadow 0.22s ease;
}

.app-card h3 {
  margin: 0 0 10px 0;
  font-size: 32px;
  letter-spacing: -0.01em;
}
.app-card p, .app-card li {
  color: var(--ink-2);
}

.app-card:hover {
  transform: translateY(-3px);
  border-color: color-mix(in srgb, var(--glass-border) 55%, var(--lstm) 30%);
  box-shadow: var(--shadow-sm), 0 30px 64px -30px var(--glow-lstm);
}

/* frosted hairline along the top edge of each card */
.app-card::before {
  content: "";
  position: absolute;
  top: -1px;
  left: 8%;
  right: 8%;
  height: 1px;
  background: linear-gradient(90deg, transparent, var(--glass-hairline), transparent);
  pointer-events: none;
}

[data-testid="stMetric"] {
  padding: 18px 18px 16px;
  overflow: hidden;
}
[data-testid="stMetric"]::before {
  content: "";
  position: absolute;
  top: 0;
  left: 0;
  right: 0;
  height: 2px;
  background: linear-gradient(90deg, var(--lstm), var(--ensemble) 70%, var(--xgb));
  opacity: 0.5;
}
[data-testid="stMetricLabel"] {
  color: var(--ink-3);
  font-size: 11px;
  text-transform: uppercase;
  letter-spacing: 0.08em;
}
[data-testid="stMetricValue"] {
  font-family: var(--code-font);
  font-weight: 700;
  font-size: 1.55rem;
  letter-spacing: -0.02em;
}

[data-testid="stExpander"] {
  overflow: hidden;
}
[data-testid="stExpander"] summary {
  font-weight: 600;
  color: var(--ink);
}

[data-testid="stAlert"] {
  border-radius: 14px;
}
[data-testid="stAlert"] [data-testid="stAlertText"] {
  color: var(--ink-2);
}

.note-box {
  border: 1px solid var(--glass-border);
  border-left: 3px solid var(--ink-3);
  background: var(--glass);
  backdrop-filter: blur(14px) saturate(150%);
  -webkit-backdrop-filter: blur(14px) saturate(150%);
  border-radius: 12px;
  padding: 14px 16px;
  font-size: 13px;
  color: var(--ink-2);
  line-height: 1.65;
}
.note-box.critical {
  border-left-color: var(--critical);
  background: color-mix(in srgb, var(--critical) 9%, var(--glass));
}

/* ---- 6. Sidebar ----------------------------------------------------------- */

section[data-testid="stSidebar"] {
  background: linear-gradient(180deg, var(--glass-strong), var(--glass));
  border-right: 1px solid var(--glass-border);
  box-shadow: inset -1px 0 0 var(--glass-hairline);
  backdrop-filter: blur(20px) saturate(160%);
  -webkit-backdrop-filter: blur(20px) saturate(160%);
}

section[data-testid="stSidebar"] [data-testid="stSidebarContent"] {
  padding-top: 1.4rem;
}

/* brand block injected from app.py */
.brand {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 0 0 18px 0;
}
.brand-dot {
  width: 34px;
  height: 34px;
  flex: none;
  border-radius: 10px;
  position: relative;
  background:
    radial-gradient(circle at 32% 26%, rgba(255,255,255,0.55), transparent 46%),
    linear-gradient(135deg, var(--lstm), var(--ensemble));
  box-shadow: 0 10px 26px -12px var(--glow-lstm), inset 0 1px 0 rgba(255,255,255,0.45);
}
.brand-dot::after {
  content: "";
  position: absolute;
  inset: 9px;
  border-radius: 50%;
  border: 1.5px dashed rgba(255,255,255,0.8);
}
.brand-title {
  font-weight: 700;
  font-size: 13.5px;
  letter-spacing: -0.01em;
  color: var(--ink);
  line-height: 1.25;
}
.brand-sub {
  margin-top: 3px;
  font-size: 10px;
  color: var(--ink-3);
  letter-spacing: 0.04em;
  text-transform: uppercase;
}

/* navigation radio re-styled as stacked glass chips */
section[data-testid="stSidebar"] [role="radiogroup"] {
  display: grid;
  gap: 6px;
  padding: 2px 0 6px;
}
section[data-testid="stSidebar"] [data-baseweb="radio"] {
  display: flex;
  align-items: center;
  gap: 10px;
  padding: 10px 13px;
  border-radius: 11px;
  cursor: pointer;
  color: var(--ink-2);
  font-weight: 500;
  font-size: 13px;
  border: 1px solid transparent;
  background: color-mix(in srgb, var(--glass) 45%, transparent);
  transition: background 0.16s ease, color 0.16s ease,
              border-color 0.16s ease, box-shadow 0.16s ease, transform 0.16s ease;
}
section[data-testid="stSidebar"] [data-baseweb="radio"]:hover {
  color: var(--ink);
  border-color: var(--glass-border);
  background: var(--glass-strong);
  transform: translateX(2px);
}
section[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) {
  color: #ffffff;
  border-color: transparent;
  background: linear-gradient(115deg, var(--lstm), var(--ensemble));
  box-shadow: 0 12px 28px -14px var(--glow-lstm);
}
section[data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked):hover {
  transform: none;
}

section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
  color: var(--ink-3);
  font-size: 11.5px;
  line-height: 1.7;
}
section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] strong {
  color: var(--ink-2);
}

/* ---- 7. Controls --------------------------------------------------------- */

.stButton > button,
.stDownloadButton > button {
  border-radius: 11px;
  font-family: var(--code-font);
  font-weight: 600;
  color: var(--ink);
  border: 1px solid var(--glass-border);
  background: var(--glass-strong);
  box-shadow: var(--shadow-sm), inset 0 1px 0 var(--glass-hairline);
  backdrop-filter: blur(10px);
  -webkit-backdrop-filter: blur(10px);
  transition: all 0.18s ease;
}
.stButton > button:hover,
.stDownloadButton > button:hover {
  color: var(--lstm);
  border-color: color-mix(in srgb, var(--glass-border) 50%, var(--lstm) 60%);
  box-shadow: 0 12px 30px -16px var(--glow-lstm);
  transform: translateY(-1px);
}
.stButton > button[kind="primary"],
.stDownloadButton > button[kind="primary"] {
  color: #fff;
  border: none;
  background: linear-gradient(115deg, var(--lstm), var(--ensemble));
  box-shadow: 0 14px 34px -16px var(--glow-lstm);
}
.stButton > button[kind="primary"]:hover,
.stDownloadButton > button[kind="primary"]:hover {
  color: #fff;
  box-shadow: 0 18px 40px -16px var(--glow-lstm);
}

[data-testid="stFileUploaderDropzone"] {
  border: 1.5px dashed color-mix(in srgb, var(--lstm) 45%, var(--glass-border));
  border-radius: 14px;
  background: var(--glass);
  backdrop-filter: blur(12px);
  -webkit-backdrop-filter: blur(12px);
  transition: border-color 0.2s ease, box-shadow 0.2s ease, background 0.2s ease;
}
[data-testid="stFileUploaderDropzone"]:hover {
  border-color: var(--lstm);
  box-shadow: 0 14px 36px -18px var(--glow-lstm);
  background: var(--glass-strong);
}

[data-testid="stTabs"] [role="tablist"] {
  border-bottom: 1px solid var(--glass-border);
  gap: 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
  color: var(--ink-3);
  font-family: var(--code-font);
  font-weight: 500;
  border-radius: 9px 9px 0 0;
  transition: color 0.15s ease, background 0.15s ease;
}
[data-testid="stTabs"] [data-baseweb="tab"][aria-selected="true"] {
  color: var(--lstm);
  background: color-mix(in srgb, var(--glass) 60%, transparent);
}
[data-testid="stTabs"] [data-baseweb="tab"]:hover {
  color: var(--ink);
}

[data-testid="stDataFrame"] {
  border: 1px solid var(--glass-border);
  border-radius: 12px;
  overflow: hidden;
  box-shadow: var(--shadow-sm);
}
[data-testid="stDataFrame"] canvas {
  border-radius: 12px;
}

/* vega charts: transparent track so the atmosphere shows through */
[data-testid="stVegaLiteChart"] svg text,
[data-testid="stPlotlyChart"] svg text {
  font-family: var(--code-font) !important;
}
[data-testid="stVegaLiteChart"] rect.background,
[data-testid="stVegaLiteChart"] rect.plot-background {
  fill: transparent !important;
}
[data-testid="stVegaLiteChart"] svg {
  background: transparent;
}

/* ---- 8. Badges & callouts ------------------------------------------------ */

.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  font-size: 11.5px;
  font-weight: 600;
  padding: 4px 11px;
  border-radius: 999px;
  border: 1px solid var(--glass-border);
  color: var(--ink-2);
  background: var(--glass);
  backdrop-filter: blur(8px);
}
.badge.lstm     { border-color: color-mix(in srgb, var(--lstm) 60%, var(--glass-border)); color: var(--lstm); }
.badge.xgb      { border-color: color-mix(in srgb, var(--xgb) 60%, var(--glass-border)); color: var(--xgb); }
.badge.ensemble { border-color: color-mix(in srgb, var(--ensemble) 60%, var(--glass-border)); color: var(--ensemble); }

.swatch-dot {
  display: inline-block;
  width: 9px;
  height: 9px;
  border-radius: 3px;
  margin-right: 6px;
  vertical-align: middle;
}

/* ---- 9. Browser chrome ----------------------------------------------------- */

[data-testid="stHeader"] {
  background: transparent;
}

footer,
[data-testid="stFooter"],
#MainMenu {
  visibility: hidden;
}

::-webkit-scrollbar {
  width: 11px;
  height: 11px;
}
::-webkit-scrollbar-track {
  background: transparent;
}
::-webkit-scrollbar-thumb {
  background: color-mix(in srgb, var(--ink-3) 35%, transparent);
  border-radius: 999px;
  border: 3px solid transparent;
  background-clip: content-box;
}
::-webkit-scrollbar-thumb:hover {
  background: color-mix(in srgb, var(--ink-3) 55%, transparent);
  background-clip: content-box;
}

/* accessibility: honour reduced motion */
@media (prefers-reduced-motion: reduce) {
  *, *::before, *::after {
    animation: none !important;
    transition: none !important;
  }
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
