"""Performance page: real evaluation numbers computed live from whatever CSV
the user uploaded on the Forecast page, not the static training time
scorecard. The training time numbers in metrics_dashboard.html are still
offered, but only as a clearly separate reference section."""

from pathlib import Path

import numpy as np
import streamlit as st

import inference
from theme import PALETTE, page_header

from view_forecast import STATE_RESULT


def _mape_text(value: float) -> str:
    return "not available" if np.isnan(value) else f"{value:.1f}%"


def _model_card(name: str, color: str, metrics: dict) -> str:
    return f"""
<div class="app-card">
<span class="badge" style="border-color:{color};color:{color};">
<span class="swatch-dot" style="background:{color};"></span>{name}</span>
<div style="margin-top:14px; display:grid; grid-template-columns: 1fr 1fr; gap:10px 18px;">
  <div><div class="eyebrow" style="margin-bottom:2px;">RMSE (in)</div><div style="font-family:'IBM Plex Mono',monospace; font-size:20px;">{metrics['rmse']:.3f}</div></div>
  <div><div class="eyebrow" style="margin-bottom:2px;">MAE (in)</div><div style="font-family:'IBM Plex Mono',monospace; font-size:20px;">{metrics['mae']:.3f}</div></div>
  <div><div class="eyebrow" style="margin-bottom:2px;">MAPE</div><div style="font-family:'IBM Plex Mono',monospace; font-size:20px;">{_mape_text(metrics['mape'])}</div></div>
  <div><div class="eyebrow" style="margin-bottom:2px;">R2</div><div style="font-family:'IBM Plex Mono',monospace; font-size:20px;">{metrics['r2']:.3f}</div></div>
  <div><div class="eyebrow" style="margin-bottom:2px;">Bias (in)</div><div style="font-family:'IBM Plex Mono',monospace; font-size:20px;">{metrics['bias']:+.3f}</div></div>
</div>
</div>
"""


def render():
    page_header(
        eyebrow="Evaluation",
        title="Model Performance On Your Uploaded Data",
        dek=(
            "These numbers are computed live from the file you uploaded on the Forecast page, "
            "using a walk forward check over your own history. They are not read from a fixed "
            "file and are not the training time scorecard."
        ),
    )

    result = st.session_state.get(STATE_RESULT)
    has_backtest = (
        result is not None
        and getattr(result, "backtest_actual", None) is not None
        and len(result.backtest_actual) > 0
    )

    if not has_backtest:
        st.markdown(
            """
<div class="note-box">
Performance on this page is computed from the file you upload on the Forecast page, not from a
fixed reference file. Once you upload a file there that is long enough to hold out an earlier
stretch of days for comparison, real per model metrics will appear here automatically.
<br><br>
Go to the Forecast section in the sidebar, upload a precipitation CSV, and return here.
</div>
            """,
            unsafe_allow_html=True,
        )
        _render_reference_scorecard()
        return

    actual = result.backtest_actual
    metrics_lstm = inference.evaluate_all(actual, result.backtest_lstm)
    metrics_xgb = inference.evaluate_all(actual, result.backtest_xgb)
    metrics_ensemble = inference.evaluate_all(actual, result.backtest_ensemble)

    st.caption(
        f"Computed over {len(actual)} day(s) of walk forward comparison from your uploaded file."
    )

    c1, c2, c3 = st.columns(3, gap="medium")
    with c1:
        st.markdown(_model_card("LSTM", PALETTE["lstm"], metrics_lstm), unsafe_allow_html=True)
    with c2:
        st.markdown(_model_card("XGBoost", PALETTE["xgb"], metrics_xgb), unsafe_allow_html=True)
    with c3:
        st.markdown(_model_card("Ensemble", PALETTE["ensemble"], metrics_ensemble), unsafe_allow_html=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("Error by rainfall intensity")
    st.markdown(
        '<p class="dek">Each row groups days from your uploaded file by how much rain actually '
        "fell, so error can be checked separately for dry days, light rain, moderate rain, and "
        "heavy rain.</p>",
        unsafe_allow_html=True,
    )

    tab_lstm, tab_xgb, tab_ensemble = st.tabs(["LSTM", "XGBoost", "Ensemble"])
    with tab_lstm:
        st.dataframe(
            inference.rainfall_intensity_performance(actual, result.backtest_lstm),
            hide_index=True,
            use_container_width=True,
        )
    with tab_xgb:
        st.dataframe(
            inference.rainfall_intensity_performance(actual, result.backtest_xgb),
            hide_index=True,
            use_container_width=True,
        )
    with tab_ensemble:
        st.dataframe(
            inference.rainfall_intensity_performance(actual, result.backtest_ensemble),
            hide_index=True,
            use_container_width=True,
        )

    _render_reference_scorecard()


def _render_reference_scorecard():
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    with st.expander("Training time reference scorecard"):
        st.markdown(
            '<p class="dek">The numbers below come from evaluating the saved model on its held '
            "out test set during training. They are read directly from the project's recorded "
            "metadata and do not reflect anything uploaded on this site.</p>",
            unsafe_allow_html=True,
        )
        html_path = Path(__file__).parent / "metrics_dashboard.html"
        if not html_path.exists():
            st.markdown(
                '<div class="note-box critical">metrics_dashboard.html was not found next to '
                "app.py, so the reference scorecard cannot be shown.</div>",
                unsafe_allow_html=True,
            )
            return
        html = html_path.read_text(encoding="utf-8")
        st.components.v1.html(html, height=3050, scrolling=True)
