"""Upload page: user provides a CSV, site runs the saved ensemble on it."""

import io

import numpy as np
import pandas as pd
import streamlit as st

import inference
from theme import PALETTE, page_header

STATE_RESULT = "forecast_result"
STATE_CLEAN_DF = "forecast_clean_df"
STATE_FILENAME = "forecast_filename"


@st.cache_resource(show_spinner=False)
def _cached_artifacts():
    return inference.load_artifacts()


def _sample_csv_bytes() -> bytes:
    dates = pd.date_range("2020-01-01", periods=90, freq="D")
    rng = np.random.default_rng(7)
    values = np.round(np.clip(rng.exponential(scale=0.15, size=len(dates)) - 0.1, 0, None), 2)
    sample = pd.DataFrame({"DATE": dates.strftime("%d/%m/%Y"), "PRCP": values})
    buf = io.StringIO()
    sample.to_csv(buf, index=False)
    return buf.getvalue().encode("utf-8")


def _clear_forecast_state():
    for key in (STATE_RESULT, STATE_CLEAN_DF, STATE_FILENAME):
        st.session_state.pop(key, None)


def _render_results(clean_df: pd.DataFrame, result: inference.InferenceResult, artifacts, filename: str):
    gap_note = inference.gap_summary(clean_df)
    if gap_note:
        st.markdown(f'<div class="note-box">{gap_note}</div>', unsafe_allow_html=True)

    st.success(
        f"Loaded {len(clean_df)} days of data from {filename}, from "
        f"{clean_df['DATE'].iloc[0].date()} to {clean_df['DATE'].iloc[-1].date()}."
    )

    st.button(
        "Upload a different file",
        on_click=_clear_forecast_state,
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("Forecast for the next day")

    fc_col1, fc_col2, fc_col3, fc_col4 = st.columns(4)
    fc_col1.metric("Forecast date", str(result.forecast_date.date()))
    fc_col2.metric("LSTM estimate (in)", f"{max(result.forecast_lstm, 0):.2f}")
    fc_col3.metric("XGBoost estimate (in)", f"{max(result.forecast_xgb, 0):.2f}")
    fc_col4.metric("Ensemble forecast (in)", f"{max(result.forecast_ensemble, 0):.2f}")

    st.caption(
        f"Ensemble forecast combines the two estimates using the locked weights "
        f"(LSTM {artifacts.weight_lstm:.2f}, XGBoost {artifacts.weight_xgb:.2f}) set during "
        "training. Negative raw outputs from the inverse transform are shown as zero, since "
        "precipitation cannot be negative."
    )

    if len(result.backtest_dates) == 0:
        st.markdown(
            '<div class="note-box">The file has exactly enough days for one forecast, so '
            "there is no earlier stretch of data left to check the model against. Upload a "
            "longer file to also see how the model performs on days you already know the "
            "outcome for.</div>",
            unsafe_allow_html=True,
        )
        return

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("How the model performs on your uploaded history")
    st.markdown(
        '<p class="dek">For every day in your file that has a full seventy day window of '
        "history before it, the model produces a prediction that can be compared against the "
        "actual value also present in your file. This is a walk forward check, not a training "
        "step.</p>",
        unsafe_allow_html=True,
    )

    chart_df = pd.DataFrame(
        {
            "Date": pd.to_datetime(result.backtest_dates),
            "Actual": result.backtest_actual,
            "LSTM": np.clip(result.backtest_lstm, 0, None),
            "XGBoost": np.clip(result.backtest_xgb, 0, None),
            "Ensemble": np.clip(result.backtest_ensemble, 0, None),
        }
    ).set_index("Date")

    st.line_chart(
        chart_df,
        color=["#898781", PALETTE["lstm"], PALETTE["xgb"], PALETTE["ensemble"]],
        height=340,
    )

    metrics = inference.evaluate_all(result.backtest_actual, result.backtest_ensemble)
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("RMSE (in)", f"{metrics['rmse']:.3f}")
    m2.metric("MAE (in)", f"{metrics['mae']:.3f}")
    mape_display = "not available" if np.isnan(metrics["mape"]) else f"{metrics['mape']:.1f}%"
    m3.metric("MAPE", mape_display, help="Computed only on days with at least 0.1 inch of actual rain.")
    m4.metric("R2", f"{metrics['r2']:.3f}")
    m5.metric("Bias (in)", f"{metrics['bias']:+.3f}", help="Mean of predicted minus actual. Positive means the model tends to over predict.")

    st.markdown("##### Error by rainfall intensity")
    intensity_df = inference.rainfall_intensity_performance(result.backtest_actual, result.backtest_ensemble)
    st.dataframe(intensity_df, hide_index=True, use_container_width=True)

    with st.expander("View the day by day prediction table"):
        table = chart_df.reset_index()
        table["Date"] = table["Date"].dt.date
        st.dataframe(table, hide_index=True, use_container_width=True)
        csv_bytes = table.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download these results as a CSV",
            data=csv_bytes,
            file_name="forecast_results.csv",
            mime="text/csv",
        )


def render():
    page_header(
        eyebrow="Run The Model",
        title="Upload Precipitation Data And Get A Forecast",
        dek=(
            "Upload a daily precipitation CSV and the site will run the saved LSTM plus "
            "XGBoost ensemble on it. The model is loaded from disk exactly as it was saved "
            "after training. Nothing is retrained or refit here."
        ),
    )

    missing = inference.missing_artifacts()
    if missing:
        st.markdown(
            f"""
<div class="note-box critical">
<b>The saved model files needed for this page are not available.</b><br>
Missing: {"; ".join(missing)}.<br><br>
This page calls a model that must already be trained and saved by the project
pipeline; it cannot train one itself. Run the training notebook
(LSTM_XGBoost_Ensemble_Precipitation_STATIC.ipynb) so that it writes the
models/v2_static artifacts, then reload this page.
</div>
            """,
            unsafe_allow_html=True,
        )
        return

    with st.spinner("Loading the saved model artifacts."):
        try:
            artifacts = _cached_artifacts()
        except inference.ArtifactsMissingError as exc:
            st.markdown(f'<div class="note-box critical">{exc}</div>', unsafe_allow_html=True)
            return

    # If a file was already uploaded and processed earlier in this session,
    # show those stored results instead of asking for the file again.
    if STATE_RESULT in st.session_state and STATE_CLEAN_DF in st.session_state:
        _render_results(
            st.session_state[STATE_CLEAN_DF],
            st.session_state[STATE_RESULT],
            artifacts,
            st.session_state.get(STATE_FILENAME, "your file"),
        )
        return

    left, right = st.columns([2, 1], gap="large")
    with right:
        st.markdown(
            f"""
<div class="app-card">
<h3>File format</h3>
<p>Two columns: <b>DATE</b> and <b>PRCP</b> (precipitation in inches). One row per
day, in order, no repeated or skipped days.</p>
<p>At least {artifacts.window_size} consecutive days are required, since every
forecast looks back over that many prior days.</p>
</div>
            """,
            unsafe_allow_html=True,
        )
        st.download_button(
            "Download a sample CSV",
            data=_sample_csv_bytes(),
            file_name="sample_precipitation.csv",
            mime="text/csv",
            use_container_width=True,
        )

    with left:
        uploaded = st.file_uploader("Upload a precipitation CSV", type=["csv"])

    if uploaded is None:
        st.info("Upload a file to see a forecast.")
        return

    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as exc:
        st.markdown(
            f'<div class="note-box critical">The file could not be read as a CSV. '
            f'Check that it is a plain comma separated file and try again. Details: {exc}</div>',
            unsafe_allow_html=True,
        )
        return

    try:
        clean_df = inference.validate_and_prepare(raw_df, artifacts.window_size)
    except inference.UploadValidationError as exc:
        st.markdown(f'<div class="note-box critical">{exc}</div>', unsafe_allow_html=True)
        return

    with st.spinner("Running the ensemble on your data."):
        result = inference.run_inference(clean_df, artifacts)

    st.session_state[STATE_CLEAN_DF] = clean_df
    st.session_state[STATE_RESULT] = result
    st.session_state[STATE_FILENAME] = uploaded.name

    _render_results(clean_df, result, artifacts, uploaded.name)
