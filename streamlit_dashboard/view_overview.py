"""Landing page: what the project is, how the model works, how to use the site."""

import streamlit as st

import inference
from theme import page_header


@st.cache_resource(show_spinner=False)
def _artifacts():
    return inference.load_artifacts()


def _model_version_card() -> str:
    """Describe the model actually loaded, read from its own metadata, rather
    than a fixed description that goes stale when the model changes."""
    try:
        s = inference.training_summary(_artifacts())
    except Exception:
        return (
            '<div class="app-card"><h3>Model version in use</h3>'
            "<p>The saved model artifacts could not be read, so the configuration cannot be "
            "described here. The Forecast page will report what is missing.</p></div>"
        )

    lc, ens, pre = s["lstm_config"], s["ensemble"], s["preprocessing"]
    ev = s["evaluation"].get("ensemble", {})
    base = s.get("baseline", {})

    headline = ""
    if ev.get("rmse") and base.get("rmse"):
        gain = (float(base["rmse"]) - float(ev["rmse"])) / float(base["rmse"]) * 100
        headline = (
            f"<p>On its held out test set this ensemble reaches an RMSE of "
            f"<b>{float(ev['rmse']):.4f}</b>, <b>{gain:.2f}%</b> below the single model baseline "
            f"of {float(base['rmse']):.4f}.</p>"
        )

    searched = (
        "<li>XGBoost hyperparameters were <b>selected by a search</b> on a validation holdout, "
        "not set by hand.</li>"
        if s.get("search")
        else "<li>XGBoost hyperparameters are fixed to a previously validated configuration.</li>"
    )
    horizon = (
        "<li>Includes <b>per lead day models</b>, so the Forecast page can show a seven day "
        "outlook alongside the one day forecast.</li>"
        if s.get("multi_horizon")
        else ""
    )

    return f"""
<div class="app-card">
<h3>Model version in use</h3>
<p>The site is serving the <b>{inference.ACTIVE_MODEL_NAME}</b> artifacts.</p>
{headline}
<ul>
<li><b>LSTM:</b> two layers of {lc.get('units_layer1','?')} and {lc.get('units_layer2','?')} units,
dropout {lc.get('dropout','?')}, {lc.get('window_size','?')} day lookback.</li>
<li><b>XGBoost:</b> {s['xgb_description']}, same {lc.get('window_size','?')} day lookback as lag
features.</li>
<li><b>Ensemble:</b> {ens.get('strategy','inverse RMSE weighting')}, weights
{float(ens.get('weight_lstm', float('nan'))):.4f} LSTM and
{float(ens.get('weight_xgb', float('nan'))):.4f} XGBoost, derived from validation error rather
than assumed.</li>
<li><b>Preprocessing:</b> Box Cox with a fixed lambda of
{float(pre.get('box_cox_lambda', float('nan'))):.4f}, then {pre.get('scaler_type','scaling')} for
the LSTM.</li>
{searched}
{horizon}
</ul>
<p>Every value above is read from the saved model's own metadata, so this card stays correct if a
different model is loaded. Nothing is tuned inside this website.</p>
</div>
"""


def render():
    page_header(
        eyebrow="Precipitation Forecasting Thesis",
        title="An LSTM plus XGBoost Ensemble for Daily Precipitation",
        dek=(
            "This site is a front end for a trained ensemble model that forecasts daily "
            "precipitation. It loads a model that was already trained and evaluated offline "
            "and uses it only to make predictions. No training happens here."
        ),
    )

    col1, col2 = st.columns([3, 2], gap="large")

    with col1:
        st.markdown(
            """
<div class="app-card">
<h3>How the model works</h3>
<p>The forecast comes from two component models combined into one ensemble.</p>
<ul>
<li><span class="badge lstm">LSTM</span> a recurrent neural network that reads the
prior seventy days of precipitation as a sequence and learns temporal patterns
across that window.</li>
<li><span class="badge xgb">XGBoost</span> a gradient boosted tree model that reads
the same seventy prior days as a flat set of lag features rather than a sequence.</li>
<li><span class="badge ensemble">Ensemble</span> the two outputs are combined using
fixed weights set by each model's inverse error on a held out validation slice
during training. The weights are locked and are not refit here.</li>
</ul>
<p>Before either model sees the data, the raw precipitation series goes through a
Box Cox transform (with a lambda value fixed during training) to make the
distribution better behaved, then the transformed values are scaled to a zero to
one range for the LSTM. XGBoost works on the Box Cox values directly since tree
models do not need a bounded scale. Every prediction is transformed back to
inches before it is shown.</p>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(
            """
<div class="app-card">
<h3>What this website does and does not do</h3>
<p>The Forecast page lets you upload your own daily precipitation CSV and get
predictions from the saved model. It runs inference only: it loads the LSTM
weights, the XGBoost model, the fitted scaler, and the Box Cox lambda exactly as
they were saved after training, and applies them to your file in the same order
the training notebook uses. It never retrains, refits a scaler, or re-estimates a
lambda from your data.</p>
<p>The Performance page shows the official evaluation numbers for the current
model version, measured on its held out test set during training, so you can see
how the model performs in general before trusting a forecast on your own data.</p>
</div>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.markdown(
            """
<div class="app-card">
<h3>What your file needs</h3>
<p>A CSV with two columns:</p>
<ul>
<li><b>DATE</b>, one row per day, in order, with no repeated or skipped days</li>
<li><b>PRCP</b>, precipitation in inches for that day (use zero for a dry day)</li>
</ul>
<p>At least seventy consecutive days are required, since each forecast looks back
over the prior seventy days of data. More history lets the site also check the
model's predictions against days you already know the outcome for.</p>
</div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown(_model_version_card(), unsafe_allow_html=True)

    st.markdown(
        """
<div class="note-box">
This site is inference only. Every number you see either comes directly from a
saved model artifact and its recorded evaluation metrics, or from running that
saved model on data you upload. Nothing shown here is invented or estimated on
the spot.
</div>
        """,
        unsafe_allow_html=True,
    )
