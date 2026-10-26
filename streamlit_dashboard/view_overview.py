"""Landing page: what the project is, how the model works, how to use the site."""

import streamlit as st

from theme import page_header


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
