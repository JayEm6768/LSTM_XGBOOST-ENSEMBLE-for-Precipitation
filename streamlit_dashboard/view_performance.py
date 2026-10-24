"""Performance page: real evaluation numbers computed live from whatever CSV
the user uploaded on the Forecast page, not the static training time
scorecard. The training time numbers in metrics_dashboard.html are still
offered, but only as a clearly separate reference section."""

from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

import charts
import inference
from theme import PALETTE, page_header

from view_forecast import STATE_CLEAN_DF, STATE_RESULT


@st.cache_resource(show_spinner=False)
def _artifacts():
    return inference.load_artifacts()


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

    preds = {
        "LSTM": result.backtest_lstm,
        "XGBoost": result.backtest_xgb,
        "Ensemble": result.backtest_ensemble,
    }
    metrics = {"LSTM": metrics_lstm, "XGBoost": metrics_xgb, "Ensemble": metrics_ensemble}

    _render_accuracy_charts(actual, preds, metrics)
    _render_ensemble_metrics(actual, preds, metrics, st.session_state.get(STATE_CLEAN_DF))
    _render_agreement_charts(result, actual, preds, metrics)
    _render_regime_charts(actual, preds)
    _render_event_charts(actual, preds)
    _render_reference_scorecard()


def _render_ensemble_metrics(actual, preds, metrics, clean_df):
    """Metrics that specifically gauge an ensemble: is combining worth it, and
    is the result better than forecasts that need no model at all."""
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("Is the ensemble worth it?")
    st.markdown(
        '<p class="dek">A weighted blend of two forecasts always lands between them, so the fair '
        "test is whether it beats the <i>better</i> of its two components, metric by metric. "
        "Winning on some and losing on others is the normal result and is worth seeing.</p>",
        unsafe_allow_html=True,
    )
    gain = inference.ensemble_gain(metrics)
    st.dataframe(gain.round(4), hide_index=True, use_container_width=True)

    wins = (gain["Ensemble beats it"] == "yes").sum()
    total = (gain["Ensemble beats it"] != "").sum()
    st.caption(
        f"On this file the ensemble beats the better component on {wins} of {total} metrics. "
        "Its forecast spread sits between the two components by construction."
    )

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("Against forecasts that need no model")
    st.markdown(
        '<p class="dek">Skill is one minus the model error divided by the reference error. Zero '
        "means the model did exactly as well as that reference; below zero means worse. "
        "Persistence repeats yesterday, climatology uses the average of the history before the "
        "scored period, and the zero forecast assumes it never rains, which on a mostly dry "
        "record is a surprisingly hard baseline to beat on MAE.</p>",
        unsafe_allow_html=True,
    )
    refs = {}
    if clean_df is not None:
        try:
            refs = inference.reference_forecasts(clean_df, _artifacts().window_size)
        except Exception:
            refs = {}
    if not refs:
        st.caption("Reference forecasts could not be built from this file.")
    else:
        rows = []
        for model, p in preds.items():
            row = {"Model": model}
            for name, ref in refs.items():
                row[name] = inference.skill_score(actual, p, ref, "mse")
            rows.append(row)
        skill = pd.DataFrame(rows)
        st.altair_chart(charts.skill_score_chart(skill), use_container_width=False)
        st.dataframe(skill.round(4), hide_index=True, use_container_width=True)

        ref_err = pd.DataFrame(
            [{"Reference forecast": n,
              "RMSE": float(np.sqrt(np.mean((np.asarray(actual) - r) ** 2))),
              "MAE": float(np.mean(np.abs(np.asarray(actual) - r)))}
             for n, r in refs.items()]
        )
        with st.expander("Error of each reference forecast"):
            st.dataframe(ref_err.round(4), hide_index=True, use_container_width=True)

    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("Where the error comes from")
    st.markdown(
        '<p class="dek">The Kling Gupta score, standard in hydrology, splits performance into '
        "three things that fail independently: whether the forecast moves with the observations "
        "at all, whether its average level is right, and whether it varies as much as reality "
        "does. All three are ideally 1. A forecast can sit at the right level while barely "
        "varying, and no single error number reveals that.</p>",
        unsafe_allow_html=True,
    )
    parts = {m: inference.kling_gupta(actual, p) for m, p in preds.items()}
    st.altair_chart(charts.kge_components(parts), use_container_width=False)

    scale_series = (clean_df["PRCP"].to_numpy(dtype=float)
                    if clean_df is not None and "PRCP" in clean_df else None)
    summary_rows = []
    for m, p in preds.items():
        k = parts[m]
        summary_rows.append(
            {
                "Model": m,
                "KGE": k["kge"],
                "Willmott d": inference.willmott_d(actual, p),
                "MASE": (inference.mase(actual, p, scale_series)
                         if scale_series is not None else float("nan")),
            }
        )
    st.dataframe(pd.DataFrame(summary_rows).round(4), hide_index=True, use_container_width=True)
    st.caption(
        "KGE of 1 is perfect and below about -0.41 is worse than simply using the average of the "
        "observations. Willmott's d runs 0 to 1 and, unlike R squared, never goes negative. MASE "
        "below 1 beats a naive one step forecast on the same series."
    )


def _render_accuracy_charts(actual, preds, metrics):
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("The four headline metrics")
    st.markdown(
        '<p class="dek">Each bar starts at zero, so the height differences are the real '
        "differences. RMSE punishes large misses hardest, MAE treats every miss in proportion, "
        "MAPE expresses error relative to how much rain actually fell, and R squared compares the "
        "forecast against simply predicting the average every day.</p>",
        unsafe_allow_html=True,
    )

    for row in ([("RMSE", "rmse"), ("MAE", "mae")], [("MAPE (%)", "mape"), ("R2", "r2")]):
        cols = st.columns(2, gap="medium")
        for col, (label, key) in zip(cols, row):
            values = {m: metrics[m][key] for m in charts.MODEL_ORDER}
            if all(np.isnan(v) for v in values.values()):
                col.caption(f"{label} is not available for this file.")
                continue
            col.altair_chart(charts.metric_comparison(values, label), use_container_width=True)

    if not np.isnan(metrics["Ensemble"]["r2"]) and metrics["Ensemble"]["r2"] < 0:
        st.markdown(
            '<div class="note-box">A negative R squared means the forecast scored worse than '
            "predicting the average of this period every single day. On a strongly zero inflated "
            "series like daily rainfall that is common and expected, and it is why the charts "
            "below matter: they show what the forecast does get right.</div>",
            unsafe_allow_html=True,
        )


def _render_agreement_charts(result, actual, preds, metrics):
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("How closely the forecast follows what happened")

    st.markdown(
        '<p class="dek">Each dot is one day. A forecast that tracks the weather scatters along '
        "the dashed one to one line; a forecast that has settled near a single value forms a flat "
        "horizontal band no matter how good its error score is.</p>",
        unsafe_allow_html=True,
    )
    st.altair_chart(charts.predicted_vs_actual(actual, preds), use_container_width=True)

    st.markdown("##### Forecast spread against observed spread")
    st.markdown(
        '<p class="dek">A forecast cannot be informative about variation it never produces. The '
        "red line is how much the actual rainfall varied over your file.</p>",
        unsafe_allow_html=True,
    )
    pred_sds = {m: metrics[m]["pred_sd"] for m in charts.MODEL_ORDER}
    observed_sd = float(metrics["Ensemble"]["actual_sd"])
    st.altair_chart(
        charts.spread_comparison(pred_sds, observed_sd), use_container_width=True
    )
    ratio = pred_sds["Ensemble"] / observed_sd * 100 if observed_sd else float("nan")
    st.caption(
        f"The ensemble forecast varies by {pred_sds['Ensemble']:.3f} against an observed "
        f"{observed_sd:.3f}, about {ratio:.1f}% as much."
    )

    st.markdown("##### Day by day")
    st.altair_chart(
        charts.forecast_vs_actual_series(result.backtest_dates, actual, preds),
        use_container_width=True,
    )

    st.markdown("##### Residual distribution")
    st.markdown(
        '<p class="dek">Forecast minus observed. Mass to the left of zero means the model '
        "under-predicted; to the right, over-predicted. The axis is clipped so the bulk of the "
        "days stay readable.</p>",
        unsafe_allow_html=True,
    )
    clip = 20.0
    n_clipped = int(
        sum(np.sum(np.abs(np.asarray(p, float) - np.asarray(actual, float)) > clip)
            for p in preds.values())
    )
    st.altair_chart(charts.residual_histogram(actual, preds, clip=clip), use_container_width=True)
    if n_clipped:
        st.caption(
            f"{n_clipped} residual(s) across the three models fall outside +/-{clip:.0f} and are "
            "shown at the edge, almost always large storm days the models under-predicted."
        )

    shape = pd.DataFrame(
        [
            {
                "Model": m,
                "Bias": round(metrics[m]["bias"], 4),
                "Residual sd": round(metrics[m]["residual_std"], 4),
                "Skew": round(metrics[m]["residual_skew"], 4),
                "Kurtosis": round(metrics[m]["residual_kurtosis"], 4),
                "Pearson r": round(metrics[m]["pearson_r"], 4),
                "Explained variance": round(metrics[m]["explained_variance"], 4),
            }
            for m in charts.MODEL_ORDER
        ]
    )
    st.markdown("##### Error shape and correlation")
    st.dataframe(shape, hide_index=True, use_container_width=True)
    st.caption(
        "Bias is the average signed error, so a negative value means the forecast runs low. "
        "Strong negative skew means the misses that do happen are large under-predictions. "
        "Pearson r measures whether the forecast moves with the observations at all, "
        "independently of how far off it is."
    )


def _render_regime_charts(actual, preds):
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("Error by rainfall intensity")
    st.markdown(
        '<p class="dek">Overall error is dominated by whichever band holds the most days, '
        "usually dry ones. Splitting by how much rain actually fell shows which model is stronger "
        "in which regime.</p>",
        unsafe_allow_html=True,
    )
    tables = {name: inference.rainfall_intensity_performance(actual, p) for name, p in preds.items()}
    st.altair_chart(charts.intensity_bars(tables, value="MAE"), use_container_width=False)

    tab_lstm, tab_xgb, tab_ensemble = st.tabs(["LSTM", "XGBoost", "Ensemble"])
    for tab, name in zip((tab_lstm, tab_xgb, tab_ensemble), charts.MODEL_ORDER):
        with tab:
            st.dataframe(tables[name], hide_index=True, use_container_width=True)


def _render_event_charts(actual, preds):
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.subheader("Treating the forecast as a rain warning")
    st.markdown(
        '<p class="dek">RMSE, MAE and MAPE all score how far off the amount was. None of them '
        "says whether the forecast put the wetter days in the right order, which is what matters "
        "for deciding whether to expect rain. These scores do. AUC is the clearest: 0.5 is a coin "
        "toss, and it cannot be improved by simply shifting every forecast up or down.</p>",
        unsafe_allow_html=True,
    )
    tables = {name: inference.event_detection(actual, p) for name, p in preds.items()}
    usable = {k: v for k, v in tables.items() if v["AUC"].notna().any()}
    if not usable:
        st.markdown(
            '<div class="note-box">Your file does not contain both wet and dry days at these '
            "thresholds, so detection scores cannot be computed from it.</div>",
            unsafe_allow_html=True,
        )
        return

    c1, c2 = st.columns(2, gap="medium")
    c1.altair_chart(charts.event_detection_chart(usable, "AUC"), use_container_width=True)
    c2.altair_chart(charts.event_detection_chart(usable, "CSI"), use_container_width=True)
    st.caption(
        "POD is the share of wet days correctly flagged, FAR the share of warnings that were "
        "wrong, CSI a combined score, and HSS the improvement over chance. Full numbers below."
    )
    tab_lstm, tab_xgb, tab_ensemble = st.tabs(["LSTM", "XGBoost", "Ensemble"])
    for tab, name in zip((tab_lstm, tab_xgb, tab_ensemble), charts.MODEL_ORDER):
        with tab:
            st.dataframe(tables[name].round(4), hide_index=True, use_container_width=True)


def _render_reference_scorecard():
    """Training-time results for whichever model directory is loaded, read from
    its own metadata.json. Previously this embedded a static HTML file, which
    showed one fixed model's numbers regardless of which model was in use."""
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    with st.expander("Training time reference scorecard"):
        try:
            summary = inference.training_summary(_artifacts())
        except Exception as exc:
            st.markdown(
                f'<div class="note-box critical">Could not read the recorded training metrics: '
                f"{exc}</div>",
                unsafe_allow_html=True,
            )
            return

        scored = summary.get("n_scored_days")
        st.markdown(
            '<p class="dek">Measured on the held out test set when this model was trained, read '
            f"directly from <code>models/{inference.ACTIVE_MODEL_NAME}/metadata.json</code>"
            + (f" over {scored:,} scored days" if scored else "")
            + ". Nothing here reflects data uploaded to this site.</p>",
            unsafe_allow_html=True,
        )

        ev = summary["evaluation"]
        if ev:
            rows = []
            for key, label in (("lstm", "LSTM"), ("xgboost", "XGBoost"), ("ensemble", "Ensemble")):
                m = ev.get(key)
                if not m:
                    continue
                rows.append(
                    {
                        "Model": label,
                        "RMSE": round(float(m["rmse"]), 4),
                        "MAE": round(float(m["mae"]), 4),
                        "MAPE (%)": round(float(m["mape"]), 2),
                        "R2": round(float(m["r2"]), 4),
                        "Bias": round(float(m["bias"]), 4),
                    }
                )
            base = summary.get("baseline") or {}
            if base.get("rmse"):
                rows.append(
                    {
                        "Model": "Baseline (LSTM only)",
                        "RMSE": round(float(base["rmse"]), 4),
                        "MAE": round(float(base.get("mae", float("nan"))), 4),
                        "MAPE (%)": round(float(base.get("mape", float("nan"))), 2),
                        "R2": None,
                        "Bias": None,
                    }
                )
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

            if base.get("rmse") and "ensemble" in ev:
                gain = (float(base["rmse"]) - float(ev["ensemble"]["rmse"])) / float(base["rmse"]) * 100
                st.caption(
                    f"Ensemble RMSE is {gain:.2f}% below the reproduced single model baseline "
                    f"of {float(base['rmse']):.4f}."
                )

        st.markdown("##### How this model was configured")
        lc = summary["lstm_config"]
        ens = summary["ensemble"]
        pre = summary["preprocessing"]
        bullets = [
            f"**XGBoost:** {summary['xgb_description']}",
            f"**LSTM:** {lc.get('units_layer1','?')} + {lc.get('units_layer2','?')} units, "
            f"dropout {lc.get('dropout','?')}, {lc.get('epochs','?')} epochs, "
            f"{lc.get('window_size','?')} day lookback",
            f"**Ensemble:** {ens.get('strategy','?')} "
            f"(LSTM {float(ens.get('weight_lstm', float('nan'))):.4f}, "
            f"XGBoost {float(ens.get('weight_xgb', float('nan'))):.4f})",
            f"**Preprocessing:** Box-Cox lambda {float(pre.get('box_cox_lambda', float('nan'))):.4f}, "
            f"{pre.get('scaler_type','?')}",
        ]
        search = summary.get("search")
        if search:
            sel = search.get("selected", {})
            bullets.append(
                f"**Hyperparameters:** selected by {search.get('technique','a search')} "
                f"over {search.get('n_fits','?')} fits, scored on "
                f"{search.get('criterion','a validation holdout')} - not hand set"
            )
            bullets.append(
                "**Selected values:** "
                + ", ".join(f"{k} {sel[k]}" for k in sorted(sel))
            )
        st.markdown("\n".join(f"- {b}" for b in bullets))

        # Event detection recorded at training time, if the notebook saved it.
        ed = (artifacts_metadata := _artifacts().metadata).get("event_detection")
        if ed:
            st.markdown("##### Rain warning scores on the test set")
            tables = {
                name: pd.DataFrame(rows)
                for name, rows in ed.items()
                if name in charts.MODEL_ORDER
            }
            if tables:
                c1, c2 = st.columns(2, gap="medium")
                c1.altair_chart(charts.event_detection_chart(tables, "AUC"),
                                use_container_width=True)
                c2.altair_chart(charts.event_detection_chart(tables, "CSI"),
                                use_container_width=True)
                st.caption(
                    "Measured on the held out test set. AUC of 0.5 is chance; a model can score "
                    "poorly on RMSE while still ordering wet days better than chance, and these "
                    "scores are what reveal that."
                )

        # Error by rainfall intensity, as recorded at training time.
        rip = artifacts_metadata.get("rainfall_intensity_performance")
        if rip:
            st.markdown("##### Test-set error by rainfall intensity")
            tables = {
                label: pd.DataFrame(rip[key])
                for key, label in (("lstm", "LSTM"), ("xgboost", "XGBoost"),
                                   ("ensemble", "Ensemble"))
                if key in rip
            }
            if tables:
                st.altair_chart(charts.intensity_bars(tables, value="MAE"),
                                use_container_width=False)

        mh = summary.get("multi_horizon")
        if mh and mh.get("per_lead_results"):
            st.markdown("##### Forecast error by lead day")
            lead = pd.DataFrame(mh["per_lead_results"])
            st.altair_chart(charts.lead_day_chart(lead, "ENS RMSE"), use_container_width=True)
            st.caption(
                "Error is essentially flat from lead day 1 to lead day 7 and stays well below the "
                "persistence baseline, so a seven day figure is not much less accurate than a one "
                "day one -- but it is not more informative either. Note the train-mean constant "
                "line: on this series a single fixed number is a genuinely hard reference to beat "
                "on RMSE, which is what the negative R squared reflects. The study evaluates the "
                "one day forecast; the longer horizon is a practical extension."
            )
            keep = [c for c in ("Lead day", "XGB method", "ENS RMSE", "ENS MAE", "ENS R2",
                                "ENS pred sd", "Persistence RMSE", "Train-mean RMSE")
                    if c in lead.columns]
            st.dataframe(lead[keep].round(4), hide_index=True, use_container_width=True)

        cost = summary.get("cost") or {}
        if cost:
            st.markdown("##### Time taken to build this model")
            st.dataframe(
                pd.DataFrame(
                    [{"Stage": k.replace("_", " "), "Seconds": round(float(v), 2)}
                     for k, v in cost.items()]
                ),
                hide_index=True,
                use_container_width=True,
            )
