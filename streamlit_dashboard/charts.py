"""Altair chart builders for the Performance page.

Every chart here is a pure function: it takes already-computed numbers and
returns a chart. Nothing in this module runs a model, reads a file, or
recomputes a metric -- the callers pass in either metrics measured live from an
uploaded CSV, or values read straight out of a saved metadata.json.

Two conventions worth knowing when reading the charts:

  * Model colours come from theme.PALETTE, so LSTM, XGBoost and Ensemble keep
    the same colour everywhere on the site.
  * Where a chart can mislead, it carries its own reference line rather than
    relying on the caption -- the observed spread on the spread chart, the 1:1
    line on the scatter, chance level on AUC. A zoomed axis without a reference
    is how a 3% difference ends up looking like a collapse.
"""

from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

from theme import PALETTE

MODEL_ORDER = ["LSTM", "XGBoost", "Ensemble"]
MODEL_COLOURS = [PALETTE["lstm"], PALETTE["xgb"], PALETTE["ensemble"]]
_MODEL_SCALE = alt.Scale(domain=MODEL_ORDER, range=MODEL_COLOURS)

# Metrics where a bigger number is worse. Used to label axes honestly rather
# than leaving the reader to guess which direction is good.
LOWER_IS_BETTER = {"RMSE", "MAE", "MAPE (%)", "Bias", "FAR"}


def metric_comparison(metrics_by_model: dict, metric: str) -> alt.Chart:
    """One metric across the three models, as bars from a zero baseline.

    Bars start at zero deliberately. A zoomed axis makes a few percent look
    like a landslide, which is exactly the trap the notebook's own lead-day
    chart falls into.
    """
    rows = [{"Model": m, "value": float(v)} for m, v in metrics_by_model.items()]
    df = pd.DataFrame(rows)
    direction = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=MODEL_ORDER, title=None),
            y=alt.Y("value:Q", title=f"{metric} ({direction})", scale=alt.Scale(zero=True)),
            color=alt.Color("Model:N", scale=_MODEL_SCALE, legend=None),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("value:Q", format=".4f", title=metric)],
        )
        .properties(height=200, title=metric)
    )


def predicted_vs_actual(actual, preds_by_model: dict) -> alt.Chart:
    """Scatter of forecast against observed, with the 1:1 line.

    This is the single most revealing chart on the page. A forecast that tracks
    the weather scatters along the diagonal; one that has settled near a
    constant forms a horizontal band, no matter how good its RMSE looks.
    """
    frames = []
    for name, p in preds_by_model.items():
        frames.append(pd.DataFrame({"Actual": np.asarray(actual, float),
                                    "Predicted": np.asarray(p, float),
                                    "Model": name}))
    df = pd.concat(frames, ignore_index=True)
    hi = float(max(df["Actual"].max(), df["Predicted"].max()))

    points = (
        alt.Chart(df)
        .mark_circle(size=34, opacity=0.45)
        .encode(
            x=alt.X("Actual:Q", title="Observed rainfall", scale=alt.Scale(domain=[0, hi])),
            y=alt.Y("Predicted:Q", title="Forecast rainfall", scale=alt.Scale(domain=[0, hi])),
            color=alt.Color("Model:N", scale=_MODEL_SCALE,
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Model:N"),
                     alt.Tooltip("Actual:Q", format=".2f"),
                     alt.Tooltip("Predicted:Q", format=".2f")],
        )
    )
    diagonal = (
        alt.Chart(pd.DataFrame({"x": [0, hi], "y": [0, hi]}))
        .mark_line(strokeDash=[6, 4], color="#8a8a8a", size=1)
        .encode(x="x:Q", y="y:Q")
    )
    return (points + diagonal).properties(height=340)


def residual_histogram(actual, preds_by_model: dict, clip: float = 20.0) -> alt.Chart:
    """Residual (forecast minus observed) distributions, one panel per model.

    The x-axis is clipped because the raw range is dominated by a handful of
    very large misses on storm days: without clipping, every panel is empty
    space around a single spike and the informative middle is invisible. The
    count of clipped points is reported by the caller.
    """
    frames = []
    for name, p in preds_by_model.items():
        resid = np.asarray(p, float) - np.asarray(actual, float)
        frames.append(pd.DataFrame({"Residual": np.clip(resid, -clip, clip), "Model": name}))
    df = pd.concat(frames, ignore_index=True)

    # The zero line uses a datum rather than its own dataframe: a faceted
    # layer in Altair requires all layers to share the facet's data source.
    bars = alt.Chart().mark_bar(opacity=0.85).encode(
        x=alt.X("Residual:Q", bin=alt.Bin(maxbins=45),
                title=f"Forecast minus observed (clipped to +/-{clip:.0f})"),
        y=alt.Y("count():Q", title="days"),
        color=alt.Color("Model:N", scale=_MODEL_SCALE, legend=None),
    )
    zero = alt.Chart().mark_rule(color="#0b0b0b", size=1).encode(x=alt.datum(0.0))
    return (
        alt.layer(bars, zero)
        .properties(height=190, width=230)
        .facet(column=alt.Column("Model:N", sort=MODEL_ORDER, title=None), data=df)
    )


def spread_comparison(pred_sds: dict, observed_sd: float) -> alt.Chart:
    """Forecast spread against the spread of the observations.

    A forecast cannot be informative about variation it never produces, so this
    puts each model's standard deviation next to the target's. The gap is
    usually the most important single fact about a flat forecast.
    """
    df = pd.DataFrame([{"Model": m, "sd": float(v)} for m, v in pred_sds.items()])
    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=MODEL_ORDER, title=None),
            y=alt.Y("sd:Q", title="standard deviation", scale=alt.Scale(zero=True)),
            color=alt.Color("Model:N", scale=_MODEL_SCALE, legend=None),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("sd:Q", format=".4f")],
        )
    )
    rule = (
        alt.Chart(pd.DataFrame({"y": [observed_sd]}))
        .mark_rule(color=PALETTE["critical"], strokeDash=[6, 4], size=2)
        .encode(y="y:Q", tooltip=alt.Tooltip("y:Q", format=".4f", title="observed sd"))
    )
    return (bars + rule).properties(height=220)


def intensity_bars(tables_by_model: dict, value: str = "MAE") -> alt.Chart:
    """Error per rainfall-intensity band, grouped by model.

    Overall error is dominated by whichever band has the most days -- usually
    dry ones. Splitting by band shows which model owns which regime, and that
    is where the two components differ most.
    """
    frames = []
    for name, tbl in tables_by_model.items():
        t = tbl.copy()
        t["Model"] = name
        frames.append(t[["Intensity", "N", value, "Model"]])
    df = pd.concat(frames, ignore_index=True)
    df = df[df["N"] > 0]
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y(f"{value}:Q", title=f"{value} (lower is better)", scale=alt.Scale(zero=True)),
            color=alt.Color("Model:N", scale=_MODEL_SCALE, legend=None),
            column=alt.Column("Intensity:N", title=None,
                              sort=["No Rain (0 in)", "Light (0-0.2 in)",
                                    "Moderate (0.2-0.5 in)", "Heavy (>0.5 in)"]),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Intensity:N"),
                     alt.Tooltip("N:Q", title="days"), alt.Tooltip(f"{value}:Q", format=".3f")],
        )
        .properties(height=200, width=110)
    )


def event_detection_chart(tables_by_model: dict, metric: str = "AUC") -> alt.Chart:
    """A detection score against rainfall threshold, one line per model.

    For AUC a chance line is drawn at 0.5: a forecast that orders days no
    better than a coin toss sits on it, however good its RMSE is.
    """
    frames = []
    for name, tbl in tables_by_model.items():
        t = tbl.copy()
        t["Model"] = name
        frames.append(t[["Threshold (in)", metric, "N events", "Model"]])
    df = pd.concat(frames, ignore_index=True).dropna(subset=[metric])

    line = (
        alt.Chart(df)
        .mark_line(point=True, size=2)
        .encode(
            x=alt.X("Threshold (in):O", title="rainfall threshold"),
            y=alt.Y(f"{metric}:Q", title=metric, scale=alt.Scale(zero=False)),
            color=alt.Color("Model:N", scale=_MODEL_SCALE,
                            legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Threshold (in):O"),
                     alt.Tooltip(f"{metric}:Q", format=".4f"),
                     alt.Tooltip("N events:Q", title="events")],
        )
    )
    if metric == "AUC":
        chance = (
            alt.Chart(pd.DataFrame({"y": [0.5]}))
            .mark_rule(color=PALETTE["critical"], strokeDash=[6, 4], size=1.5)
            .encode(y="y:Q", tooltip=alt.Tooltip("y:Q", title="chance"))
        )
        return (line + chance).properties(height=280)
    return line.properties(height=280)


def lead_day_chart(per_lead: pd.DataFrame, value: str = "ENS RMSE") -> alt.Chart:
    """Error against forecast lead day, with the reference forecasts drawn in.

    The y-axis includes zero. The notebook's version of this chart zooms to the
    data range, which turns a 4% spread in MAE into what looks like a collapse;
    including zero keeps the real size of the differences visible. Persistence
    and the train-mean constant are plotted alongside, because the useful
    question is not whether error grows with lead day but whether the model
    beats a forecast that needed no model at all.
    """
    df = per_lead.copy()
    keep = ["Lead day", "XGB method", value]
    extras = [c for c in ("Persistence RMSE", "Train-mean RMSE") if c in df.columns]
    melted = df[keep].rename(columns={value: "value"})
    melted["Series"] = "ensemble (" + melted["XGB method"].astype(str) + ")"
    frames = [melted[["Lead day", "Series", "value"]]]

    if value == "ENS RMSE":
        for col, label in [("Persistence RMSE", "persistence"),
                           ("Train-mean RMSE", "train-mean constant")]:
            if col in extras:
                ref = df[df["XGB method"] == df["XGB method"].iloc[0]][["Lead day", col]]
                ref = ref.rename(columns={col: "value"})
                ref["Series"] = label
                frames.append(ref[["Lead day", "Series", "value"]])

    plot = pd.concat(frames, ignore_index=True)
    return (
        alt.Chart(plot)
        .mark_line(point=True, size=2)
        .encode(
            x=alt.X("Lead day:O", title="lead day"),
            y=alt.Y("value:Q", title=value, scale=alt.Scale(zero=True)),
            color=alt.Color("Series:N", legend=alt.Legend(orient="top", title=None)),
            strokeDash=alt.StrokeDash("Series:N", legend=None),
            tooltip=[alt.Tooltip("Series:N"), alt.Tooltip("Lead day:O"),
                     alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=300)
    )


def kge_components(parts_by_model: dict) -> alt.Chart:
    """The three parts of the Kling-Gupta score, against their ideal of 1.

    Plotted together because they fail independently: a forecast can sit at the
    right average level (beta near 1) while barely varying at all (gamma near
    0), and a single error number cannot tell those apart.
    """
    labels = {"r": "r (moves together)", "beta": "beta (level)", "gamma": "gamma (variability)"}
    rows = []
    for model, parts in parts_by_model.items():
        for key, label in labels.items():
            rows.append({"Model": model, "Component": label, "value": float(parts[key])})
    df = pd.DataFrame(rows).dropna()

    bars = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("value:Q", title="value (1.0 is ideal)", scale=alt.Scale(zero=True)),
            color=alt.Color("Model:N", scale=_MODEL_SCALE, legend=None),
            column=alt.Column("Component:N", title=None,
                              sort=list(labels.values())),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Component:N"),
                     alt.Tooltip("value:Q", format=".4f")],
        )
        .properties(height=220, width=150)
    )
    return bars


def skill_score_chart(skill_df: pd.DataFrame) -> alt.Chart:
    """Skill against each reference forecast, with zero marked.

    Zero is the line that matters: it means the model did exactly as well as a
    forecast that needed no model. Above is useful, below is worse than doing
    nothing.
    """
    melted = skill_df.melt(id_vars="Model", var_name="Reference", value_name="skill").dropna()
    bars = (
        alt.Chart(melted)
        .mark_bar()
        .encode(
            x=alt.X("Model:N", sort=MODEL_ORDER, title=None, axis=alt.Axis(labelAngle=0)),
            y=alt.Y("skill:Q", title="MSE skill (0 = no better than reference)",
                    scale=alt.Scale(zero=True)),
            color=alt.Color("Model:N", scale=_MODEL_SCALE, legend=None),
            column=alt.Column("Reference:N", title=None),
            tooltip=[alt.Tooltip("Model:N"), alt.Tooltip("Reference:N"),
                     alt.Tooltip("skill:Q", format=".4f")],
        )
        .properties(height=220, width=135)
    )
    return bars


def forecast_vs_actual_series(dates, actual, preds_by_model: dict) -> alt.Chart:
    """The forecast laid over the observed series, day by day."""
    frames = [pd.DataFrame({"Date": pd.to_datetime(dates),
                            "value": np.asarray(actual, float), "Series": "Observed"})]
    for name, p in preds_by_model.items():
        frames.append(pd.DataFrame({"Date": pd.to_datetime(dates),
                                    "value": np.asarray(p, float), "Series": name}))
    df = pd.concat(frames, ignore_index=True)
    scale = alt.Scale(domain=["Observed"] + MODEL_ORDER,
                      range=["#8a8a8a"] + MODEL_COLOURS)
    return (
        alt.Chart(df)
        .mark_line(size=1.6)
        .encode(
            x=alt.X("Date:T", title=None),
            y=alt.Y("value:Q", title="rainfall"),
            color=alt.Color("Series:N", scale=scale, legend=alt.Legend(orient="top", title=None)),
            tooltip=[alt.Tooltip("Date:T"), alt.Tooltip("Series:N"),
                     alt.Tooltip("value:Q", format=".2f")],
        )
        .properties(height=300)
    )
