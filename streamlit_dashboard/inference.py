"""
Inference-only helper module for the Streamlit website.

This module loads the already trained v2_static ensemble artifacts (LSTM
weights, XGBoost model, fitted Box-Cox lambda and MinMax scaler, locked
ensemble weights) and runs the exact same preprocessing and prediction
steps the training notebook (LSTM_XGBoost_Ensemble_Precipitation_STATIC.ipynb)
uses at inference time, in the same order:

    1. Box-Cox transform on (precipitation + 1) using the SAVED lambda
       (never re-estimated from uploaded data).
    2. Min-Max scaling of the Box-Cox values using the SAVED scaler
       (transform only, never re-fit) -- this scaled series feeds the LSTM.
    3. The unscaled Box-Cox values feed XGBoost directly (XGBoost is
       scale-invariant, matching the notebook's design).
    4. Both models consume a sliding window of WINDOW_SIZE = 70 preceding
       days per prediction, exactly as split_sequences() does in the
       notebook.
    5. The two model outputs are combined with the locked inverse-RMSE
       ensemble weights saved in metadata.json.
    6. Predictions are inverse Box-Cox transformed back to inches.

Nothing in this module fits, trains, or tunes anything. It only loads
saved artifacts and applies saved parameters to new data.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import boxcox as boxcox_apply
from scipy.special import inv_boxcox
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    explained_variance_score,
    roc_auc_score,
)

# Which trained-model folder under models/ the site loads and visualises.
# Defaults to the locked "v2_static" pipeline. Set the PRECIP_MODEL_DIR
# environment variable before launching Streamlit to point the whole site at
# a different run, e.g. the diagnostic copy:
#     PRECIP_MODEL_DIR=v2_static_xgb_experiment streamlit run app.py
# The value is a folder name directly under models/ (not a full path).
ACTIVE_MODEL_NAME = os.environ.get("PRECIP_MODEL_DIR", "v2_prefinal").strip() or "v2_prefinal"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / ACTIVE_MODEL_NAME
REQUIRED_ARTIFACTS = {
    "metadata": MODEL_DIR / "metadata.json",
    "lstm_model": MODEL_DIR / "lstm_model.keras",
    "xgboost_model": MODEL_DIR / "xgboost_model.json",
    "scaler": MODEL_DIR / "scaler.joblib",
}

# Multi-day forecasting is OPTIONAL. A model directory that has one XGBoost
# model per lead day (trained directly on y[t+h], never fed its own output
# back) can produce a seven-day outlook; one that does not simply shows the
# one-day forecast. v2_static has no horizon models, so it keeps working.
FORECAST_HORIZON = 7
HORIZON_ARTIFACTS = {
    h: MODEL_DIR / f"xgboost_model_h{h}.json" for h in range(1, FORECAST_HORIZON + 1)
}

RAIN_INTENSITY_BINS = [-np.inf, 1e-9, 0.2, 0.5, np.inf]
# Label text matches the training notebook's RAIN_INTENSITY_LABELS exactly
# (not just the numeric bins), since models/v2_static/metadata.json stores
# its rainfall_intensity_performance rows keyed on these exact strings --
# the live performance page needs to line the uploaded-data numbers up
# against the baked-in reference numbers apples to apples.
RAIN_INTENSITY_LABELS = ["No Rain (0 in)", "Light (0-0.2 in)", "Moderate (0.2-0.5 in)", "Heavy (>0.5 in)"]


class ArtifactsMissingError(Exception):
    """Raised when one or more saved model artifacts cannot be found on disk."""


class UploadValidationError(Exception):
    """Raised when an uploaded file does not match the format the model expects."""


def missing_artifacts() -> list[str]:
    """Return a list of human readable descriptions of any missing artifact files."""
    missing = []
    for name, path in REQUIRED_ARTIFACTS.items():
        if not path.exists():
            missing.append(f"{name} (expected at {path})")
    return missing


@dataclass
class Artifacts:
    metadata: dict
    lstm_model: object
    xgb_model: object
    scaler: object
    window_size: int
    box_cox_lambda: float
    weight_lstm: float
    weight_xgb: float
    # Optional multi-day support. Empty when the model directory has no
    # per-horizon models, in which case the site shows the one-day forecast only.
    horizon_models: dict = field(default_factory=dict)
    horizon_weights: dict = field(default_factory=dict)

    @property
    def supports_multi_day(self) -> bool:
        return len(self.horizon_models) > 0


def load_artifacts() -> Artifacts:
    """Load the saved LSTM model, XGBoost model, scaler, and metadata from disk.

    Raises ArtifactsMissingError if any required file is not present. Never
    trains or fits anything; every statistical parameter used below (the
    Box-Cox lambda, the scaler's fitted range, the ensemble weights) comes
    straight out of the saved artifacts.
    """
    missing = missing_artifacts()
    if missing:
        raise ArtifactsMissingError(
            "The saved model files this page needs are not present. Missing: "
            + "; ".join(missing)
            + ". Run the training notebook (LSTM_XGBoost_Ensemble_Precipitation_STATIC.ipynb) "
            "to produce the models/v2_static artifacts, then reload this page."
        )

    import json

    import joblib
    import xgboost as xgb
    import tensorflow as tf

    metadata = json.loads(REQUIRED_ARTIFACTS["metadata"].read_text(encoding="utf-8"))
    lstm_model = tf.keras.models.load_model(REQUIRED_ARTIFACTS["lstm_model"])
    xgb_model = xgb.XGBRegressor()
    xgb_model.load_model(str(REQUIRED_ARTIFACTS["xgboost_model"]))
    scaler = joblib.load(REQUIRED_ARTIFACTS["scaler"])

    # Optional per-horizon models. Loaded only if every lead day is present;
    # a partial set is treated as absent rather than silently forecasting a
    # shorter horizon than the page advertises.
    horizon_models: dict = {}
    horizon_weights: dict = {}
    if all(p.exists() for p in HORIZON_ARTIFACTS.values()):
        for h, path in HORIZON_ARTIFACTS.items():
            m = xgb.XGBRegressor()
            m.load_model(str(path))
            horizon_models[h] = m
        # Weights derived per lead day at training time by the same
        # inverse-RMSE rule as the one-day forecast. Falls back to the one-day
        # weights for any lead the metadata does not carry.
        per_lead = metadata.get("ensemble", {}).get("per_lead_weights", {}).get("direct", [])
        for row in per_lead:
            horizon_weights[int(row["lead"])] = (float(row["w_lstm"]), float(row["w_xgb"]))
        for h in horizon_models:
            horizon_weights.setdefault(
                h,
                (float(metadata["ensemble"]["weight_lstm"]),
                 float(metadata["ensemble"]["weight_xgb"])),
            )

    return Artifacts(
        metadata=metadata,
        lstm_model=lstm_model,
        xgb_model=xgb_model,
        scaler=scaler,
        window_size=int(metadata["lstm_config"]["window_size"]),
        box_cox_lambda=float(metadata["preprocessing"]["box_cox_lambda"]),
        weight_lstm=float(metadata["ensemble"]["weight_lstm"]),
        weight_xgb=float(metadata["ensemble"]["weight_xgb"]),
        horizon_models=horizon_models,
        horizon_weights=horizon_weights,
    )


def boxcox1p_transform(data: np.ndarray, lam: float) -> np.ndarray:
    """Apply a Box-Cox transform with a FIXED, already fitted lambda to
    (data + 1), matching boxcox1p() at training time. This never estimates
    a new lambda from the data passed in."""
    return boxcox_apply(np.asarray(data, dtype=float) + 1.0, lam)


def inv_boxcox1p(bcdata: np.ndarray, lam: float) -> np.ndarray:
    """Inverse Box-Cox, clipped just under the transform's asymptote.

    For a negative lambda the inverse is only defined below -1/lambda (here
    about 0.695); a prediction at or above that returns NaN. The earlier
    pseudo-Huber model never reached it, but the current quantile model has a
    far wider predictive spread and can, so predictions are clipped rather
    than allowed to produce a silent NaN on the page.
    """
    bc = np.asarray(bcdata, dtype=float)
    if lam < 0:
        bc = np.minimum(bc, (-1.0 / lam) - 1e-9)
    return inv_boxcox(bc, lam) - 1.0


def split_sequences(series: np.ndarray, window: int):
    """Identical sliding window logic to the notebook: x[i] = series[i:i+window],
    y[i] = series[i+window]."""
    x, y = [], []
    for i in range(len(series)):
        end = i + window
        if end > len(series) - 1:
            break
        x.append(series[i:end])
        y.append(series[end])
    return np.array(x), np.array(y)


DATE_COLUMN_ALIASES = {"date", "dates", "day"}
PRCP_COLUMN_ALIASES = {"prcp", "precip", "precipitation", "rain", "rainfall"}


def validate_and_prepare(df: pd.DataFrame, window_size: int) -> pd.DataFrame:
    """Check that an uploaded dataframe matches the shape the model expects
    and return a cleaned dataframe with columns DATE (datetime) and PRCP
    (float), sorted chronologically with no duplicate dates.

    Raises UploadValidationError with a plain description of the problem
    and what to do about it.
    """
    if df.empty:
        raise UploadValidationError("The file has no rows. Upload a CSV with at least one row of data.")

    columns_lower = {c.lower().strip(): c for c in df.columns}

    date_col = None
    for alias in DATE_COLUMN_ALIASES:
        if alias in columns_lower:
            date_col = columns_lower[alias]
            break

    prcp_col = None
    for alias in PRCP_COLUMN_ALIASES:
        if alias in columns_lower:
            prcp_col = columns_lower[alias]
            break

    if date_col is None or prcp_col is None:
        raise UploadValidationError(
            "The file needs a date column and a precipitation column. Found columns: "
            + ", ".join(str(c) for c in df.columns)
            + ". Rename the date column to DATE and the precipitation column (in inches) to PRCP, then upload again."
        )

    clean = df[[date_col, prcp_col]].copy()
    clean.columns = ["DATE", "PRCP"]

    try:
        clean["DATE"] = pd.to_datetime(clean["DATE"], dayfirst=True, errors="raise")
    except Exception as exc:
        raise UploadValidationError(
            "The date column could not be read as dates. Use a consistent format such as "
            "DD/MM/YYYY or YYYY-MM-DD for every row, then upload again."
        ) from exc

    clean["PRCP"] = pd.to_numeric(clean["PRCP"], errors="coerce")
    if clean["PRCP"].isna().any():
        bad_rows = int(clean["PRCP"].isna().sum())
        raise UploadValidationError(
            f"The precipitation column has {bad_rows} value(s) that are not numbers. "
            "Every row needs a numeric precipitation amount in inches (use 0 for a dry day), "
            "then upload again."
        )

    if (clean["PRCP"] < 0).any():
        raise UploadValidationError(
            "The precipitation column has negative values. Precipitation cannot be less than "
            "zero. Correct those rows and upload again."
        )

    clean = clean.dropna(subset=["DATE"]).sort_values("DATE")
    duplicate_count = int(clean["DATE"].duplicated().sum())
    if duplicate_count:
        raise UploadValidationError(
            f"The date column has {duplicate_count} repeated date(s). Each row needs one unique "
            "day, with no gaps filled by more than one entry. Remove the duplicates and upload again."
        )

    clean = clean.reset_index(drop=True)

    if len(clean) < window_size:
        raise UploadValidationError(
            f"The file has {len(clean)} day(s) of data. The model needs at least {window_size} "
            "consecutive days of daily precipitation to build one prediction, because each "
            f"forecast looks back over the prior {window_size} days. Add more rows and upload again."
        )

    return clean


def gap_summary(df: pd.DataFrame) -> Optional[str]:
    """Return a plain description of any gaps in the daily date sequence, or
    None if the series is perfectly contiguous. This is informational only;
    it does not block a prediction, because the model has no other way to
    know a date was skipped."""
    deltas = df["DATE"].diff().dropna().dt.days
    gaps = int((deltas > 1).sum())
    if gaps == 0:
        return None
    largest = int(deltas.max())
    return (
        f"The date column has {gaps} gap(s) where more than one day passes between rows "
        f"(the largest gap skips {largest} days). The model treats every row as one consecutive "
        "day regardless of the calendar date, so predictions spanning a gap should be read with "
        "that in mind."
    )


def safe_mape(y_true, y_pred, min_actual=0.1):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true >= min_actual
    if mask.sum() == 0:
        return float("nan"), 0
    pct_errors = np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])
    return float(np.mean(pct_errors) * 100.0), int(mask.sum())


def evaluate_all(y_true, y_pred) -> dict:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    mape, mape_n = safe_mape(y_true, y_pred)
    r2 = r2_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    corr = float(pearsonr(y_true, y_pred)[0]) if len(y_true) > 1 else float("nan")
    expl_var = explained_variance_score(y_true, y_pred) if len(y_true) > 1 else float("nan")
    resid = y_pred - y_true
    bias = float(np.mean(resid))
    return {
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "mape_n": mape_n,
        "r2": r2,
        "pearson_r": corr,
        "explained_variance": float(expl_var),
        "bias": bias,
        "residual_mean": bias,
        "residual_std": float(np.std(resid)) if len(resid) else float("nan"),
        "residual_skew": float(skew(resid)) if len(resid) > 2 else float("nan"),
        "residual_kurtosis": float(kurtosis(resid)) if len(resid) > 2 else float("nan"),
        # How much the forecast actually moves. Compared against the spread of
        # the observations, this is what separates a forecast that tracks the
        # weather from one that has settled near a constant -- something the
        # error metrics on their own cannot show.
        "pred_sd": float(np.std(y_pred)) if len(y_pred) else float("nan"),
        "actual_sd": float(np.std(y_true)) if len(y_true) else float("nan"),
    }


def _safe_div(num, den):
    return float(num) / float(den) if den not in (0, 0.0) and np.isfinite(den) else float("nan")


def skill_score(y_true, y_pred, y_ref, metric: str = "mse") -> float:
    """1 - error(model) / error(reference). Zero means no better than the
    reference; negative means worse. Definition taken from the project's
    retuned evaluation notebook so the numbers mean the same thing here."""
    yt = np.asarray(y_true, dtype=float)
    if metric == "mae":
        m = np.mean(np.abs(yt - np.asarray(y_pred, dtype=float)))
        r = np.mean(np.abs(yt - np.asarray(y_ref, dtype=float)))
    else:
        m = np.mean((yt - np.asarray(y_pred, dtype=float)) ** 2)
        r = np.mean((yt - np.asarray(y_ref, dtype=float)) ** 2)
        if metric == "rmse":
            m, r = math.sqrt(m), math.sqrt(r)
    return _safe_div(r - m, r)


def mase(y_true, y_pred, scale_series, m: int = 1) -> float:
    """Mean absolute scaled error: MAE divided by the MAE of a one-step naive
    forecast on the supplied series. Below 1 beats the naive forecast."""
    s = np.asarray(scale_series, dtype=float)
    if len(s) <= m:
        return float("nan")
    q = np.mean(np.abs(s[m:] - s[:-m]))
    return _safe_div(
        np.mean(np.abs(np.asarray(y_true, dtype=float) - np.asarray(y_pred, dtype=float))), q
    )


def kling_gupta(y_true, y_pred) -> dict:
    """Kling-Gupta efficiency and the three parts it is built from.

    KGE is standard in hydrology because it separates three distinct ways a
    forecast can be wrong, which a single error number hides:
      r     -- does the forecast move with the observations at all
      beta  -- is its average level right (1.0 is unbiased)
      gamma -- does it vary as much as the observations (1.0 matches)
    A forecast that is nearly constant scores a gamma near zero however good
    its RMSE looks, which is precisely the failure mode worth surfacing here.
    KGE of 1 is perfect; below about -0.41 is worse than the mean of the
    observations.
    """
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    if len(yt) < 2:
        return {"kge": float("nan"), "r": float("nan"), "beta": float("nan"), "gamma": float("nan")}
    mu_t, mu_p = float(np.mean(yt)), float(np.mean(yp))
    sd_t, sd_p = float(np.std(yt)), float(np.std(yp))
    r = float(pearsonr(yt, yp)[0]) if sd_t > 0 and sd_p > 0 else float("nan")
    beta = _safe_div(mu_p, mu_t)
    gamma = _safe_div(sd_p / mu_p if mu_p else float("nan"), sd_t / mu_t if mu_t else float("nan"))
    if not all(np.isfinite([r, beta, gamma])):
        return {"kge": float("nan"), "r": r, "beta": beta, "gamma": gamma}
    kge = 1.0 - math.sqrt((r - 1) ** 2 + (beta - 1) ** 2 + (gamma - 1) ** 2)
    return {"kge": kge, "r": r, "beta": beta, "gamma": gamma}


def willmott_d(y_true, y_pred) -> float:
    """Willmott's index of agreement, bounded 0 to 1. Unlike R squared it does
    not go negative, so it stays readable when a forecast is worse than the
    mean."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    mu = float(np.mean(yt))
    denom = float(np.sum((np.abs(yp - mu) + np.abs(yt - mu)) ** 2))
    return 1.0 - _safe_div(float(np.sum((yt - yp) ** 2)), denom)


def reference_forecasts(clean_df: pd.DataFrame, window: int, train_mean: float = None) -> dict:
    """The forecasts a model has to beat to be worth anything, aligned to the
    same days the backtest scores.

    persistence  -- tomorrow equals today, the standard naive weather forecast
    climatology  -- the average of the history preceding the scored period
    zero         -- it never rains, which on a mostly dry record is a strong
                    baseline for MAE and the one people forget to check
    """
    prcp = clean_df["PRCP"].to_numpy(dtype=float)
    n = len(prcp) - window
    if n <= 0:
        return {}
    idx = np.arange(window, window + n)
    refs = {
        "Persistence (yesterday)": prcp[idx - 1],
        "Climatology (history mean)": np.full(n, float(np.mean(prcp[:window]))),
        "Always zero": np.zeros(n),
    }
    if train_mean is not None and np.isfinite(train_mean):
        refs["Training mean"] = np.full(n, float(train_mean))
    return refs


def ensemble_gain(metrics_by_model: dict) -> pd.DataFrame:
    """Does combining actually beat using either component on its own?

    This is the question an ensemble has to answer, and no single error number
    answers it. A convex blend of two forecasts is mathematically bound to land
    between them on any metric, so the honest comparison is against the better
    component, metric by metric -- not against the worse one, and not against a
    baseline.
    """
    rows = []
    for label, key, lower_better in (
        ("RMSE", "rmse", True),
        ("MAE", "mae", True),
        ("MAPE (%)", "mape", True),
        ("R2", "r2", False),
        ("Forecast spread", "pred_sd", None),
    ):
        lstm, xgb_, ens = (metrics_by_model[m][key] for m in ("LSTM", "XGBoost", "Ensemble"))
        if lower_better is None:
            rows.append({"Metric": label, "LSTM": lstm, "XGBoost": xgb_, "Ensemble": ens,
                         "Best component": float("nan"), "Ensemble beats it": ""})
            continue
        best = min(lstm, xgb_) if lower_better else max(lstm, xgb_)
        beats = (ens < best) if lower_better else (ens > best)
        rows.append({
            "Metric": label, "LSTM": lstm, "XGBoost": xgb_, "Ensemble": ens,
            "Best component": best, "Ensemble beats it": "yes" if beats else "no",
        })
    return pd.DataFrame(rows)


EVENT_THRESHOLDS = [0.0, 0.5, 2.0, 5.0]


def event_detection(y_true, y_pred, thresholds=EVENT_THRESHOLDS) -> pd.DataFrame:
    """Score the forecast as a warning system rather than as a number.

    RMSE, MAE and MAPE all measure how far off the magnitude is. They are blind
    to whether the forecast ORDERS days correctly, which is what matters for
    deciding "will tomorrow be wet". POD, FAR, CSI and HSS use the forecast's own
    crossing of each threshold; AUC is threshold free and is unchanged by any
    rescaling of the forecast, so it cannot be inflated by shifting its level.

    Ported from the training notebook so the same definitions are used here.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    rows = []
    for thr in thresholds:
        obs = y_true > thr
        fc = y_pred > thr
        a = int(np.sum(obs & fc))      # hits
        b = int(np.sum(~obs & fc))     # false alarms
        c = int(np.sum(obs & ~fc))     # misses
        d = int(np.sum(~obs & ~fc))    # correct negatives
        pod = a / (a + c) if (a + c) else float("nan")
        far = b / (a + b) if (a + b) else float("nan")
        csi = a / (a + b + c) if (a + b + c) else float("nan")
        denom = (a + c) * (c + d) + (a + b) * (b + d)
        hss = (2.0 * (a * d - b * c) / denom) if denom else float("nan")
        try:
            auc = (
                float(roc_auc_score(obs.astype(int), y_pred))
                if obs.any() and (~obs).any()
                else float("nan")
            )
        except ValueError:
            auc = float("nan")
        rows.append(
            {
                "Threshold (in)": thr,
                "N events": int(obs.sum()),
                "hits": a,
                "false alarms": b,
                "misses": c,
                "POD": pod,
                "FAR": far,
                "CSI": csi,
                "HSS": hss,
                "AUC": auc,
            }
        )
    return pd.DataFrame(rows)


def rainfall_intensity_performance(y_true, y_pred) -> pd.DataFrame:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    buckets = np.asarray(pd.cut(y_true, bins=RAIN_INTENSITY_BINS, labels=RAIN_INTENSITY_LABELS))
    rows = []
    for label in RAIN_INTENSITY_LABELS:
        mask = buckets == label
        n = int(mask.sum())
        if n == 0:
            rows.append({"Intensity": label, "N": 0, "RMSE": float("nan"), "MAE": float("nan"), "MAPE (%)": float("nan")})
            continue
        rmse_b = math.sqrt(mean_squared_error(y_true[mask], y_pred[mask]))
        mae_b = mean_absolute_error(y_true[mask], y_pred[mask])
        if label == "No Rain (0 in)":
            mape_b = float("nan")
        else:
            mape_b, _ = safe_mape(y_true[mask], y_pred[mask], min_actual=1e-9)
        rows.append({"Intensity": label, "N": n, "RMSE": rmse_b, "MAE": mae_b, "MAPE (%)": mape_b})
    return pd.DataFrame(rows)


@dataclass
class InferenceResult:
    backtest_dates: np.ndarray
    backtest_actual: Optional[np.ndarray]
    backtest_lstm: np.ndarray
    backtest_xgb: np.ndarray
    backtest_ensemble: np.ndarray
    forecast_date: pd.Timestamp
    forecast_lstm: float
    forecast_xgb: float
    forecast_ensemble: float
    # Seven-day outlook. None when the loaded model has no per-horizon models.
    # Columns: Lead day, Date, LSTM, XGBoost, Ensemble.
    forecast_7day: Optional[pd.DataFrame] = None


def training_summary(artifacts: Artifacts) -> dict:
    """Normalise the recorded training-time information out of metadata.json.

    Model directories written by different notebooks use slightly different
    keys -- the static pipeline stores its test metrics under "evaluation",
    the prefinal one under "evaluation_1day" and adds the hyperparameter
    search and the multi-day results. Everything returned here is read from
    the saved file; nothing is recomputed or estimated.
    """
    md = artifacts.metadata
    ev = md.get("evaluation_1day") or md.get("evaluation") or {}

    out = {
        "model_type": md.get("model_type", "unknown"),
        "evaluation": {k: ev[k] for k in ("lstm", "xgboost", "ensemble") if k in ev},
        "n_scored_days": ev.get("n_scored_days"),
        "baseline": md.get("baseline_reference", {}),
        "lstm_config": md.get("lstm_config", {}),
        "xgboost_config": md.get("xgboost_config", {}),
        "preprocessing": md.get("preprocessing", {}),
        "ensemble": md.get("ensemble", {}),
        "search": md.get("hyperparameter_search"),
        "multi_horizon": md.get("multi_horizon"),
        "cost": md.get("computational_cost", {}),
    }

    # A readable one-line description of the XGBoost side, which differs in
    # shape between the two notebooks.
    xc = out["xgboost_config"]
    if "fit_strategy" in xc:
        booster = xc["fit_strategy"].get("booster", "?")
        obj = xc.get("objective", {}).get("objective", "?")
        out["xgb_description"] = f"{booster} / {obj}"
    else:
        out["xgb_description"] = f"gbtree / {'reg:pseudohubererror' if 'huber_slope' in xc else '?'}"

    return out


def multi_day_forecast(
    lstm_series: np.ndarray,
    xgb_series: np.ndarray,
    last_date: pd.Timestamp,
    artifacts: Artifacts,
) -> Optional[pd.DataFrame]:
    """Forecast each of the next FORECAST_HORIZON days from the final window.

    XGBoost uses one model per lead day, each trained directly on the value h
    days ahead. Nothing is fed back, so there is no error accumulation and the
    day-7 forecast keeps the same predictive spread as day 1 -- a recursive
    rollout loses about 15% of its spread by day 7 on this model, which makes
    the later days look stable while quietly carrying less information.

    The LSTM has no per-horizon variant (its architecture is fixed across the
    project), so it is rolled forward recursively, exactly as at training time.
    The two are then combined per lead day with that lead's own inverse-RMSE
    weights.
    """
    if not artifacts.supports_multi_day:
        return None

    window = artifacts.window_size
    lam = artifacts.box_cox_lambda

    lstm_window = lstm_series[-window:].copy()
    xgb_window = xgb_series[-window:].reshape(1, window)

    rows = []
    for h in sorted(artifacts.horizon_models):
        # LSTM: recursive, one step at a time, in scaled Box-Cox space.
        step_scaled = artifacts.lstm_model.predict(
            lstm_window.reshape(1, window, 1), verbose=0
        ).flatten()
        step_bc = artifacts.scaler.inverse_transform(step_scaled.reshape(-1, 1))[:, 0]
        lstm_h = float(inv_boxcox1p(step_bc, lam)[0])
        lstm_window = np.concatenate([lstm_window[1:], step_scaled])

        # XGBoost: the model trained for this lead, on the unchanged window.
        xgb_h = float(inv_boxcox1p(artifacts.horizon_models[h].predict(xgb_window), lam)[0])

        w_lstm, w_xgb = artifacts.horizon_weights.get(
            h, (artifacts.weight_lstm, artifacts.weight_xgb)
        )
        rows.append(
            {
                "Lead day": h,
                "Date": pd.Timestamp(last_date) + pd.Timedelta(days=h),
                "LSTM": max(lstm_h, 0.0),
                "XGBoost": max(xgb_h, 0.0),
                "Ensemble": max(w_lstm * lstm_h + w_xgb * xgb_h, 0.0),
            }
        )

    return pd.DataFrame(rows)


def run_inference(clean_df: pd.DataFrame, artifacts: Artifacts) -> InferenceResult:
    """Run the saved ensemble on an uploaded, already validated series.

    Produces two kinds of output:
      - a walk-forward backtest over every day in the file that has a full
        preceding window (so it can be scored against the actual value
        that was also in the file), useful for checking the model against
        a period the user already knows the outcome for.
      - a single true out-of-sample forecast for the day immediately after
        the last date in the file, using the last WINDOW_SIZE days as
        context. This has no known actual to compare against.

    Preprocessing order matches the notebook exactly: Box-Cox with the
    saved lambda first, then Min-Max scaling with the saved (already
    fitted) scaler for the LSTM path, while XGBoost consumes the unscaled
    Box-Cox values.
    """
    window = artifacts.window_size
    lam = artifacts.box_cox_lambda

    prcp = clean_df["PRCP"].to_numpy(dtype=float)
    bc = boxcox1p_transform(prcp, lam)
    scaled = artifacts.scaler.transform(bc.reshape(-1, 1))[:, 0]

    xgb_series = bc
    lstm_series = scaled

    x_lstm, y_lstm_bc = split_sequences(lstm_series, window)
    x_lag, y_lag_bc = split_sequences(xgb_series, window)

    if len(x_lstm) > 0:
        x_lstm_r = x_lstm.reshape(-1, window, 1)
        lstm_pred_scaled = artifacts.lstm_model.predict(x_lstm_r, verbose=0)
        lstm_pred_bc = artifacts.scaler.inverse_transform(lstm_pred_scaled)[:, 0]
        backtest_lstm = inv_boxcox1p(lstm_pred_bc, lam)

        xgb_pred_bc = artifacts.xgb_model.predict(x_lag)
        backtest_xgb = inv_boxcox1p(xgb_pred_bc, lam)

        backtest_ensemble = artifacts.weight_lstm * backtest_lstm + artifacts.weight_xgb * backtest_xgb
        backtest_actual = inv_boxcox1p(y_lag_bc, lam)
        backtest_dates = clean_df["DATE"].to_numpy()[window : window + len(backtest_lstm)]
    else:
        backtest_lstm = np.array([])
        backtest_xgb = np.array([])
        backtest_ensemble = np.array([])
        backtest_actual = None
        backtest_dates = np.array([])

    last_window_lstm = lstm_series[-window:].reshape(1, window, 1)
    last_window_xgb = xgb_series[-window:].reshape(1, window)

    forecast_lstm_scaled = artifacts.lstm_model.predict(last_window_lstm, verbose=0)
    forecast_lstm_bc = artifacts.scaler.inverse_transform(forecast_lstm_scaled)[:, 0]
    forecast_lstm = float(inv_boxcox1p(forecast_lstm_bc, lam)[0])

    forecast_xgb_bc = artifacts.xgb_model.predict(last_window_xgb)
    forecast_xgb = float(inv_boxcox1p(forecast_xgb_bc, lam)[0])

    forecast_ensemble = artifacts.weight_lstm * forecast_lstm + artifacts.weight_xgb * forecast_xgb
    last_date = pd.Timestamp(clean_df["DATE"].iloc[-1])
    forecast_date = last_date + pd.Timedelta(days=1)

    forecast_7day = multi_day_forecast(lstm_series, xgb_series, last_date, artifacts)

    return InferenceResult(
        backtest_dates=backtest_dates,
        backtest_actual=backtest_actual,
        backtest_lstm=backtest_lstm,
        backtest_xgb=backtest_xgb,
        backtest_ensemble=backtest_ensemble,
        forecast_date=forecast_date,
        forecast_lstm=forecast_lstm,
        forecast_xgb=forecast_xgb,
        forecast_ensemble=forecast_ensemble,
        forecast_7day=forecast_7day,
    )
