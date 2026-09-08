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
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.special import boxcox as boxcox_apply
from scipy.special import inv_boxcox
from scipy.stats import pearsonr, skew, kurtosis
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, explained_variance_score

# Which trained-model folder under models/ the site loads and visualises.
# Defaults to the locked "v2_static" pipeline. Set the PRECIP_MODEL_DIR
# environment variable before launching Streamlit to point the whole site at
# a different run, e.g. the diagnostic copy:
#     PRECIP_MODEL_DIR=v2_static_xgb_experiment streamlit run app.py
# The value is a folder name directly under models/ (not a full path).
ACTIVE_MODEL_NAME = os.environ.get("PRECIP_MODEL_DIR", "v2_static").strip() or "v2_static"
MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / ACTIVE_MODEL_NAME
REQUIRED_ARTIFACTS = {
    "metadata": MODEL_DIR / "metadata.json",
    "lstm_model": MODEL_DIR / "lstm_model.keras",
    "xgboost_model": MODEL_DIR / "xgboost_model.json",
    "scaler": MODEL_DIR / "scaler.joblib",
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

    return Artifacts(
        metadata=metadata,
        lstm_model=lstm_model,
        xgb_model=xgb_model,
        scaler=scaler,
        window_size=int(metadata["lstm_config"]["window_size"]),
        box_cox_lambda=float(metadata["preprocessing"]["box_cox_lambda"]),
        weight_lstm=float(metadata["ensemble"]["weight_lstm"]),
        weight_xgb=float(metadata["ensemble"]["weight_xgb"]),
    )


def boxcox1p_transform(data: np.ndarray, lam: float) -> np.ndarray:
    """Apply a Box-Cox transform with a FIXED, already fitted lambda to
    (data + 1), matching boxcox1p() at training time. This never estimates
    a new lambda from the data passed in."""
    return boxcox_apply(np.asarray(data, dtype=float) + 1.0, lam)


def inv_boxcox1p(bcdata: np.ndarray, lam: float) -> np.ndarray:
    return inv_boxcox(np.asarray(bcdata, dtype=float), lam) - 1.0


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
    }


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
    forecast_date = pd.Timestamp(clean_df["DATE"].iloc[-1]) + pd.Timedelta(days=1)

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
    )
