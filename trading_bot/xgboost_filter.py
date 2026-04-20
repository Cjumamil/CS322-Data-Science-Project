"""Shared feature engineering and artifact helpers for XGBoost entry filters."""

from __future__ import annotations

import pickle
import site
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd


USER_SITE_PACKAGES = site.getusersitepackages()
WORKSPACE_XGBOOST_PACKAGES = Path(__file__).resolve().parents[1] / "workspace_pkgs" / "xgboost_local"
LEGACY_WORKSPACE_XGBOOST_PACKAGES = Path(__file__).resolve().parents[1] / ".vendor_pkgs" / "xgboost_local"
if WORKSPACE_XGBOOST_PACKAGES.exists() and str(WORKSPACE_XGBOOST_PACKAGES) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_XGBOOST_PACKAGES))
elif LEGACY_WORKSPACE_XGBOOST_PACKAGES.exists() and str(LEGACY_WORKSPACE_XGBOOST_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LEGACY_WORKSPACE_XGBOOST_PACKAGES))
if USER_SITE_PACKAGES and USER_SITE_PACKAGES not in sys.path:
    site.addsitedir(USER_SITE_PACKAGES)


NUMERIC_FEATURE_COLUMNS = [
    "Close",
    "EMA12",
    "EMA26",
    "EMA200_high",
    "EMA200_close",
    "EMA200_low",
    "MACD",
    "signal_line",
    "histogram",
    "histogram_previous",
    "ema_band_slope_frac",
    "recent_long_extension_frac",
    "recent_short_extension_frac",
    "macd_slope",
    "macd_slope_frac",
    "stop_price",
    "stop_distance",
    "stop_distance_frac_of_price",
]

BOOLEAN_FEATURE_COLUMNS = [
    "close_above_ema_band",
    "close_below_ema_band",
    "close_inside_ema_band",
    "long_trend",
    "short_trend",
    "sideways_market",
    "prior_impulse_long",
    "prior_impulse_short",
    "pullback_touch_long_now",
    "pullback_touch_short_now",
    "had_recent_pullback_long",
    "had_recent_pullback_short",
    "pullback_active_long",
    "pullback_active_short",
    "macd_pullback_context_long",
    "macd_pullback_context_short",
    "histogram_rising",
    "histogram_falling",
    "pullback_breakout_long",
    "pullback_breakout_short",
    "bullish_reentry_trigger",
    "bearish_reentry_trigger",
    "entries_allowed",
    "opening_no_trade_window",
    "entry_risk_plan_valid",
    "entry_rejected_max_stop_distance",
    "long_entry_setup",
    "short_entry_setup",
    "entry_blocked_by_stop_filter",
    "entry_blocked_by_max_stop_distance",
    "long_entry_signal",
    "short_entry_signal",
]


def _empty_float_series(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, dtype="float64")


def _empty_bool_series(index: pd.Index) -> pd.Series:
    return pd.Series(False, index=index, dtype="bool")


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert one prepared strategy DataFrame into a stable ML feature matrix."""
    features = pd.DataFrame(index=df.index)

    for column in NUMERIC_FEATURE_COLUMNS:
        series = df.get(column, _empty_float_series(df.index))
        cleaned = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).fillna(0.0)
        features[column] = cleaned.astype("float64")

    for column in BOOLEAN_FEATURE_COLUMNS:
        series = df.get(column, _empty_bool_series(df.index))
        features[column] = series.fillna(False).astype(bool).astype("int8")

    trend_bias = df.get("trend_bias", pd.Series("neutral", index=df.index, dtype="object")).fillna("neutral").astype(str)
    entry_action = df.get("entry_action", pd.Series(index=df.index, dtype="object"))

    features["trend_bias_long"] = (trend_bias == "long").astype("int8")
    features["trend_bias_short"] = (trend_bias == "short").astype("int8")
    features["candidate_is_long"] = (entry_action == "BUY").astype("int8")
    features["candidate_is_short"] = (entry_action == "SELL").astype("int8")

    return features


def align_feature_columns(feature_frame: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Match a feature frame to the exact model-training column order."""
    aligned = pd.DataFrame(index=feature_frame.index)
    for column in feature_columns:
        if column in feature_frame.columns:
            aligned[column] = pd.to_numeric(feature_frame[column], errors="coerce").fillna(0.0)
        else:
            aligned[column] = 0.0
    return aligned.astype("float64")


@lru_cache(maxsize=16)
def load_filter_artifact(model_path: str) -> dict:
    """Load one saved XGBoost filter artifact from disk."""
    path = Path(model_path)
    if not path.exists():
        raise ValueError(f"XGBoost filter artifact not found: {path}")

    with path.open("rb") as artifact_file:
        artifact = pickle.load(artifact_file)

    if not isinstance(artifact, dict):
        raise ValueError(f"Invalid XGBoost filter artifact format: {path}")
    if "model" not in artifact or "feature_columns" not in artifact:
        raise ValueError(f"XGBoost filter artifact is missing required keys: {path}")

    return artifact


def predict_trade_quality_probabilities(df: pd.DataFrame, model_path: str) -> pd.Series:
    """Return the positive-class probability for each prepared strategy row."""
    artifact = load_filter_artifact(str(Path(model_path).resolve()))
    feature_frame = build_feature_frame(df)
    aligned = align_feature_columns(feature_frame, list(artifact["feature_columns"]))
    probabilities = artifact["model"].predict_proba(aligned)[:, 1]
    return pd.Series(probabilities, index=df.index, dtype="float64")


def build_training_examples(prepared_df: pd.DataFrame, trades_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map executed training trades back to their entry-signal bars."""
    if trades_df.empty:
        return pd.DataFrame(), pd.Series(dtype="int64")

    feature_frame = build_feature_frame(prepared_df)
    rows: list[pd.Series] = []
    labels: list[int] = []

    for _, trade in trades_df.iterrows():
        entry_signal_time = pd.Timestamp(trade["entry_signal_time"])
        if entry_signal_time not in feature_frame.index:
            continue
        rows.append(feature_frame.loc[entry_signal_time])
        labels.append(1 if float(trade["pnl"]) > 0 else 0)

    if not rows:
        return pd.DataFrame(), pd.Series(dtype="int64")

    return pd.DataFrame(rows).reset_index(drop=True), pd.Series(labels, dtype="int64")
