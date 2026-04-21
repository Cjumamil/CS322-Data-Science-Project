"""Shared feature engineering and artifact helpers for XGBoost entry filters."""

from __future__ import annotations

import pickle
import re
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

CONTEXT_FEATURE_COLUMNS = [
    "close_to_ema12_frac",
    "close_to_ema26_frac",
    "close_to_ema200_high_frac",
    "close_to_ema200_close_frac",
    "close_to_ema200_low_frac",
    "ema12_to_ema26_frac",
    "ema12_to_ema200_close_frac",
    "ema26_to_ema200_close_frac",
    "intrabar_range_frac",
    "candle_body_frac",
    "upper_wick_frac",
    "lower_wick_frac",
    "gap_from_prev_close_frac",
    "close_return_1bar_frac",
    "close_return_3bar_frac",
    "close_return_6bar_frac",
    "realized_vol_6bar",
    "realized_vol_12bar",
    "range_mean_6bar_frac",
    "range_mean_12bar_frac",
    "volume_log1p",
    "volume_ratio_20bar",
    "stop_distance_vs_range_6bar",
    "stop_distance_vs_range_12bar",
]

METADATA_COLUMNS = {
    "symbol",
    "entry_signal_time",
    "side",
    "pnl",
    "return_pct",
    "bars_held",
    "label",
}


def _empty_float_series(index: pd.Index) -> pd.Series:
    return pd.Series(0.0, index=index, dtype="float64")


def _empty_bool_series(index: pd.Index) -> pd.Series:
    return pd.Series(False, index=index, dtype="bool")


def _safe_numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    return pd.to_numeric(df.get(column, _empty_float_series(df.index)), errors="coerce").replace(
        [np.inf, -np.inf],
        np.nan,
    )


def _safe_ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    clean_denominator = denominator.replace(0, np.nan)
    return numerator.divide(clean_denominator).replace([np.inf, -np.inf], np.nan)


def _symbol_feature_name(symbol: str) -> str:
    cleaned = re.sub(r"[^A-Z0-9]+", "_", str(symbol).upper()).strip("_")
    return f"symbol_{cleaned or 'UNKNOWN'}"


def build_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Convert one prepared strategy DataFrame into a stable ML feature matrix."""
    features = pd.DataFrame(index=df.index)
    close = _safe_numeric_series(df, "Close")
    open_price = _safe_numeric_series(df, "Open")
    high = _safe_numeric_series(df, "High")
    low = _safe_numeric_series(df, "Low")
    volume = _safe_numeric_series(df, "Volume")

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

    ema12 = _safe_numeric_series(df, "EMA12")
    ema26 = _safe_numeric_series(df, "EMA26")
    ema200_high = _safe_numeric_series(df, "EMA200_high")
    ema200_close = _safe_numeric_series(df, "EMA200_close")
    ema200_low = _safe_numeric_series(df, "EMA200_low")
    prev_close = close.shift(1)
    bar_range = (high - low).clip(lower=0)
    candle_body = close - open_price
    upper_wick = high - pd.concat([open_price, close], axis=1).max(axis=1)
    lower_wick = pd.concat([open_price, close], axis=1).min(axis=1) - low
    close_returns = close.pct_change().replace([np.inf, -np.inf], np.nan)
    rolling_volume_mean = volume.rolling(20, min_periods=5).mean()
    rolling_range_mean_6 = bar_range.rolling(6, min_periods=3).mean()
    rolling_range_mean_12 = bar_range.rolling(12, min_periods=6).mean()

    context_features = {
        "close_to_ema12_frac": _safe_ratio(close - ema12, close),
        "close_to_ema26_frac": _safe_ratio(close - ema26, close),
        "close_to_ema200_high_frac": _safe_ratio(close - ema200_high, close),
        "close_to_ema200_close_frac": _safe_ratio(close - ema200_close, close),
        "close_to_ema200_low_frac": _safe_ratio(close - ema200_low, close),
        "ema12_to_ema26_frac": _safe_ratio(ema12 - ema26, close),
        "ema12_to_ema200_close_frac": _safe_ratio(ema12 - ema200_close, close),
        "ema26_to_ema200_close_frac": _safe_ratio(ema26 - ema200_close, close),
        "intrabar_range_frac": _safe_ratio(bar_range, close),
        "candle_body_frac": _safe_ratio(candle_body, close),
        "upper_wick_frac": _safe_ratio(upper_wick.clip(lower=0), close),
        "lower_wick_frac": _safe_ratio(lower_wick.clip(lower=0), close),
        "gap_from_prev_close_frac": _safe_ratio(open_price - prev_close, prev_close),
        "close_return_1bar_frac": close_returns,
        "close_return_3bar_frac": close.pct_change(3).replace([np.inf, -np.inf], np.nan),
        "close_return_6bar_frac": close.pct_change(6).replace([np.inf, -np.inf], np.nan),
        "realized_vol_6bar": close_returns.rolling(6, min_periods=3).std(),
        "realized_vol_12bar": close_returns.rolling(12, min_periods=6).std(),
        "range_mean_6bar_frac": _safe_ratio(rolling_range_mean_6, close),
        "range_mean_12bar_frac": _safe_ratio(rolling_range_mean_12, close),
        "volume_log1p": np.log1p(volume.clip(lower=0)),
        "volume_ratio_20bar": _safe_ratio(volume, rolling_volume_mean),
        "stop_distance_vs_range_6bar": _safe_ratio(_safe_numeric_series(df, "stop_distance"), rolling_range_mean_6),
        "stop_distance_vs_range_12bar": _safe_ratio(_safe_numeric_series(df, "stop_distance"), rolling_range_mean_12),
    }
    for column in CONTEXT_FEATURE_COLUMNS:
        features[column] = context_features[column].fillna(0.0).astype("float64")

    symbol_series = df.get("symbol", pd.Series("UNKNOWN", index=df.index, dtype="object")).fillna("UNKNOWN").astype(str)
    for symbol in sorted(symbol_series.str.upper().unique()):
        features[_symbol_feature_name(symbol)] = (symbol_series.str.upper() == symbol).astype("int8")

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


def apply_xgboost_entry_filter(
    prepared_df: pd.DataFrame,
    model_path: str,
    probability_threshold: float,
) -> pd.DataFrame:
    """Apply one saved XGBoost entry gate to a prepared MACD pullback frame."""
    work = prepared_df.copy()
    base_entry_action = work["entry_action"].copy()
    candidate_mask = base_entry_action.notna()
    probabilities = pd.Series(np.nan, index=work.index, dtype="float64")

    if candidate_mask.any():
        candidate_probabilities = predict_trade_quality_probabilities(work.loc[candidate_mask], model_path)
        probabilities.loc[candidate_mask] = candidate_probabilities

    filter_pass = probabilities >= float(probability_threshold)

    work["base_entry_action"] = base_entry_action
    work["xgb_trade_quality_prob"] = probabilities
    work["xgb_probability_threshold"] = float(probability_threshold)
    work["xgb_filter_pass"] = filter_pass.fillna(False)
    work["entry_blocked_by_xgb_filter"] = candidate_mask & ~work["xgb_filter_pass"]
    work["entry_action"] = base_entry_action.where(work["xgb_filter_pass"])
    work["long_entry_signal"] = work["entry_action"] == "BUY"
    work["short_entry_signal"] = work["entry_action"] == "SELL"

    exposure_signal = pd.Series(index=work.index, dtype="float64")
    exposure_signal.loc[work["long_entry_signal"]] = 1
    exposure_signal.loc[work["short_entry_signal"]] = -1
    work["signal"] = exposure_signal.ffill().fillna(0).astype(int)
    work["crossover"] = 0
    work.loc[work["long_entry_signal"], "crossover"] = 1
    work.loc[work["short_entry_signal"], "crossover"] = -1
    work["strategy_return"] = work["signal"].shift(1).fillna(0) * work["daily_return"].fillna(0)
    return work


def build_labeled_trade_examples(
    prepared_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    *,
    symbol: str,
) -> pd.DataFrame:
    """Map executed trades back to entry-signal bars with features and metadata."""
    if trades_df.empty:
        return pd.DataFrame()

    work = prepared_df.copy()
    work["symbol"] = str(symbol).upper()
    feature_frame = build_feature_frame(work)
    rows: list[dict] = []

    for _, trade in trades_df.iterrows():
        entry_signal_time = pd.Timestamp(trade["entry_signal_time"])
        if entry_signal_time not in feature_frame.index:
            continue
        feature_row = feature_frame.loc[entry_signal_time].to_dict()
        rows.append(
            {
                **feature_row,
                "symbol": str(symbol).upper(),
                "entry_signal_time": entry_signal_time.isoformat(),
                "side": str(trade["side"]),
                "pnl": float(trade["pnl"]),
                "return_pct": float(trade["return_pct"]),
                "bars_held": int(trade["bars_held"]),
                "label": 1 if float(trade["pnl"]) > 0 else 0,
            }
        )

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def build_training_examples(prepared_df: pd.DataFrame, trades_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Map executed training trades back to their entry-signal bars."""
    symbol = str(
        prepared_df.get("symbol", pd.Series(["UNKNOWN"], dtype="object")).iloc[0]
        if not prepared_df.empty
        else "UNKNOWN"
    )
    labeled_examples = build_labeled_trade_examples(prepared_df, trades_df, symbol=symbol)
    if labeled_examples.empty:
        return pd.DataFrame(), pd.Series(dtype="int64")
    feature_columns = [column for column in labeled_examples.columns if column not in METADATA_COLUMNS]
    return labeled_examples[feature_columns].copy(), labeled_examples["label"].astype("int64").copy()


def select_feature_columns(
    feature_frame: pd.DataFrame,
    *,
    include_symbol_features: bool,
    include_context_features: bool,
) -> list[str]:
    """Choose one feature subset for a pooled model variant."""
    feature_columns: list[str] = []
    for column in feature_frame.columns:
        if column in METADATA_COLUMNS:
            continue
        if column.startswith("symbol_") and not include_symbol_features:
            continue
        if column in CONTEXT_FEATURE_COLUMNS and not include_context_features:
            continue
        feature_columns.append(column)
    return feature_columns
