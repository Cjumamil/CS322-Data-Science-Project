"""VWAP + RSI mean reversion strategy with light reversal confirmation."""

from dataclasses import dataclass

import pandas as pd

from trading_bot.risk import assess_stop_distance, round_price
from trading_bot.strategies.base import PositionContext, TradingStrategy, calculate_standard_backtest_metrics


NEW_YORK_TIMEZONE = "America/New_York"
REGULAR_MARKET_OPEN_MINUTE = (9 * 60) + 30
REGULAR_MARKET_CLOSE_MINUTE = 16 * 60


def _safe_float(value) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def _safe_bool(value) -> bool:
    if pd.isna(value):
        return False
    return bool(value)


def _calculate_rsi(close_series: pd.Series, window: int) -> pd.Series:
    """Calculate Wilder-style RSI using exponentially smoothed gains/losses."""
    delta = close_series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)

    avg_gain = gain.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    avg_loss = loss.ewm(alpha=1 / window, adjust=False, min_periods=window).mean()

    relative_strength = avg_gain / avg_loss.replace(0, pd.NA)
    rsi = 100 - (100 / (1 + relative_strength))
    rsi = rsi.where(avg_loss != 0, 100.0)
    rsi = rsi.where(~((avg_gain == 0) & (avg_loss == 0)), 50.0)
    return rsi


@dataclass(frozen=True)
class VwapRsiMeanReversionStrategy(TradingStrategy):
    """Fade intraday extensions away from VWAP with RSI stretch confirmation."""

    interval: str
    lookback_bars: int
    rsi_window: int
    rsi_oversold_threshold: float
    rsi_overbought_threshold: float
    min_distance_from_vwap_frac: float
    confirmation_lookback_bars: int
    min_reversal_move_frac_of_price: float
    require_reversal_bar: bool
    require_candle_color_confirmation: bool
    stop_reference_lookback_bars: int
    stop_buffer_pct: float
    take_profit_vwap_fraction: float
    max_stop_distance_frac_of_price: float | None
    entry_start_minutes_after_open: int
    entry_end_minutes_before_close: int
    enable_time_stop: bool
    max_bars_in_trade: int
    enable_extreme_trend_filter: bool
    trend_lookback_bars: int
    max_trend_move_frac_of_price: float
    name: str = "vwap_rsi_mean_reversion"
    version: str = "v1"

    def min_required_bars(self) -> int:
        return max(
            self.rsi_window + 5,
            self.confirmation_lookback_bars + 2,
            self.stop_reference_lookback_bars + 2,
            self.trend_lookback_bars + 2,
        )

    def _interval_minutes(self) -> int:
        interval = self.interval.strip().lower()
        if not interval.endswith("m"):
            raise ValueError(f"{self.name} currently expects an intraday minute interval, got: {self.interval}")
        return int(interval[:-1])

    def _bars_in_trade_exceeded(self, position_context: PositionContext | None) -> bool:
        if not self.enable_time_stop or position_context is None or position_context.bars_in_trade is None:
            return False
        return position_context.bars_in_trade >= self.max_bars_in_trade

    def _build_entry_risk_snapshot(
        self,
        *,
        entry_price: float | None,
        vwap: float | None,
        stop_reference_price: float | None,
        position_side: str,
    ) -> dict:
        """Build one candidate entry risk plan around a recent extreme and VWAP target."""
        snapshot = {
            "position_side": position_side,
            "stop_price": None,
            "stop_distance": None,
            "stop_distance_frac_of_price": None,
            "max_stop_distance_frac_of_price": self.max_stop_distance_frac_of_price,
            "rejected_due_to_max_stop_distance": False,
            "is_valid": False,
            "rejection_reason": None,
            "risk_reward_multiple": None,
            "take_profit_price": None,
            "take_profit_source": "vwap_reversion_fraction",
            "take_profit_vwap_fraction": self.take_profit_vwap_fraction,
            "stop_source": "recent_extreme_buffer",
            "stop_reference_price": stop_reference_price,
            "stop_buffer_pct": self.stop_buffer_pct,
            "target_vwap": vwap,
        }

        if entry_price is None or entry_price <= 0:
            snapshot["rejection_reason"] = "invalid_entry_price"
            return snapshot

        if vwap is None:
            snapshot["rejection_reason"] = "missing_vwap"
            return snapshot

        if stop_reference_price is None:
            snapshot["rejection_reason"] = "missing_stop_reference"
            return snapshot

        if self.take_profit_vwap_fraction <= 0:
            snapshot["rejection_reason"] = "invalid_take_profit_vwap_fraction"
            return snapshot

        if position_side == "long":
            stop_price = round_price(float(stop_reference_price) * (1 - self.stop_buffer_pct))
        elif position_side == "short":
            stop_price = round_price(float(stop_reference_price) * (1 + self.stop_buffer_pct))
        else:
            snapshot["rejection_reason"] = "invalid_position_side"
            return snapshot

        validation = assess_stop_distance(
            entry_price=entry_price,
            stop_price=stop_price,
            position_side=position_side,
            max_stop_distance_frac_of_price=self.max_stop_distance_frac_of_price,
        )
        snapshot.update(validation)
        if not validation["is_valid"]:
            return snapshot

        target_price = entry_price + ((float(vwap) - entry_price) * self.take_profit_vwap_fraction)
        target_price = round_price(target_price)
        snapshot["take_profit_price"] = target_price

        if position_side == "long" and target_price <= entry_price:
            snapshot["rejection_reason"] = "target_not_above_entry"
            snapshot["is_valid"] = False
            return snapshot
        if position_side == "short" and target_price >= entry_price:
            snapshot["rejection_reason"] = "target_not_below_entry"
            snapshot["is_valid"] = False
            return snapshot

        snapshot["is_valid"] = True
        return snapshot

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add VWAP, RSI, and mean-reversion concept columns to price data."""
        work = df.copy().sort_index()
        if "Volume" not in work.columns:
            raise ValueError(f"{self.name} requires Volume data to compute intraday VWAP.")

        localized_index = work.index
        if getattr(localized_index, "tz", None) is None:
            localized_index = localized_index.tz_localize("UTC")
        localized_index = localized_index.tz_convert(NEW_YORK_TIMEZONE)
        session_dates = pd.Index(localized_index.date)

        typical_price = (work["High"] + work["Low"] + work["Close"]) / 3
        volume = work["Volume"].fillna(0)
        cumulative_volume = volume.groupby(session_dates).cumsum()
        cumulative_tpv = (typical_price * volume).groupby(session_dates).cumsum()

        work["VWAP"] = cumulative_tpv / cumulative_volume.where(cumulative_volume > 0)
        work["RSI"] = _calculate_rsi(work["Close"], self.rsi_window)
        work["rsi_oversold_level"] = self.rsi_oversold_threshold
        work["rsi_overbought_level"] = self.rsi_overbought_threshold

        work["price_minus_vwap"] = work["Close"] - work["VWAP"]
        work["price_distance_from_vwap"] = work["price_minus_vwap"].abs()
        work["price_distance_from_vwap_frac"] = work["price_distance_from_vwap"] / work["Close"].where(work["Close"] > 0)
        work["below_vwap"] = (work["Close"] < work["VWAP"]).fillna(False)
        work["above_vwap"] = (work["Close"] > work["VWAP"]).fillna(False)
        work["rsi_oversold"] = (work["RSI"] <= self.rsi_oversold_threshold).fillna(False)
        work["rsi_overbought"] = (work["RSI"] >= self.rsi_overbought_threshold).fillna(False)
        work["stretched_from_vwap_long"] = (
            work["below_vwap"] & (work["price_distance_from_vwap_frac"] >= self.min_distance_from_vwap_frac)
        ).fillna(False)
        work["stretched_from_vwap_short"] = (
            work["above_vwap"] & (work["price_distance_from_vwap_frac"] >= self.min_distance_from_vwap_frac)
        ).fillna(False)

        work["recent_low_confirmation"] = work["Low"].rolling(
            self.confirmation_lookback_bars,
            min_periods=1,
        ).min()
        work["recent_high_confirmation"] = work["High"].rolling(
            self.confirmation_lookback_bars,
            min_periods=1,
        ).max()
        work["reversal_move_long_frac"] = (
            (work["Close"] - work["recent_low_confirmation"]) / work["Close"].where(work["Close"] > 0)
        )
        work["reversal_move_short_frac"] = (
            (work["recent_high_confirmation"] - work["Close"]) / work["Close"].where(work["Close"] > 0)
        )
        work["reversal_bar_long"] = (work["Close"] > work["Close"].shift(1)).fillna(False)
        work["reversal_bar_short"] = (work["Close"] < work["Close"].shift(1)).fillna(False)
        work["bullish_signal_bar"] = (work["Close"] > work["Open"]).fillna(False)
        work["bearish_signal_bar"] = (work["Close"] < work["Open"]).fillna(False)

        stabilization_long = (work["reversal_move_long_frac"] >= self.min_reversal_move_frac_of_price).fillna(False)
        stabilization_short = (work["reversal_move_short_frac"] >= self.min_reversal_move_frac_of_price).fillna(False)
        if self.require_reversal_bar:
            stabilization_long &= work["reversal_bar_long"]
            stabilization_short &= work["reversal_bar_short"]
        if self.require_candle_color_confirmation:
            stabilization_long &= work["bullish_signal_bar"]
            stabilization_short &= work["bearish_signal_bar"]
        work["stabilization_confirmation_long"] = stabilization_long.fillna(False)
        work["stabilization_confirmation_short"] = stabilization_short.fillna(False)

        work["recent_price_change_frac"] = (
            (work["Close"] - work["Close"].shift(self.trend_lookback_bars))
            / work["Close"].where(work["Close"] > 0)
        )
        work["extreme_downtrend"] = (
            work["recent_price_change_frac"] <= -self.max_trend_move_frac_of_price
        ).fillna(False)
        work["extreme_uptrend"] = (
            work["recent_price_change_frac"] >= self.max_trend_move_frac_of_price
        ).fillna(False)
        if self.enable_extreme_trend_filter:
            work["trend_filter_pass_long"] = ~work["extreme_downtrend"]
            work["trend_filter_pass_short"] = ~work["extreme_uptrend"]
        else:
            work["trend_filter_pass_long"] = True
            work["trend_filter_pass_short"] = True
        work["trend_filter_blocked_long"] = (~work["trend_filter_pass_long"]).fillna(False)
        work["trend_filter_blocked_short"] = (~work["trend_filter_pass_short"]).fillna(False)

        minutes_of_day = (localized_index.hour * 60) + localized_index.minute
        start_minute = REGULAR_MARKET_OPEN_MINUTE + max(0, self.entry_start_minutes_after_open)
        end_minute = REGULAR_MARKET_CLOSE_MINUTE - max(0, self.entry_end_minutes_before_close)
        work["regular_session_bar"] = (
            (minutes_of_day >= REGULAR_MARKET_OPEN_MINUTE)
            & (minutes_of_day < REGULAR_MARKET_CLOSE_MINUTE)
        )
        work["time_window_active"] = (
            work["regular_session_bar"]
            & (minutes_of_day >= start_minute)
            & (minutes_of_day < end_minute)
        )
        work["entries_allowed"] = work["time_window_active"]

        work["long_core_setup"] = (
            work["entries_allowed"]
            & work["below_vwap"]
            & work["rsi_oversold"]
            & work["stretched_from_vwap_long"]
        )
        work["short_core_setup"] = (
            work["entries_allowed"]
            & work["above_vwap"]
            & work["rsi_overbought"]
            & work["stretched_from_vwap_short"]
        )
        work["long_setup"] = work["long_core_setup"]
        work["short_setup"] = work["short_core_setup"]

        work["recent_low_stop_reference"] = work["Low"].rolling(
            self.stop_reference_lookback_bars,
            min_periods=1,
        ).min()
        work["recent_high_stop_reference"] = work["High"].rolling(
            self.stop_reference_lookback_bars,
            min_periods=1,
        ).max()

        long_risk_snapshots = []
        short_risk_snapshots = []
        for _, row in work.iterrows():
            long_risk_snapshots.append(
                self._build_entry_risk_snapshot(
                    entry_price=_safe_float(row["Close"]),
                    vwap=_safe_float(row["VWAP"]),
                    stop_reference_price=_safe_float(row["recent_low_stop_reference"]),
                    position_side="long",
                )
            )
            short_risk_snapshots.append(
                self._build_entry_risk_snapshot(
                    entry_price=_safe_float(row["Close"]),
                    vwap=_safe_float(row["VWAP"]),
                    stop_reference_price=_safe_float(row["recent_high_stop_reference"]),
                    position_side="short",
                )
            )

        long_risk_df = pd.DataFrame(long_risk_snapshots, index=work.index)
        short_risk_df = pd.DataFrame(short_risk_snapshots, index=work.index)
        work["long_stop_price"] = long_risk_df["stop_price"]
        work["short_stop_price"] = short_risk_df["stop_price"]
        work["long_take_profit_price"] = long_risk_df["take_profit_price"]
        work["short_take_profit_price"] = short_risk_df["take_profit_price"]
        work["long_stop_distance"] = long_risk_df["stop_distance"]
        work["short_stop_distance"] = short_risk_df["stop_distance"]
        work["long_stop_distance_frac_of_price"] = long_risk_df["stop_distance_frac_of_price"]
        work["short_stop_distance_frac_of_price"] = short_risk_df["stop_distance_frac_of_price"]
        work["long_entry_risk_plan_valid"] = long_risk_df["is_valid"].fillna(False)
        work["short_entry_risk_plan_valid"] = short_risk_df["is_valid"].fillna(False)
        work["long_entry_rejected_max_stop_distance"] = (
            long_risk_df["rejected_due_to_max_stop_distance"].fillna(False)
        )
        work["short_entry_rejected_max_stop_distance"] = (
            short_risk_df["rejected_due_to_max_stop_distance"].fillna(False)
        )
        work["long_entry_rejection_reason"] = long_risk_df["rejection_reason"]
        work["short_entry_rejection_reason"] = short_risk_df["rejection_reason"]

        work["long_entry_setup"] = (
            work["long_core_setup"]
            & work["stabilization_confirmation_long"]
            & work["trend_filter_pass_long"]
        )
        work["short_entry_setup"] = (
            work["short_core_setup"]
            & work["stabilization_confirmation_short"]
            & work["trend_filter_pass_short"]
        )
        work["long_entry_signal"] = work["long_entry_setup"] & work["long_entry_risk_plan_valid"]
        work["short_entry_signal"] = work["short_entry_setup"] & work["short_entry_risk_plan_valid"]
        work["entry_blocked_by_stop_filter"] = (
            (work["long_entry_setup"] & ~work["long_entry_risk_plan_valid"])
            | (work["short_entry_setup"] & ~work["short_entry_risk_plan_valid"])
        )
        work["entry_blocked_by_max_stop_distance"] = (
            (work["long_entry_setup"] & work["long_entry_rejected_max_stop_distance"])
            | (work["short_entry_setup"] & work["short_entry_rejected_max_stop_distance"])
        )

        work["entry_action"] = pd.Series(index=work.index, dtype="object")
        work.loc[work["long_entry_signal"], "entry_action"] = "BUY"
        work.loc[work["short_entry_signal"], "entry_action"] = "SELL"
        work["exit_action"] = pd.Series(index=work.index, dtype="object")
        work["long_exit_signal"] = False
        work["short_exit_signal"] = False

        exposure_signal = pd.Series(index=work.index, dtype="float64")
        exposure_signal.loc[work["long_entry_signal"]] = 1
        exposure_signal.loc[work["short_entry_signal"]] = -1
        work["signal"] = exposure_signal.ffill().fillna(0).astype(int)
        work["crossover"] = 0
        work.loc[work["long_entry_signal"], "crossover"] = 1
        work.loc[work["short_entry_signal"], "crossover"] = -1
        work["daily_return"] = work["Close"].pct_change()
        work["strategy_return"] = work["signal"].shift(1).fillna(0) * work["daily_return"].fillna(0)
        work["predicted_close"] = work["Close"].shift(1)
        work["abs_error"] = (work["Close"] - work["predicted_close"]).abs()

        work = work.dropna(subset=["VWAP", "RSI"]).copy()
        return work

    def recent_display_columns(self) -> list[str]:
        return [
            "Close",
            "VWAP",
            "RSI",
            "price_distance_from_vwap_frac",
            "long_setup",
            "short_setup",
            "stabilization_confirmation_long",
            "stabilization_confirmation_short",
            "long_entry_signal",
            "short_entry_signal",
            "entry_action",
        ]

    def latest_signal_label(self) -> str:
        return "Latest entry action"

    def latest_signal_value(self, latest_row: pd.Series):
        return self.entry_action(latest_row) or "NONE"

    def event_key(self, latest_row: pd.Series):
        bar_timestamp = latest_row.name.isoformat() if hasattr(latest_row.name, "isoformat") else str(latest_row.name)
        return f"{bar_timestamp}:{self.latest_signal_value(latest_row)}"

    def entry_action(self, latest_row: pd.Series) -> str | None:
        action = latest_row.get("entry_action")
        if pd.isna(action):
            return None
        return str(action)

    def exit_action(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> str | None:
        if position_context is None:
            return None
        if position_context.side == "long" and self.should_exit_long(latest_row, position_context):
            return "SELL"
        if position_context.side == "short" and self.should_exit_short(latest_row, position_context):
            return "BUY"
        return None

    def should_enter_long(self, latest_row: pd.Series) -> bool:
        return self.entry_action(latest_row) == "BUY"

    def should_enter_short(self, latest_row: pd.Series) -> bool:
        return self.entry_action(latest_row) == "SELL"

    def should_exit_long(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> bool:
        return self._bars_in_trade_exceeded(position_context)

    def should_exit_short(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> bool:
        return self._bars_in_trade_exceeded(position_context)

    def entry_reason(self) -> str:
        return self.name

    def exit_reason(
        self,
        latest_row: pd.Series | None = None,
        position_context: PositionContext | None = None,
    ) -> str:
        if self._bars_in_trade_exceeded(position_context):
            return "time_stop"
        return self.name

    def build_entry_risk_plan(self, latest_row: pd.Series, position_side: str, risk_settings) -> dict | None:
        if position_side == "long":
            rejection_reason = latest_row.get("long_entry_rejection_reason")
            return {
                "position_side": "long",
                "stop_price": _safe_float(latest_row.get("long_stop_price")),
                "take_profit_price": _safe_float(latest_row.get("long_take_profit_price")),
                "take_profit_source": "vwap_reversion_fraction",
                "risk_reward_multiple": None,
                "stop_distance": _safe_float(latest_row.get("long_stop_distance")),
                "stop_distance_frac_of_price": _safe_float(latest_row.get("long_stop_distance_frac_of_price")),
                "max_stop_distance_frac_of_price": self.max_stop_distance_frac_of_price,
                "rejected_due_to_max_stop_distance": _safe_bool(
                    latest_row.get("long_entry_rejected_max_stop_distance")
                ),
                "is_valid": _safe_bool(latest_row.get("long_entry_risk_plan_valid")),
                "rejection_reason": None if pd.isna(rejection_reason) else str(rejection_reason),
                "stop_source": "recent_extreme_buffer",
                "stop_reference_price": _safe_float(latest_row.get("recent_low_stop_reference")),
                "stop_buffer_pct": self.stop_buffer_pct,
                "target_vwap": _safe_float(latest_row.get("VWAP")),
                "take_profit_vwap_fraction": self.take_profit_vwap_fraction,
            }
        if position_side == "short":
            rejection_reason = latest_row.get("short_entry_rejection_reason")
            return {
                "position_side": "short",
                "stop_price": _safe_float(latest_row.get("short_stop_price")),
                "take_profit_price": _safe_float(latest_row.get("short_take_profit_price")),
                "take_profit_source": "vwap_reversion_fraction",
                "risk_reward_multiple": None,
                "stop_distance": _safe_float(latest_row.get("short_stop_distance")),
                "stop_distance_frac_of_price": _safe_float(latest_row.get("short_stop_distance_frac_of_price")),
                "max_stop_distance_frac_of_price": self.max_stop_distance_frac_of_price,
                "rejected_due_to_max_stop_distance": _safe_bool(
                    latest_row.get("short_entry_rejected_max_stop_distance")
                ),
                "is_valid": _safe_bool(latest_row.get("short_entry_risk_plan_valid")),
                "rejection_reason": None if pd.isna(rejection_reason) else str(rejection_reason),
                "stop_source": "recent_extreme_buffer",
                "stop_reference_price": _safe_float(latest_row.get("recent_high_stop_reference")),
                "stop_buffer_pct": self.stop_buffer_pct,
                "target_vwap": _safe_float(latest_row.get("VWAP")),
                "take_profit_vwap_fraction": self.take_profit_vwap_fraction,
            }
        return None

    def risk_reward_multiple(self) -> float | None:
        return None

    def build_strategy_parameters(self) -> dict:
        return {
            "stop_model": "recent_extreme_buffer",
            "take_profit_model": "vwap_reversion_fraction",
            "rsi_window": self.rsi_window,
            "rsi_oversold_threshold": self.rsi_oversold_threshold,
            "rsi_overbought_threshold": self.rsi_overbought_threshold,
            "min_distance_from_vwap_frac": self.min_distance_from_vwap_frac,
            "confirmation_lookback_bars": self.confirmation_lookback_bars,
            "min_reversal_move_frac_of_price": self.min_reversal_move_frac_of_price,
            "require_reversal_bar": self.require_reversal_bar,
            "require_candle_color_confirmation": self.require_candle_color_confirmation,
            "stop_reference_lookback_bars": self.stop_reference_lookback_bars,
            "stop_buffer_pct": self.stop_buffer_pct,
            "take_profit_vwap_fraction": self.take_profit_vwap_fraction,
            "max_stop_distance_frac_of_price": self.max_stop_distance_frac_of_price,
            "entry_start_minutes_after_open": self.entry_start_minutes_after_open,
            "entry_end_minutes_before_close": self.entry_end_minutes_before_close,
            "enable_time_stop": self.enable_time_stop,
            "max_bars_in_trade": self.max_bars_in_trade,
            "enable_extreme_trend_filter": self.enable_extreme_trend_filter,
            "trend_lookback_bars": self.trend_lookback_bars,
            "max_trend_move_frac_of_price": self.max_trend_move_frac_of_price,
        }

    def build_strategy_signals(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> dict:
        bars_in_trade = None if position_context is None else position_context.bars_in_trade
        time_stop_triggered = self._bars_in_trade_exceeded(position_context)
        resolved_exit_action = self.exit_action(latest_row, position_context)
        long_rejection_reason = latest_row.get("long_entry_rejection_reason")
        short_rejection_reason = latest_row.get("short_entry_rejection_reason")

        return {
            "vwap": _safe_float(latest_row.get("VWAP")),
            "price_minus_vwap": _safe_float(latest_row.get("price_minus_vwap")),
            "price_distance_from_vwap": _safe_float(latest_row.get("price_distance_from_vwap")),
            "price_distance_from_vwap_frac": _safe_float(latest_row.get("price_distance_from_vwap_frac")),
            "rsi": _safe_float(latest_row.get("RSI")),
            "rsi_oversold": _safe_bool(latest_row.get("rsi_oversold")),
            "rsi_overbought": _safe_bool(latest_row.get("rsi_overbought")),
            "below_vwap": _safe_bool(latest_row.get("below_vwap")),
            "above_vwap": _safe_bool(latest_row.get("above_vwap")),
            "stretched_from_vwap_long": _safe_bool(latest_row.get("stretched_from_vwap_long")),
            "stretched_from_vwap_short": _safe_bool(latest_row.get("stretched_from_vwap_short")),
            "reversal_move_long_frac": _safe_float(latest_row.get("reversal_move_long_frac")),
            "reversal_move_short_frac": _safe_float(latest_row.get("reversal_move_short_frac")),
            "reversal_bar_long": _safe_bool(latest_row.get("reversal_bar_long")),
            "reversal_bar_short": _safe_bool(latest_row.get("reversal_bar_short")),
            "bullish_signal_bar": _safe_bool(latest_row.get("bullish_signal_bar")),
            "bearish_signal_bar": _safe_bool(latest_row.get("bearish_signal_bar")),
            "stabilization_confirmation_long": _safe_bool(latest_row.get("stabilization_confirmation_long")),
            "stabilization_confirmation_short": _safe_bool(latest_row.get("stabilization_confirmation_short")),
            "recent_price_change_frac": _safe_float(latest_row.get("recent_price_change_frac")),
            "extreme_downtrend": _safe_bool(latest_row.get("extreme_downtrend")),
            "extreme_uptrend": _safe_bool(latest_row.get("extreme_uptrend")),
            "trend_filter_pass_long": _safe_bool(latest_row.get("trend_filter_pass_long")),
            "trend_filter_pass_short": _safe_bool(latest_row.get("trend_filter_pass_short")),
            "trend_filter_blocked_long": _safe_bool(latest_row.get("trend_filter_blocked_long")),
            "trend_filter_blocked_short": _safe_bool(latest_row.get("trend_filter_blocked_short")),
            "regular_session_bar": _safe_bool(latest_row.get("regular_session_bar")),
            "time_window_active": _safe_bool(latest_row.get("time_window_active")),
            "entries_allowed": _safe_bool(latest_row.get("entries_allowed")),
            "long_core_setup": _safe_bool(latest_row.get("long_core_setup")),
            "short_core_setup": _safe_bool(latest_row.get("short_core_setup")),
            "long_setup": _safe_bool(latest_row.get("long_setup")),
            "short_setup": _safe_bool(latest_row.get("short_setup")),
            "long_entry_setup": _safe_bool(latest_row.get("long_entry_setup")),
            "short_entry_setup": _safe_bool(latest_row.get("short_entry_setup")),
            "long_entry_signal": _safe_bool(latest_row.get("long_entry_signal")),
            "short_entry_signal": _safe_bool(latest_row.get("short_entry_signal")),
            "long_stop_price": _safe_float(latest_row.get("long_stop_price")),
            "short_stop_price": _safe_float(latest_row.get("short_stop_price")),
            "long_take_profit_price": _safe_float(latest_row.get("long_take_profit_price")),
            "short_take_profit_price": _safe_float(latest_row.get("short_take_profit_price")),
            "long_stop_distance_frac_of_price": _safe_float(latest_row.get("long_stop_distance_frac_of_price")),
            "short_stop_distance_frac_of_price": _safe_float(latest_row.get("short_stop_distance_frac_of_price")),
            "long_entry_risk_plan_valid": _safe_bool(latest_row.get("long_entry_risk_plan_valid")),
            "short_entry_risk_plan_valid": _safe_bool(latest_row.get("short_entry_risk_plan_valid")),
            "long_entry_rejected_max_stop_distance": _safe_bool(
                latest_row.get("long_entry_rejected_max_stop_distance")
            ),
            "short_entry_rejected_max_stop_distance": _safe_bool(
                latest_row.get("short_entry_rejected_max_stop_distance")
            ),
            "long_entry_rejection_reason": None if pd.isna(long_rejection_reason) else str(long_rejection_reason),
            "short_entry_rejection_reason": None if pd.isna(short_rejection_reason) else str(short_rejection_reason),
            "entry_blocked_by_stop_filter": _safe_bool(latest_row.get("entry_blocked_by_stop_filter")),
            "entry_blocked_by_max_stop_distance": _safe_bool(
                latest_row.get("entry_blocked_by_max_stop_distance")
            ),
            "long_exit_signal": _safe_bool(latest_row.get("long_exit_signal")),
            "short_exit_signal": _safe_bool(latest_row.get("short_exit_signal")),
            "entry_action": self.entry_action(latest_row),
            "bars_in_trade": bars_in_trade,
            "time_stop_triggered": time_stop_triggered,
            "exit_action": resolved_exit_action,
            "exit_reason": None if resolved_exit_action is None else self.exit_reason(latest_row, position_context),
        }

    def calculate_backtest_metrics(self, df: pd.DataFrame) -> dict:
        return calculate_standard_backtest_metrics(df)
