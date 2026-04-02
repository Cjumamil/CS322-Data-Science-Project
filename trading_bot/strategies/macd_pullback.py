"""MACD pullback strategy with EMA-200 trend and sideways filtering."""

from dataclasses import dataclass

import pandas as pd

from trading_bot.risk import assess_stop_distance
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


@dataclass(frozen=True)
class MacdPullbackStrategy(TradingStrategy):
    """Trend-following MACD pullback entries with EMA-200 trend confirmation."""

    interval: str
    lookback_bars: int
    macd_fast_window: int
    macd_slow_window: int
    macd_signal_window: int
    trend_ema_window: int
    opening_no_trade_bars: int
    ema_slope_lookback: int
    recent_range_lookback: int
    ema_slope_threshold: float
    min_recent_range_frac_of_price: float
    macd_near_zero_lookback: int
    sideways_macd_near_zero_threshold: float
    pullback_reset_zone_threshold: float
    pullback_memory_bars: int
    ema_stop_buffer_pct: float
    take_profit_risk_multiple: float
    max_stop_distance_frac_of_price: float | None
    enable_histogram_entry_confirmation: bool
    enable_time_stop: bool
    max_bars_in_trade: int
    enable_macd_failure_exit: bool
    min_bars_before_macd_exit: int
    name: str = "macd_pullback"
    version: str = "v1"

    def min_required_bars(self) -> int:
        return max(
            self.trend_ema_window + self.ema_slope_lookback,
            self.macd_slow_window + self.macd_signal_window,
            self.recent_range_lookback,
            self.macd_near_zero_lookback,
            self.pullback_memory_bars,
            self.opening_no_trade_bars,
        ) + 2

    def _interval_minutes(self) -> int:
        interval = self.interval.strip().lower()
        if not interval.endswith("m"):
            raise ValueError(f"{self.name} currently expects an intraday minute interval, got: {self.interval}")
        return int(interval[:-1])

    def _bars_in_trade_exceeded(self, position_context: PositionContext | None) -> bool:
        """Exit stagnant trades once they overstay the configured holding window."""
        if not self.enable_time_stop or position_context is None or position_context.bars_in_trade is None:
            return False
        return position_context.bars_in_trade >= self.max_bars_in_trade

    def _macd_failure_triggered(self, latest_row: pd.Series, position_context: PositionContext | None) -> bool:
        """Exit when momentum flips back against the open trade direction."""
        if not self.enable_macd_failure_exit or position_context is None:
            return False
        if position_context.bars_in_trade is None or position_context.bars_in_trade < self.min_bars_before_macd_exit:
            return False
        if position_context.side == "long":
            return _safe_bool(latest_row.get("macd_failure_long_signal"))
        if position_context.side == "short":
            return _safe_bool(latest_row.get("macd_failure_short_signal"))
        return False

    def _build_entry_risk_snapshot(
        self,
        *,
        entry_price: float | None,
        ema200: float | None,
        position_side: str,
    ) -> dict:
        """Build one preview or execution risk snapshot for a candidate entry."""
        snapshot = {
            "position_side": position_side,
            "stop_price": None,
            "stop_distance": None,
            "stop_distance_frac_of_price": None,
            "max_stop_distance_frac_of_price": self.max_stop_distance_frac_of_price,
            "rejected_due_to_max_stop_distance": False,
            "is_valid": False,
            "rejection_reason": None,
            "risk_reward_multiple": self.take_profit_risk_multiple,
            "stop_source": "ema200_buffer",
            "ema200": ema200,
            "ema_stop_buffer_pct": self.ema_stop_buffer_pct,
        }

        if ema200 is None or position_side not in {"long", "short"}:
            snapshot["rejection_reason"] = "invalid_risk_context"
            return snapshot

        if position_side == "long":
            stop_price = round(ema200 * (1 - self.ema_stop_buffer_pct), 2)
        else:
            stop_price = round(ema200 * (1 + self.ema_stop_buffer_pct), 2)

        validation = assess_stop_distance(
            entry_price=entry_price,
            stop_price=stop_price,
            position_side=position_side,
            max_stop_distance_frac_of_price=self.max_stop_distance_frac_of_price,
        )

        snapshot.update(validation)
        snapshot["stop_source"] = "ema200_buffer"
        snapshot["risk_reward_multiple"] = self.take_profit_risk_multiple
        snapshot["ema200"] = ema200
        snapshot["ema_stop_buffer_pct"] = self.ema_stop_buffer_pct
        return snapshot

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add MACD pullback indicators, filters, and helper columns."""
        work = df.copy().sort_index()

        work["EMA200"] = work["Close"].ewm(span=self.trend_ema_window, adjust=False).mean()
        work["MACD"] = (
            work["Close"].ewm(span=self.macd_fast_window, adjust=False).mean()
            - work["Close"].ewm(span=self.macd_slow_window, adjust=False).mean()
        )
        work["signal_line"] = work["MACD"].ewm(span=self.macd_signal_window, adjust=False).mean()
        work["histogram"] = work["MACD"] - work["signal_line"]
        work["histogram_previous"] = work["histogram"].shift(1)

        work["ema_slope"] = work["EMA200"] - work["EMA200"].shift(self.ema_slope_lookback)
        rolling_high = work["High"].rolling(self.recent_range_lookback).max()
        rolling_low = work["Low"].rolling(self.recent_range_lookback).min()
        work["recent_range"] = rolling_high - rolling_low
        work["recent_range_frac_of_price"] = work["recent_range"] / work["Close"]

        work["normalized_ema_slope"] = (
            work["ema_slope"].abs() / work["recent_range"].where(work["recent_range"] > 0)
        )
        work["recent_max_abs_macd"] = work["MACD"].abs().rolling(
            self.macd_near_zero_lookback,
            min_periods=self.macd_near_zero_lookback,
        ).max()

        work["recent_range_too_small"] = (
            work["recent_range_frac_of_price"] < self.min_recent_range_frac_of_price
        ).fillna(True)
        work["macd_near_zero"] = (
            work["MACD"].abs()
            <= (self.sideways_macd_near_zero_threshold * work["recent_max_abs_macd"])
        ).fillna(False)
        work["sideways_market"] = (
            work["recent_range_too_small"]
            | (
                (work["normalized_ema_slope"] < self.ema_slope_threshold).fillna(True)
                & work["macd_near_zero"]
            )
        )
        work["trend_filter_tradable"] = ~work["recent_range_too_small"]

        work["in_reset_zone_now"] = (
            work["MACD"].abs()
            <= (self.pullback_reset_zone_threshold * work["recent_max_abs_macd"])
        ).fillna(False)
        work["had_recent_reset"] = (
            work["in_reset_zone_now"].astype(int).rolling(self.pullback_memory_bars, min_periods=1).max() > 0
        )

        previous_macd = work["MACD"].shift(1)
        previous_signal = work["signal_line"].shift(1)
        work["macd_cross_above_signal"] = (
            (work["MACD"] > work["signal_line"]) & (previous_macd <= previous_signal)
        ).fillna(False)
        work["macd_cross_below_signal"] = (
            (work["MACD"] < work["signal_line"]) & (previous_macd >= previous_signal)
        ).fillna(False)
        work["macd_failure_long_signal"] = work["macd_cross_below_signal"]
        work["macd_failure_short_signal"] = work["macd_cross_above_signal"]

        # Require histogram expansion in the trade direction to avoid weak crossovers.
        work["histogram_long_confirmed"] = (
            (work["histogram"] > 0) & (work["histogram"] > work["histogram_previous"])
        ).fillna(False)
        work["histogram_short_confirmed"] = (
            (work["histogram"] < 0) & (work["histogram"] < work["histogram_previous"])
        ).fillna(False)
        work["price_above_ema200"] = work["Close"] > work["EMA200"]
        work["price_below_ema200"] = work["Close"] < work["EMA200"]
        work["trend_direction"] = "flat"
        work.loc[work["price_above_ema200"], "trend_direction"] = "above_ema200"
        work.loc[work["price_below_ema200"], "trend_direction"] = "below_ema200"
        work["candidate_position_side"] = "flat"
        work.loc[work["price_above_ema200"], "candidate_position_side"] = "long"
        work.loc[work["price_below_ema200"], "candidate_position_side"] = "short"
        if self.enable_histogram_entry_confirmation:
            work["histogram_entry_confirmed"] = (
                work["price_above_ema200"] & work["histogram_long_confirmed"]
            ) | (
                work["price_below_ema200"] & work["histogram_short_confirmed"]
            )
        else:
            work["histogram_entry_confirmed"] = True

        risk_snapshots = []
        for _, row in work.iterrows():
            risk_snapshots.append(
                self._build_entry_risk_snapshot(
                    entry_price=_safe_float(row["Close"]),
                    ema200=_safe_float(row["EMA200"]),
                    position_side=str(row["candidate_position_side"]),
                )
            )
        risk_snapshot_df = pd.DataFrame(risk_snapshots, index=work.index)
        work["stop_price"] = risk_snapshot_df["stop_price"]
        work["stop_distance"] = risk_snapshot_df["stop_distance"]
        work["stop_distance_frac_of_price"] = risk_snapshot_df["stop_distance_frac_of_price"]
        work["entry_risk_plan_valid"] = risk_snapshot_df["is_valid"].fillna(False)
        work["entry_rejected_max_stop_distance"] = risk_snapshot_df["rejected_due_to_max_stop_distance"].fillna(False)
        work["entry_rejection_reason"] = risk_snapshot_df["rejection_reason"]

        localized_index = work.index
        if getattr(localized_index, "tz", None) is None:
            localized_index = localized_index.tz_localize("UTC")
        localized_index = localized_index.tz_convert(NEW_YORK_TIMEZONE)

        minutes_of_day = (localized_index.hour * 60) + localized_index.minute
        opening_window_minutes = self.opening_no_trade_bars * self._interval_minutes()
        work["regular_session_bar"] = (
            (minutes_of_day >= REGULAR_MARKET_OPEN_MINUTE)
            & (minutes_of_day < REGULAR_MARKET_CLOSE_MINUTE)
        )
        work["opening_no_trade_window"] = (
            work["regular_session_bar"]
            & (minutes_of_day < (REGULAR_MARKET_OPEN_MINUTE + opening_window_minutes))
        )
        work["entries_allowed"] = (
            work["regular_session_bar"]
            & ~work["opening_no_trade_window"]
            & work["trend_filter_tradable"]
            & ~work["sideways_market"]
        )

        work["long_entry_setup"] = (
            work["entries_allowed"]
            & work["price_above_ema200"]
            & work["had_recent_reset"]
            & work["macd_cross_above_signal"]
            & work["histogram_entry_confirmed"]
        )
        work["short_entry_setup"] = (
            work["entries_allowed"]
            & work["price_below_ema200"]
            & work["had_recent_reset"]
            & work["macd_cross_below_signal"]
            & work["histogram_entry_confirmed"]
        )
        work["long_entry_signal"] = work["long_entry_setup"] & work["entry_risk_plan_valid"]
        work["short_entry_signal"] = work["short_entry_setup"] & work["entry_risk_plan_valid"]
        work["entry_blocked_by_stop_filter"] = (
            (work["long_entry_setup"] | work["short_entry_setup"])
            & ~work["entry_risk_plan_valid"]
        )
        work["entry_blocked_by_max_stop_distance"] = (
            work["entry_blocked_by_stop_filter"] & work["entry_rejected_max_stop_distance"]
        )
        work["long_exit_signal"] = work["macd_failure_long_signal"] if self.enable_macd_failure_exit else False
        work["short_exit_signal"] = work["macd_failure_short_signal"] if self.enable_macd_failure_exit else False

        work["entry_action"] = pd.Series(index=work.index, dtype="object")
        work.loc[work["long_entry_signal"], "entry_action"] = "BUY"
        work.loc[work["short_entry_signal"], "entry_action"] = "SELL"
        work["exit_action"] = pd.Series(index=work.index, dtype="object")

        exposure_signal = pd.Series(pd.NA, index=work.index, dtype="object")
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

        required_columns = [
            "EMA200",
            "MACD",
            "signal_line",
            "histogram",
            "recent_range",
            "normalized_ema_slope",
            "recent_max_abs_macd",
            "stop_price",
        ]
        work = work.dropna(subset=required_columns).copy()

        return work

    def recent_display_columns(self) -> list[str]:
        return [
            "Close",
            "EMA200",
            "normalized_ema_slope",
            "recent_range",
            "MACD",
            "signal_line",
            "histogram",
            "histogram_previous",
            "stop_distance_frac_of_price",
            "entry_rejected_max_stop_distance",
            "had_recent_reset",
            "sideways_market",
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
        return self._macd_failure_triggered(latest_row, position_context) or self._bars_in_trade_exceeded(position_context)

    def should_exit_short(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> bool:
        return self._macd_failure_triggered(latest_row, position_context) or self._bars_in_trade_exceeded(position_context)

    def entry_reason(self) -> str:
        return self.name

    def exit_reason(
        self,
        latest_row: pd.Series | None = None,
        position_context: PositionContext | None = None,
    ) -> str:
        if latest_row is not None and self._macd_failure_triggered(latest_row, position_context):
            return "macd_failure"
        if self._bars_in_trade_exceeded(position_context):
            return "time_stop"
        return self.name

    def build_entry_risk_plan(self, latest_row: pd.Series, position_side: str, risk_settings) -> dict | None:
        return self._build_entry_risk_snapshot(
            entry_price=_safe_float(latest_row.get("Close")),
            ema200=_safe_float(latest_row.get("EMA200")),
            position_side=position_side,
        )

    def risk_reward_multiple(self) -> float | None:
        return self.take_profit_risk_multiple

    def build_strategy_parameters(self) -> dict:
        return {
            "stop_model": "ema200_buffer",
            "take_profit_model": "risk_reward_multiple",
            "macd_fast_window": self.macd_fast_window,
            "macd_slow_window": self.macd_slow_window,
            "macd_signal_window": self.macd_signal_window,
            "enable_histogram_entry_confirmation": self.enable_histogram_entry_confirmation,
            "trend_ema_window": self.trend_ema_window,
            "opening_no_trade_bars": self.opening_no_trade_bars,
            "ema_slope_lookback": self.ema_slope_lookback,
            "recent_range_lookback": self.recent_range_lookback,
            "ema_slope_threshold": self.ema_slope_threshold,
            "min_recent_range_frac_of_price": self.min_recent_range_frac_of_price,
            "macd_near_zero_lookback": self.macd_near_zero_lookback,
            "sideways_macd_near_zero_threshold": self.sideways_macd_near_zero_threshold,
            "pullback_reset_zone_threshold": self.pullback_reset_zone_threshold,
            "pullback_memory_bars": self.pullback_memory_bars,
            "ema_stop_buffer_pct": self.ema_stop_buffer_pct,
            "take_profit_risk_multiple": self.take_profit_risk_multiple,
            "max_stop_distance_frac_of_price": self.max_stop_distance_frac_of_price,
            "enable_time_stop": self.enable_time_stop,
            "max_bars_in_trade": self.max_bars_in_trade,
            "enable_macd_failure_exit": self.enable_macd_failure_exit,
            "min_bars_before_macd_exit": self.min_bars_before_macd_exit,
        }

    def build_strategy_signals(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> dict:
        bars_in_trade = None if position_context is None else position_context.bars_in_trade
        macd_failure_exit_armed = (
            self.enable_macd_failure_exit
            and bars_in_trade is not None
            and bars_in_trade >= self.min_bars_before_macd_exit
        )
        macd_failure_triggered = self._macd_failure_triggered(latest_row, position_context)
        time_stop_triggered = self._bars_in_trade_exceeded(position_context)
        resolved_exit_action = self.exit_action(latest_row, position_context)
        return {
            "ema200": _safe_float(latest_row["EMA200"]),
            "normalized_ema_slope": _safe_float(latest_row["normalized_ema_slope"]),
            "recent_range": _safe_float(latest_row["recent_range"]),
            "recent_range_frac_of_price": _safe_float(latest_row["recent_range_frac_of_price"]),
            "macd": _safe_float(latest_row["MACD"]),
            "signal_line": _safe_float(latest_row["signal_line"]),
            "histogram": _safe_float(latest_row["histogram"]),
            "histogram_previous": _safe_float(latest_row["histogram_previous"]),
            "histogram_current": _safe_float(latest_row["histogram"]),
            "histogram_long_confirmed": _safe_bool(latest_row["histogram_long_confirmed"]),
            "histogram_short_confirmed": _safe_bool(latest_row["histogram_short_confirmed"]),
            "histogram_entry_confirmed": _safe_bool(latest_row["histogram_entry_confirmed"]),
            "recent_max_abs_macd": _safe_float(latest_row["recent_max_abs_macd"]),
            "in_reset_zone_now": _safe_bool(latest_row["in_reset_zone_now"]),
            "had_recent_reset": _safe_bool(latest_row["had_recent_reset"]),
            "sideways_market": _safe_bool(latest_row["sideways_market"]),
            "trend_direction": str(latest_row["trend_direction"]),
            "trend_filter_tradable": _safe_bool(latest_row["trend_filter_tradable"]),
            "entries_allowed": _safe_bool(latest_row["entries_allowed"]),
            "candidate_position_side": str(latest_row["candidate_position_side"]),
            "stop_price": _safe_float(latest_row["stop_price"]),
            "stop_distance": _safe_float(latest_row["stop_distance"]),
            "stop_distance_frac_of_price": _safe_float(latest_row["stop_distance_frac_of_price"]),
            "max_stop_distance_frac_of_price": self.max_stop_distance_frac_of_price,
            "entry_risk_plan_valid": _safe_bool(latest_row["entry_risk_plan_valid"]),
            "entry_rejected_max_stop_distance": _safe_bool(latest_row["entry_rejected_max_stop_distance"]),
            "entry_rejection_reason": None
            if pd.isna(latest_row["entry_rejection_reason"])
            else str(latest_row["entry_rejection_reason"]),
            "long_entry_setup": _safe_bool(latest_row["long_entry_setup"]),
            "short_entry_setup": _safe_bool(latest_row["short_entry_setup"]),
            "entry_blocked_by_stop_filter": _safe_bool(latest_row["entry_blocked_by_stop_filter"]),
            "entry_blocked_by_max_stop_distance": _safe_bool(latest_row["entry_blocked_by_max_stop_distance"]),
            "opening_no_trade_window": _safe_bool(latest_row["opening_no_trade_window"]),
            "macd_failure_long_signal": _safe_bool(latest_row["macd_failure_long_signal"]),
            "macd_failure_short_signal": _safe_bool(latest_row["macd_failure_short_signal"]),
            "min_bars_before_macd_exit": self.min_bars_before_macd_exit,
            "macd_failure_exit_armed": macd_failure_exit_armed,
            "long_entry_signal": _safe_bool(latest_row["long_entry_signal"]),
            "short_entry_signal": _safe_bool(latest_row["short_entry_signal"]),
            "long_exit_signal": _safe_bool(latest_row["long_exit_signal"]),
            "short_exit_signal": _safe_bool(latest_row["short_exit_signal"]),
            "entry_action": self.entry_action(latest_row),
            "bars_in_trade": bars_in_trade,
            "macd_failure_triggered": macd_failure_triggered,
            "time_stop_triggered": time_stop_triggered,
            "exit_action": resolved_exit_action,
            "exit_reason": None if resolved_exit_action is None else self.exit_reason(latest_row, position_context),
        }

    def calculate_backtest_metrics(self, df: pd.DataFrame) -> dict:
        return calculate_standard_backtest_metrics(df)
