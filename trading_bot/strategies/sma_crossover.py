"""Simple SMA crossover strategy implementation."""

from dataclasses import dataclass

import pandas as pd

from trading_bot.strategies.base import PositionContext, TradingStrategy, calculate_standard_backtest_metrics


@dataclass(frozen=True)
class SmaCrossoverStrategy(TradingStrategy):
    """Trade when a fast SMA crosses a slower SMA."""

    fast_window: int
    slow_window: int
    interval: str
    lookback_bars: int
    name: str = "sma_crossover"
    version: str = "v1"

    def min_required_bars(self) -> int:
        return max(self.fast_window, self.slow_window) + 1

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add SMA signals and performance helper columns to price data."""
        work = df.copy()

        work["SMA_FAST"] = work["Close"].rolling(self.fast_window).mean()
        work["SMA_SLOW"] = work["Close"].rolling(self.slow_window).mean()

        work = work.dropna(subset=["SMA_FAST", "SMA_SLOW"]).copy()

        work["signal"] = 0
        work.loc[work["SMA_FAST"] > work["SMA_SLOW"], "signal"] = 1

        work["crossover"] = work["signal"].diff()
        work["daily_return"] = work["Close"].pct_change()
        work["strategy_return"] = work["signal"].shift(1) * work["daily_return"]
        work["predicted_close"] = work["Close"].shift(1)
        work["abs_error"] = (work["Close"] - work["predicted_close"]).abs()

        return work

    def recent_display_columns(self) -> list[str]:
        return ["Close", "SMA_FAST", "SMA_SLOW", "signal", "crossover"]

    def latest_signal_label(self) -> str:
        return "Latest crossover"

    def latest_signal_value(self, latest_row: pd.Series):
        return float(latest_row["crossover"])

    def event_key(self, latest_row: pd.Series):
        return float(latest_row["crossover"])

    def entry_action(self, latest_row: pd.Series) -> str | None:
        if self.should_enter_long(latest_row):
            return "BUY"
        if self.should_enter_short(latest_row):
            return "SELL"
        return None

    def exit_action(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> str | None:
        position_side = "flat" if position_context is None else position_context.side
        if position_side == "long" and self.should_exit_long(latest_row, position_context):
            return "SELL"
        if position_side == "short" and self.should_exit_short(latest_row, position_context):
            return "BUY"
        return None

    def should_enter_long(self, latest_row: pd.Series) -> bool:
        return float(latest_row["crossover"]) == 1.0

    def should_enter_short(self, latest_row: pd.Series) -> bool:
        return False

    def should_exit_long(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> bool:
        return float(latest_row["crossover"]) == -1.0

    def should_exit_short(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> bool:
        return False

    def entry_reason(self) -> str:
        return self.name

    def exit_reason(self, latest_row: pd.Series | None = None, position_context: PositionContext | None = None) -> str:
        return self.name

    def build_entry_risk_plan(self, latest_row: pd.Series, position_side: str, risk_settings) -> dict | None:
        return None

    def risk_reward_multiple(self) -> float | None:
        return None

    def build_strategy_parameters(self) -> dict:
        return {
            "fast_window": self.fast_window,
            "slow_window": self.slow_window,
        }

    def build_strategy_signals(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> dict:
        return {
            "sma_fast": float(latest_row["SMA_FAST"]),
            "sma_slow": float(latest_row["SMA_SLOW"]),
            "signal": int(latest_row["signal"]),
            "crossover": float(latest_row["crossover"]),
        }

    def calculate_backtest_metrics(self, df: pd.DataFrame) -> dict:
        return calculate_standard_backtest_metrics(df)
