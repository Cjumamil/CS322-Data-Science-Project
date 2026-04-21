"""MACD pullback strategy with a saved XGBoost model filtering entries."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from trading_bot.strategies.base import PositionContext, TradingStrategy
from trading_bot.strategies.macd_pullback import MacdPullbackStrategy
from trading_bot.xgboost_filter import apply_xgboost_entry_filter


@dataclass(frozen=True)
class MacdPullbackXgboostFilterStrategy(TradingStrategy):
    """Wrap the current MACD pullback strategy with an XGBoost entry gate."""

    base_strategy: MacdPullbackStrategy
    xgb_model_path: str
    xgb_probability_threshold: float
    symbol: str | None = None
    name: str = "macd_pullback_xgboost_filter"
    version: str = "v1"

    @property
    def interval(self) -> str:
        return self.base_strategy.interval

    @property
    def lookback_bars(self) -> int:
        return self.base_strategy.lookback_bars

    def min_required_bars(self) -> int:
        return self.base_strategy.min_required_bars()

    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare the base MACD pullback frame, then drop low-quality entries."""
        work = self.base_strategy.prepare_dataframe(df)
        if not self.xgb_model_path:
            raise ValueError("macd_pullback_xgboost_filter requires xgb_model_path in the strategy config.")
        if self.symbol:
            work["symbol"] = str(self.symbol).upper()
        return apply_xgboost_entry_filter(work, self.xgb_model_path, float(self.xgb_probability_threshold))

    def recent_display_columns(self) -> list[str]:
        return [
            *self.base_strategy.recent_display_columns(),
            "base_entry_action",
            "xgb_trade_quality_prob",
            "xgb_filter_pass",
            "entry_blocked_by_xgb_filter",
        ]

    def latest_signal_label(self) -> str:
        return "Latest filtered entry action"

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
        return self.base_strategy.exit_action(latest_row, position_context)

    def should_enter_long(self, latest_row: pd.Series) -> bool:
        return self.entry_action(latest_row) == "BUY"

    def should_enter_short(self, latest_row: pd.Series) -> bool:
        return self.entry_action(latest_row) == "SELL"

    def should_exit_long(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> bool:
        return self.base_strategy.should_exit_long(latest_row, position_context)

    def should_exit_short(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> bool:
        return self.base_strategy.should_exit_short(latest_row, position_context)

    def entry_reason(self) -> str:
        return self.name

    def exit_reason(
        self,
        latest_row: pd.Series | None = None,
        position_context: PositionContext | None = None,
    ) -> str:
        return self.base_strategy.exit_reason(latest_row, position_context)

    def build_entry_risk_plan(self, latest_row: pd.Series, position_side: str, risk_settings) -> dict | None:
        return self.base_strategy.build_entry_risk_plan(latest_row, position_side, risk_settings)

    def risk_reward_multiple(self) -> float | None:
        return self.base_strategy.risk_reward_multiple()

    def build_strategy_parameters(self) -> dict:
        return {
            **self.base_strategy.build_strategy_parameters(),
            "xgb_model_path": self.xgb_model_path,
            "xgb_probability_threshold": float(self.xgb_probability_threshold),
            "entry_filter_model": "xgboost_binary_classifier",
            "symbol": None if self.symbol is None else str(self.symbol).upper(),
        }

    def build_strategy_signals(self, latest_row: pd.Series, position_context: PositionContext | None = None) -> dict:
        signals = self.base_strategy.build_strategy_signals(latest_row, position_context)
        base_entry_action = latest_row.get("base_entry_action")
        return {
            **signals,
            "base_entry_action": None if pd.isna(base_entry_action) else str(base_entry_action),
            "xgb_trade_quality_prob": None
            if pd.isna(latest_row.get("xgb_trade_quality_prob"))
            else float(latest_row["xgb_trade_quality_prob"]),
            "xgb_probability_threshold": float(latest_row.get("xgb_probability_threshold", self.xgb_probability_threshold)),
            "xgb_filter_pass": bool(latest_row.get("xgb_filter_pass", False)),
            "entry_blocked_by_xgb_filter": bool(latest_row.get("entry_blocked_by_xgb_filter", False)),
            "entry_action": self.entry_action(latest_row),
        }

    def calculate_backtest_metrics(self, df: pd.DataFrame) -> dict:
        return self.base_strategy.calculate_backtest_metrics(df)
