"""Shared strategy interfaces and generic backtest helpers."""

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class TradingStrategy(Protocol):
    """Small interface for pluggable trading strategies."""

    name: str
    version: str
    interval: str
    lookback_bars: int

    def min_required_bars(self) -> int: ...
    def prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame: ...
    def recent_display_columns(self) -> list[str]: ...
    def latest_signal_label(self) -> str: ...
    def latest_signal_value(self, latest_row: pd.Series): ...
    def event_key(self, latest_row: pd.Series): ...
    def should_enter_long(self, latest_row: pd.Series) -> bool: ...
    def should_exit_long(self, latest_row: pd.Series) -> bool: ...
    def entry_reason(self) -> str: ...
    def exit_reason(self) -> str: ...
    def build_strategy_parameters(self) -> dict: ...
    def build_strategy_signals(self, latest_row: pd.Series) -> dict: ...
    def calculate_backtest_metrics(self, df: pd.DataFrame) -> dict: ...


@dataclass(frozen=True)
class StrategyMetadata:
    """Stable metadata describing one active strategy configuration."""

    name: str
    version: str
    interval: str


def calculate_standard_backtest_metrics(df: pd.DataFrame) -> dict:
    """Summarize strategy performance on a prepared DataFrame.

    Strategies can reuse this helper as long as they produce the common
    columns used by the current student project:
    `daily_return`, `strategy_return`, `crossover`, and `abs_error`.
    """
    work = df.copy()

    if work.empty:
        return {}

    work["daily_return"] = work["daily_return"].fillna(0)
    work["strategy_return"] = work["strategy_return"].fillna(0)

    cumulative_strategy = (1 + work["strategy_return"]).cumprod()
    cumulative_buy_hold = (1 + work["daily_return"]).cumprod()

    strategy_return_pct = (cumulative_strategy.iloc[-1] - 1) * 100
    buy_hold_return_pct = (cumulative_buy_hold.iloc[-1] - 1) * 100

    strategy_std = work["strategy_return"].std()
    if strategy_std and strategy_std != 0:
        sharpe_ratio = (work["strategy_return"].mean() / strategy_std) * (252 ** 0.5)
    else:
        sharpe_ratio = 0.0

    running_max = cumulative_strategy.cummax()
    drawdown = (cumulative_strategy / running_max) - 1
    max_drawdown_pct = drawdown.min() * 100

    buy_signals = int((work["crossover"] == 1).sum())
    sell_signals = int((work["crossover"] == -1).sum())
    mae_close_prediction = work["abs_error"].dropna().mean()

    return {
        "strategy_return_pct": strategy_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "mae_close_prediction": mae_close_prediction,
    }
