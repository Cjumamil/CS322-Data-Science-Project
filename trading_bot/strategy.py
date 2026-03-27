"""Builds strategy signals and simple backtest metrics for the bot."""

import pandas as pd


def build_strategy_dataframe(
    df: pd.DataFrame,
    fast_window: int,
    slow_window: int,
) -> pd.DataFrame:
    """Add SMA signals and performance helper columns to price data."""
    df = df.copy()

    df["SMA_FAST"] = df["Close"].rolling(fast_window).mean()
    df["SMA_SLOW"] = df["Close"].rolling(slow_window).mean()

    df = df.dropna(subset=["SMA_FAST", "SMA_SLOW"]).copy()

    df["signal"] = 0
    df.loc[df["SMA_FAST"] > df["SMA_SLOW"], "signal"] = 1

    df["crossover"] = df["signal"].diff()
    df["daily_return"] = df["Close"].pct_change()
    # Shift the signal forward one bar so the backtest does not use
    # today's signal to trade on today's close.
    df["strategy_return"] = df["signal"].shift(1) * df["daily_return"]

    # This is a simple baseline forecast metric, not part of order logic.
    df["predicted_close"] = df["Close"].shift(1)
    df["abs_error"] = (df["Close"] - df["predicted_close"]).abs()

    return df


def calculate_backtest_metrics(df: pd.DataFrame) -> dict:
    """Summarize how the strategy performed on the prepared DataFrame."""
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
