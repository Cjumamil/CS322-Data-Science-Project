"""Fetches recent Alpaca market data used by the trading bot."""

import os
from datetime import datetime, timedelta, timezone

import pandas as pd
from alpaca.data.enums import DataFeed
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame, TimeFrameUnit


API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


def _create_data_client() -> StockHistoricalDataClient:
    """Create an Alpaca market data client from environment variables."""
    if not API_KEY or not SECRET_KEY:
        raise ValueError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables."
        )

    return StockHistoricalDataClient(API_KEY, SECRET_KEY)


def _parse_interval(interval: str) -> TimeFrame:
    """Convert a short interval string like 5m or 1d into an Alpaca timeframe."""
    interval = interval.strip().lower()

    if interval.endswith("m"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Minute)
    if interval.endswith("h"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Hour)
    if interval.endswith("d"):
        return TimeFrame(int(interval[:-1]), TimeFrameUnit.Day)

    raise ValueError(f"Unsupported interval format: {interval}")


def _interval_to_timedelta(interval: str) -> timedelta:
    """Convert the strategy interval string into a Python timedelta."""
    interval = interval.strip().lower()

    if interval.endswith("m"):
        return timedelta(minutes=int(interval[:-1]))
    if interval.endswith("h"):
        return timedelta(hours=int(interval[:-1]))
    if interval.endswith("d"):
        return timedelta(days=int(interval[:-1]))

    raise ValueError(f"Unsupported interval format: {interval}")


def _estimate_request_start(end: datetime, interval: str, lookback_bars: int) -> datetime:
    """Estimate a safe start time so Alpaca returns enough history for indicators.

    Intraday requests need extra room for overnight gaps, weekends, and holidays.
    Daily requests also use a wider window so non-trading days do not starve the bot.
    """
    bar_delta = _interval_to_timedelta(interval)
    interval = interval.strip().lower()

    if interval.endswith(("m", "h")):
        safety_multiplier = 12
    else:
        safety_multiplier = 3

    return end - (bar_delta * lookback_bars * safety_multiplier)


def download_data(symbol: str, interval: str, lookback_bars: int) -> pd.DataFrame:
    """Download recent OHLCV bars from Alpaca for one symbol.

    `interval` controls the bar size, while `lookback_bars` controls how
    much recent history we pull to compute indicators like SMA-20 and SMA-50.
    """
    end = datetime.now(timezone.utc)
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_parse_interval(interval),
        start=_estimate_request_start(end, interval, lookback_bars),
        end=end,
        # IEX is Alpaca's free stock feed. It is good enough for paper
        # testing here, but it can differ from broader consolidated feeds.
        feed=DataFeed.IEX,
    )

    bars = _create_data_client().get_stock_bars(request)
    df = bars.df.copy()

    if df.empty:
        return pd.DataFrame(columns=["Open", "High", "Low", "Close", "Volume"])

    if isinstance(df.index, pd.MultiIndex):
        try:
            df = df.xs(symbol, level="symbol")
        except (KeyError, ValueError):
            df = df.xs(symbol)

    df = df.rename(
        columns={
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )

    cleaned = df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
    if len(cleaned) > lookback_bars:
        cleaned = cleaned.tail(lookback_bars).copy()

    return cleaned
