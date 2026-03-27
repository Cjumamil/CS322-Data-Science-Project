"""Fetches recent Alpaca market data used by the trading bot."""

import os
from datetime import datetime, timezone

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


def download_data(symbol: str, interval: str, lookback_bars: int) -> pd.DataFrame:
    """Download recent OHLCV bars from Alpaca for one symbol.

    `interval` controls the bar size, while `lookback_bars` controls how
    much recent history we pull to compute indicators like SMA-20 and SMA-50.
    """
    request = StockBarsRequest(
        symbol_or_symbols=symbol,
        timeframe=_parse_interval(interval),
        end=datetime.now(timezone.utc),
        limit=lookback_bars,
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

    return df[["Open", "High", "Low", "Close", "Volume"]].dropna().copy()
