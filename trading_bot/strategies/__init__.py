"""Strategy implementations and shared strategy interfaces."""

from copy import deepcopy

from trading_bot.strategies.macd_pullback import MacdPullbackStrategy
from trading_bot.strategies.sma_crossover import SmaCrossoverStrategy


DEFAULT_STRATEGY_CONFIGS = {
    "sma_crossover": {
        "name": "sma_crossover",
        "version": "v1",
        "interval": "1m",
        "lookback_bars": 1000,
        "fast_window": 8,
        "slow_window": 20,
    },
    "macd_pullback": {
        "name": "macd_pullback",
        "version": "v1",
        "interval": "5m",
        "lookback_bars": 500,
        "macd_fast_window": 12,
        "macd_slow_window": 26,
        "macd_signal_window": 9,
        "trend_ema_window": 200,
        "opening_no_trade_bars": 6,
        "ema_slope_lookback": 12,
        "recent_range_lookback": 12,
        "ema_slope_threshold": 0.08,
        "min_recent_range_frac_of_price": 0.003,
        "macd_near_zero_lookback": 20,
        "sideways_macd_near_zero_threshold": 0.20,
        "pullback_reset_zone_threshold": 0.35,
        "pullback_memory_bars": 8,
        "ema_stop_buffer_pct": 0.001,
        "take_profit_risk_multiple": 1.5,
        "max_stop_distance_frac_of_price": 0.04,
        "enable_histogram_entry_confirmation": True,
        "enable_time_stop": True,
        "max_bars_in_trade": 18,
        "enable_macd_failure_exit": True,
        "min_bars_before_macd_exit": 6,
    },
}


def list_supported_strategies() -> list[str]:
    """Return the supported built-in strategy names."""
    return sorted(DEFAULT_STRATEGY_CONFIGS)


def default_strategy_config(name: str) -> dict:
    """Return a copy of the default configuration for one strategy."""
    if name not in DEFAULT_STRATEGY_CONFIGS:
        raise ValueError(f"Unsupported strategy in config: {name}")
    return deepcopy(DEFAULT_STRATEGY_CONFIGS[name])


def create_strategy(strategy_config: dict):
    """Build one strategy instance from config data."""
    name = strategy_config["name"]

    if name == "sma_crossover":
        return SmaCrossoverStrategy(
            fast_window=int(strategy_config["fast_window"]),
            slow_window=int(strategy_config["slow_window"]),
            interval=str(strategy_config["interval"]),
            lookback_bars=int(strategy_config["lookback_bars"]),
            version=str(strategy_config.get("version", "v1")),
        )

    if name == "macd_pullback":
        return MacdPullbackStrategy(
            interval=str(strategy_config["interval"]),
            lookback_bars=int(strategy_config["lookback_bars"]),
            macd_fast_window=int(strategy_config["macd_fast_window"]),
            macd_slow_window=int(strategy_config["macd_slow_window"]),
            macd_signal_window=int(strategy_config["macd_signal_window"]),
            trend_ema_window=int(strategy_config["trend_ema_window"]),
            opening_no_trade_bars=int(strategy_config["opening_no_trade_bars"]),
            ema_slope_lookback=int(strategy_config["ema_slope_lookback"]),
            recent_range_lookback=int(strategy_config["recent_range_lookback"]),
            ema_slope_threshold=float(strategy_config["ema_slope_threshold"]),
            min_recent_range_frac_of_price=float(strategy_config["min_recent_range_frac_of_price"]),
            macd_near_zero_lookback=int(strategy_config["macd_near_zero_lookback"]),
            sideways_macd_near_zero_threshold=float(strategy_config["sideways_macd_near_zero_threshold"]),
            pullback_reset_zone_threshold=float(strategy_config["pullback_reset_zone_threshold"]),
            pullback_memory_bars=int(strategy_config["pullback_memory_bars"]),
            ema_stop_buffer_pct=float(strategy_config["ema_stop_buffer_pct"]),
            take_profit_risk_multiple=float(strategy_config["take_profit_risk_multiple"]),
            max_stop_distance_frac_of_price=(
                None
                if strategy_config.get("max_stop_distance_frac_of_price") in {None, ""}
                else float(strategy_config["max_stop_distance_frac_of_price"])
            ),
            enable_histogram_entry_confirmation=bool(
                strategy_config.get("enable_histogram_entry_confirmation", True)
            ),
            enable_time_stop=bool(strategy_config.get("enable_time_stop", True)),
            max_bars_in_trade=int(strategy_config.get("max_bars_in_trade", 18)),
            enable_macd_failure_exit=bool(strategy_config.get("enable_macd_failure_exit", True)),
            min_bars_before_macd_exit=int(strategy_config.get("min_bars_before_macd_exit", 4)),
            version=str(strategy_config.get("version", "v1")),
        )

    raise ValueError(f"Unsupported strategy in config: {name}")


__all__ = [
    "MacdPullbackStrategy",
    "SmaCrossoverStrategy",
    "create_strategy",
    "default_strategy_config",
    "list_supported_strategies",
]
