"""Strategy implementations and shared strategy interfaces."""

from trading_bot.strategies.sma_crossover import SmaCrossoverStrategy


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

    raise ValueError(f"Unsupported strategy in config: {name}")


__all__ = ["SmaCrossoverStrategy", "create_strategy"]
