"""Loads bot settings and per-symbol strategy assignments from TOML."""

from typing import Any
from dataclasses import dataclass
from pathlib import Path
import tomllib

from trading_bot.strategies import create_strategy


@dataclass(frozen=True)
class BotSettings:
    version: str
    run_continuously: bool
    poll_interval_seconds: int
    paper_trading: bool


@dataclass(frozen=True)
class ExecutionSettings:
    order_status_poll_seconds: int
    order_status_timeout_seconds: int
    enable_broker_side_stop_loss: bool
    force_test_trade: bool
    force_direction: str


@dataclass(frozen=True)
class RiskSettings:
    max_position_qty: int
    risk_fraction_of_buying_power: float
    stop_loss_pct: float
    take_profit_pct: float


@dataclass(frozen=True)
class SymbolAssignment:
    ticker: str
    strategy: Any


@dataclass(frozen=True)
class BotConfig:
    bot: BotSettings
    execution: ExecutionSettings
    risk: RiskSettings
    symbols: list[SymbolAssignment]


def load_config(path: str = "bot_config.toml") -> BotConfig:
    """Load the bot configuration from a TOML file."""
    config_path = Path(path)
    with config_path.open("rb") as config_file:
        raw = tomllib.load(config_file)

    bot = BotSettings(**raw["bot"])
    execution = ExecutionSettings(**raw["execution"])
    risk = RiskSettings(**raw["risk"])
    symbols = [
        SymbolAssignment(
            ticker=entry["ticker"],
            strategy=create_strategy(entry["strategy"]),
        )
        for entry in raw["symbols"]
    ]

    return BotConfig(
        bot=bot,
        execution=execution,
        risk=risk,
        symbols=symbols,
    )
