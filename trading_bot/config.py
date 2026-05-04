"""Loads bot settings and per-symbol strategy assignments from TOML."""

from typing import Any
from dataclasses import dataclass
from pathlib import Path
import tomllib

from trading_bot.strategies import create_strategy, default_strategy_config


@dataclass(frozen=True)
class BotSettings:
    version: str
    run_continuously: bool
    poll_interval_seconds: int
    paper_trading: bool
    preflight_only: bool = False


@dataclass(frozen=True)
class ExecutionSettings:
    order_status_poll_seconds: int
    order_status_timeout_seconds: int
    enable_broker_side_stop_loss: bool
    force_test_trade: bool
    force_direction: str
    require_easy_to_borrow_for_short_entries: bool = False


@dataclass(frozen=True)
class RiskSettings:
    max_position_qty: int
    risk_fraction_of_buying_power: float
    stop_loss_pct: float
    take_profit_pct: float
    flatten_before_close: bool
    flatten_minutes_before_close: int
    max_total_position_fraction_of_buying_power: float = 0.0


@dataclass(frozen=True)
class LoggingSettings:
    enable_full_decision_log: bool = False
    enable_full_strategy_context_log: bool = False
    enable_notable_decision_log: bool = True
    enable_notable_strategy_context_log: bool = True


@dataclass(frozen=True)
class SymbolAssignment:
    ticker: str
    strategy: Any


@dataclass(frozen=True)
class AccountAssignment:
    name: str
    paper_trading: bool
    api_key_env: str
    secret_key_env: str
    fallback_api_key_env: str
    fallback_secret_key_env: str
    state_namespace: str
    symbols: list[SymbolAssignment]


@dataclass(frozen=True)
class BotConfig:
    bot: BotSettings
    execution: ExecutionSettings
    risk: RiskSettings
    logging: LoggingSettings
    symbols: list[SymbolAssignment]
    accounts: list[AccountAssignment]


def load_raw_config(path: str = "bot_config.toml") -> dict:
    """Load the raw TOML structure from disk."""
    config_path = Path(path)
    with config_path.open("rb") as config_file:
        return tomllib.load(config_file)


def load_config(path: str = "bot_config.toml") -> BotConfig:
    """Load the bot configuration from a TOML file."""
    raw = load_raw_config(path)

    bot = BotSettings(**raw["bot"])
    execution = ExecutionSettings(**raw["execution"])
    risk = RiskSettings(**raw["risk"])
    logging = LoggingSettings(**raw.get("logging", {}))

    def build_symbols(symbol_entries: list[dict]) -> list[SymbolAssignment]:
        return [
            SymbolAssignment(
                ticker=entry["ticker"],
                strategy=create_strategy(
                    {
                        **default_strategy_config(entry["strategy"]["name"]),
                        "symbol": entry["ticker"],
                        **entry["strategy"],
                    }
                ),
            )
            for entry in symbol_entries
        ]

    symbols = build_symbols(raw.get("symbols", []))
    accounts = [
        AccountAssignment(
            name=str(account.get("name", "default")),
            paper_trading=bool(account.get("paper_trading", bot.paper_trading)),
            api_key_env=str(account.get("api_key_env", "ALPACA_API_KEY")),
            secret_key_env=str(account.get("secret_key_env", "ALPACA_SECRET_KEY")),
            fallback_api_key_env=str(account.get("fallback_api_key_env", "")),
            fallback_secret_key_env=str(account.get("fallback_secret_key_env", "")),
            state_namespace=str(account.get("state_namespace", account.get("name", "default"))),
            symbols=build_symbols(account.get("symbols", [])),
        )
        for account in raw.get("accounts", [])
    ]

    if not accounts:
        accounts = [
            AccountAssignment(
                name="default",
                paper_trading=bot.paper_trading,
                api_key_env="ALPACA_API_KEY",
                secret_key_env="ALPACA_SECRET_KEY",
                fallback_api_key_env="",
                fallback_secret_key_env="",
                state_namespace="",
                symbols=symbols,
            )
        ]

    account_names = [account.name for account in accounts]
    duplicate_account_names = {name for name in account_names if account_names.count(name) > 1}
    if duplicate_account_names:
        raise ValueError(f"Duplicate account names in config: {', '.join(sorted(duplicate_account_names))}")

    empty_accounts = [account.name for account in accounts if not account.symbols]
    if empty_accounts:
        raise ValueError(f"Account(s) must configure at least one symbol: {', '.join(empty_accounts)}")

    state_keys = [
        symbol.ticker if not account.state_namespace else f"{account.state_namespace}:{symbol.ticker}"
        for account in accounts
        for symbol in account.symbols
    ]
    duplicate_state_keys = {state_key for state_key in state_keys if state_keys.count(state_key) > 1}
    if duplicate_state_keys:
        raise ValueError(f"Duplicate account state keys in config: {', '.join(sorted(duplicate_state_keys))}")

    return BotConfig(
        bot=bot,
        execution=execution,
        risk=risk,
        logging=logging,
        symbols=symbols,
        accounts=accounts,
    )
