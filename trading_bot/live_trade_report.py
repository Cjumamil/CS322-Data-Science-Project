"""Build backtest-style reports from executed paper trades."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from alpaca.common.enums import Sort
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from trading_bot.backtest import (
    NEW_YORK_TIMEZONE,
    _aligned_timestamp_for_index,
    interval_to_minutes,
    make_run_timestamp,
    max_drawdown_pct,
    trade_chart_padding_bars,
)
from trading_bot.broker import connect_alpaca
from trading_bot.config import BotConfig, load_config
from trading_bot.data import download_data_range
from trading_bot.strategies import create_strategy, default_strategy_config


PAPER_TRADE_REPORT_OUTPUT_DIR = "paper_trade_reports"
DEFAULT_FALLBACK_INITIAL_EQUITY = 100_000.0
TRADE_LOG_PATH = Path("trade_log.csv")
NOTABLE_CONTEXT_LOG_PATH = Path("notable_strategy_context_log.jsonl")

PAPER_REPORT_THEME = {
    "figure_bg": "#0f1513",
    "axes_bg": "#16201d",
    "text": "#edf4ee",
    "muted_text": "#bccbc1",
    "spine": "#567062",
    "grid": "#31433b",
    "day_boundary": "#7da996",
    "session_extended": "#244038",
    "session_overnight": "#173128",
    "candle_up": "#5eead4",
    "candle_down": "#ff8a65",
    "ema12": "#8ecae6",
    "ema26": "#ffd166",
    "ema200_high": "#ffb4a2",
    "ema200_close": "#d0b7ff",
    "ema200_low": "#95f9c3",
    "vwap": "#ffe066",
    "stop": "#ff6b6b",
    "target": "#80ed99",
    "signal": "#edf4ee",
    "entry": "#57ccff",
    "exit": "#ffbe0b",
    "macd": "#8ecae6",
    "signal_line": "#ffd166",
    "histogram_pos_strong": "#06d6a0",
    "histogram_pos_soft": "#a8f0dc",
    "histogram_neg_strong": "#ef476f",
    "histogram_neg_soft": "#ffc2cf",
    "rsi": "#b5e48c",
    "oversold": "#72efdd",
    "overbought": "#ff8fab",
    "equity": "#7bf1a8",
    "pnl_gain": "#5eead4",
    "pnl_loss": "#ff8a65",
}


@dataclass(frozen=True)
class ReconstructedTrade:
    trade_id: str
    account: str
    symbol: str
    strategy: str
    strategy_version: str
    interval: str
    side: str
    qty: int
    entry_decision_id: str
    exit_decision_id: str
    entry_signal_time: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    entry_reason: str
    exit_reason: str
    bars_held: int
    holding_minutes: float
    pnl: float
    return_pct: float
    stop_price: float | None
    take_profit_price: float | None
    stop_source: str | None
    take_profit_source: str | None
    entry_signal_context_json: str | None
    exit_signal_context_json: str | None
    strategy_config_json: str | None
    chart_group_key: str
    note: str | None = None


@dataclass(frozen=True)
class ReconstructionStats:
    filled_order_count: int
    entry_order_count: int
    exit_order_count: int
    completed_trade_count: int
    orphan_exit_count: int
    dangling_entry_count: int


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for one paper-trade report run."""
    parser = argparse.ArgumentParser(description="Build a backtest-style report from executed paper trades.")
    parser.add_argument("--account", help="Optional account filter, for example vwap_rsi_paper.")
    parser.add_argument("--strategy", help="Optional strategy filter, for example vwap_rsi_mean_reversion.")
    parser.add_argument("--symbol", help="Optional symbol filter, for example NVDA.")
    parser.add_argument("--start", help="Optional start date/datetime filter.")
    parser.add_argument("--end", help="Optional end date/datetime filter.")
    parser.add_argument("--config", default="bot_config.toml", help="Path to the bot TOML config.")
    parser.add_argument(
        "--output-dir",
        default=PAPER_TRADE_REPORT_OUTPUT_DIR,
        help="Directory where paper-trade report runs should be saved.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        help="Optional initial equity override used for the overview equity curve and return metrics.",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip chart generation even if matplotlib is available.",
    )
    parser.add_argument(
        "--skip-broker-history",
        action="store_true",
        help="Use only the local trade log and skip Alpaca order-history supplements.",
    )
    return parser.parse_args()


def normalize_filter_timestamp(raw_value: str, *, end_boundary: bool) -> pd.Timestamp:
    """Parse a user-facing date or datetime filter into UTC."""
    if len(raw_value.strip()) <= 10:
        timestamp = pd.Timestamp(raw_value)
        timestamp = timestamp.tz_localize(NEW_YORK_TIMEZONE)
        if end_boundary:
            timestamp += pd.Timedelta(days=1)
        return timestamp.tz_convert("UTC")

    timestamp = pd.Timestamp(raw_value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize(NEW_YORK_TIMEZONE)
    if end_boundary and raw_value.strip().count(":") < 2:
        timestamp += pd.Timedelta(minutes=1)
    return timestamp.tz_convert("UTC")


def load_trade_log(path: Path) -> pd.DataFrame:
    """Load the trade log and keep the original schema stable."""
    if not path.exists():
        raise FileNotFoundError(f"Trade log not found: {path}")
    df = pd.read_csv(path, dtype=str).fillna("")
    if df.empty:
        return df
    return df


def load_notable_context_index(path: Path) -> dict[str, dict]:
    """Load strategy-context rows keyed by decision_id."""
    if not path.exists():
        return {}

    context_index: dict[str, dict] = {}
    with path.open("r", encoding="utf-8") as context_file:
        for line in context_file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            decision_id = str(row.get("decision_id", "")).strip()
            if decision_id:
                context_index[decision_id] = row
    return context_index


def build_strategy_config_index(config: BotConfig) -> dict[tuple[str, str, str], dict]:
    """Index the active config by account, symbol, and strategy for fallback use."""
    strategy_index: dict[tuple[str, str, str], dict] = {}
    for account in config.accounts:
        for symbol_assignment in account.symbols:
            strategy = symbol_assignment.strategy
            strategy_index[(account.name, symbol_assignment.ticker, strategy.name)] = {
                **default_strategy_config(strategy.name),
                **strategy.build_strategy_parameters(),
                "name": strategy.name,
                "version": strategy.version,
                "interval": strategy.interval,
                "lookback_bars": strategy.lookback_bars,
                "symbol": symbol_assignment.ticker,
            }
    return strategy_index


def build_account_symbol_strategy_index(config: BotConfig) -> dict[tuple[str, str], dict]:
    """Index the active account/symbol assignments for broker-history supplements."""
    assignment_index: dict[tuple[str, str], dict] = {}
    for account in config.accounts:
        for symbol_assignment in account.symbols:
            strategy = symbol_assignment.strategy
            assignment_index[(account.name, symbol_assignment.ticker.upper())] = {
                "account_name": account.name,
                "api_key_env": account.api_key_env,
                "secret_key_env": account.secret_key_env,
                "fallback_api_key_env": account.fallback_api_key_env,
                "fallback_secret_key_env": account.fallback_secret_key_env,
                "paper_trading": account.paper_trading,
                "strategy_name": strategy.name,
                "strategy_version": strategy.version,
            }
    return assignment_index


def fetch_broker_history_supplements(
    *,
    config: BotConfig,
    filtered_orders_df: pd.DataFrame,
    account_symbol_strategy_index: dict[tuple[str, str], dict],
    start_utc: pd.Timestamp | None,
    end_utc: pd.Timestamp | None,
) -> pd.DataFrame:
    """Fetch filled broker-side stop exits that never landed in the local trade log."""
    if filtered_orders_df.empty:
        return pd.DataFrame()

    seen_order_ids = {
        str(order_id).strip()
        for order_id in filtered_orders_df.get("order_id", pd.Series(dtype=str)).tolist()
        if str(order_id).strip()
    }
    relevant_accounts = sorted(set(filtered_orders_df["account"].astype(str)))
    earliest_local_fill = parse_timestamp_series(filtered_orders_df["filled_at"]).min()
    request_start = start_utc
    if request_start is None and pd.notna(earliest_local_fill):
        request_start = earliest_local_fill - pd.Timedelta(days=2)
    if request_start is None:
        request_start = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=30))
    request_end = end_utc or pd.Timestamp(datetime.now(timezone.utc) + timedelta(days=1))

    supplemental_rows: list[dict] = []
    for account in config.accounts:
        if account.name not in relevant_accounts:
            continue

        trading_client = connect_alpaca(
            paper=account.paper_trading,
            api_key_env=account.api_key_env,
            secret_key_env=account.secret_key_env,
            fallback_api_key_env=account.fallback_api_key_env,
            fallback_secret_key_env=account.fallback_secret_key_env,
        )
        order_request = GetOrdersRequest(
            status=QueryOrderStatus.CLOSED,
            limit=500,
            after=request_start.to_pydatetime(),
            until=request_end.to_pydatetime(),
            direction=Sort.ASC,
            nested=False,
        )
        orders = list(trading_client.get_orders(filter=order_request))
        for order in orders:
            order_id = str(getattr(order, "id", "")).strip()
            if not order_id or order_id in seen_order_ids:
                continue

            status = str(getattr(getattr(order, "status", None), "value", getattr(order, "status", ""))).lower()
            if status != "filled":
                continue

            filled_at = getattr(order, "filled_at", None)
            if filled_at is None:
                continue
            filled_at_timestamp = pd.Timestamp(filled_at)
            if filled_at_timestamp.tzinfo is None:
                filled_at_timestamp = filled_at_timestamp.tz_localize("UTC")
            else:
                filled_at_timestamp = filled_at_timestamp.tz_convert("UTC")

            order_type = str(
                getattr(getattr(order, "order_type", None), "value", getattr(order, "order_type", ""))
            ).lower()
            if "stop" not in order_type:
                continue

            symbol = str(getattr(order, "symbol", "")).upper()
            assignment = account_symbol_strategy_index.get((account.name, symbol))
            if assignment is None:
                continue

            order_side = str(getattr(getattr(order, "side", None), "value", getattr(order, "side", ""))).upper()
            if order_side == "SELL":
                position_side_before = "long"
            elif order_side == "BUY":
                position_side_before = "short"
            else:
                continue

            submitted_at = parse_timestamp(getattr(order, "submitted_at", None))
            canceled_at = parse_timestamp(getattr(order, "canceled_at", None))
            failed_at = parse_timestamp(getattr(order, "failed_at", None))
            expired_at = parse_timestamp(getattr(order, "expired_at", None))
            supplemental_rows.append(
                {
                    "decision_id": f"broker_history:{account.name}:{order_id}",
                    "timestamp": filled_at_timestamp.isoformat(),
                    "bot_version": "",
                    "account": account.name,
                    "symbol": symbol,
                    "strategy": assignment["strategy_name"],
                    "strategy_version": assignment["strategy_version"],
                    "intended_action": order_side,
                    "position_side_before": position_side_before,
                    "position_side_after_expected": "flat",
                    "strategy_reason": "stop_loss",
                    "order_id": order_id,
                    "client_order_id": str(getattr(order, "client_order_id", "")).strip(),
                    "order_type": order_type,
                    "requested_qty": str(getattr(order, "qty", "")),
                    "filled_qty": str(getattr(order, "filled_qty", "")),
                    "final_status": status,
                    "decision_price": "",
                    "filled_avg_price": str(getattr(order, "filled_avg_price", "")),
                    "submitted_at": "" if submitted_at is None else submitted_at.isoformat(),
                    "filled_at": filled_at_timestamp.isoformat(),
                    "canceled_at": "" if canceled_at is None else canceled_at.isoformat(),
                    "failed_at": "" if failed_at is None else failed_at.isoformat(),
                    "expired_at": "" if expired_at is None else expired_at.isoformat(),
                    "note": "broker_history_stop_fill",
                    "filled_timestamp": filled_at_timestamp,
                    "submitted_timestamp": filled_at_timestamp if submitted_at is None else submitted_at,
                    "log_timestamp": filled_at_timestamp,
                }
            )

    if not supplemental_rows:
        return pd.DataFrame()

    return pd.DataFrame(supplemental_rows).sort_values(
        by=["filled_timestamp", "submitted_timestamp", "log_timestamp", "decision_id"]
    ).reset_index(drop=True)


def filter_filled_orders(
    trade_log_df: pd.DataFrame,
    *,
    account: str | None,
    strategy: str | None,
    symbol: str | None,
) -> pd.DataFrame:
    """Filter the trade log down to filled paper-trade rows."""
    if trade_log_df.empty:
        return trade_log_df.copy()

    work = trade_log_df.copy()
    work["final_status_normalized"] = work["final_status"].str.strip().str.lower()
    work = work[work["final_status_normalized"] == "filled"].copy()
    work = work[work["filled_avg_price"].str.strip() != ""].copy()
    work = work[work["filled_at"].str.strip() != ""].copy()

    if account:
        work = work[work["account"].str.strip() == account].copy()
    if strategy:
        work = work[work["strategy"].str.strip() == strategy].copy()
    if symbol:
        work = work[work["symbol"].str.strip().str.upper() == symbol.upper()].copy()

    if work.empty:
        return work

    work["filled_timestamp"] = parse_timestamp_series(work["filled_at"])
    work["submitted_timestamp"] = parse_timestamp_series(work["submitted_at"])
    work["log_timestamp"] = parse_timestamp_series(work["timestamp"])
    work = work.sort_values(
        by=["filled_timestamp", "submitted_timestamp", "log_timestamp", "decision_id"],
        na_position="last",
    ).reset_index(drop=True)
    return work


def parse_float(raw_value, default: float | None = None) -> float | None:
    """Parse one optional float field."""
    if raw_value in {None, ""}:
        return default
    try:
        return float(raw_value)
    except (TypeError, ValueError):
        return default


def parse_int(raw_value, default: int = 0) -> int:
    """Parse one optional integer field."""
    if raw_value in {None, ""}:
        return default
    try:
        return int(float(raw_value))
    except (TypeError, ValueError):
        return default


def parse_timestamp(raw_value) -> pd.Timestamp | None:
    """Parse one timestamp field as UTC."""
    if raw_value in {None, ""}:
        return None
    raw_text = str(raw_value).strip()
    if not raw_text or raw_text.lower() in {"none", "nan", "nat"}:
        return None
    timestamp = pd.Timestamp(raw_text)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def parse_timestamp_series(raw_values: pd.Series) -> pd.Series:
    """Parse one timestamp series while tolerating mixed text formats."""
    return raw_values.apply(lambda raw_value: pd.NaT if parse_timestamp(raw_value) is None else parse_timestamp(raw_value))


def coerce_json_text(value: dict | None) -> str | None:
    """Serialize one optional JSON object for flat CSV output."""
    if value is None:
        return None
    return json.dumps(value, separators=(",", ":"), sort_keys=True)


def first_present_timestamp(*raw_values) -> pd.Timestamp | None:
    """Return the first successfully parsed timestamp among the provided values."""
    for raw_value in raw_values:
        timestamp = parse_timestamp(raw_value)
        if timestamp is not None:
            return timestamp
    return None


def build_strategy_config_for_trade(
    *,
    entry_context: dict | None,
    exit_context: dict | None,
    fallback_strategy_config: dict | None,
    symbol: str,
    strategy_name: str,
    strategy_version: str,
) -> dict:
    """Resolve the most faithful strategy config available for one live trade."""
    context = entry_context or exit_context or {}
    metadata = context.get("strategy_metadata", {})
    parameters = context.get("strategy_parameters", {})
    market_data = context.get("market_data", {})

    if fallback_strategy_config is not None:
        strategy_config = dict(fallback_strategy_config)
    else:
        strategy_config = default_strategy_config(strategy_name)

    strategy_config.update(parameters)
    strategy_config["name"] = strategy_name
    strategy_config["version"] = str(metadata.get("version", strategy_version))
    strategy_config["interval"] = str(metadata.get("interval", strategy_config.get("interval", "5m")))
    lookback_bars = market_data.get("lookback_bars", strategy_config.get("lookback_bars"))
    if lookback_bars not in {None, ""}:
        strategy_config["lookback_bars"] = int(lookback_bars)
    strategy_config["symbol"] = symbol
    return strategy_config


def resolve_trade_side(entry_row: pd.Series) -> str:
    """Resolve the logical trade side from one entry order row."""
    side = str(entry_row.get("position_side_after_expected", "")).strip().lower()
    if side in {"long", "short"}:
        return side
    return "long" if str(entry_row.get("intended_action", "")).strip().upper() == "BUY" else "short"


def calculate_trade_pnl(side: str, qty: int, entry_price: float, exit_price: float) -> float:
    """Calculate realized PnL from actual filled prices."""
    if side == "short":
        return (entry_price - exit_price) * qty
    return (exit_price - entry_price) * qty


def context_bar_end(context: dict | None) -> pd.Timestamp | None:
    """Return the signal-bar end timestamp from one context row."""
    if context is None:
        return None
    return parse_timestamp(context.get("market_data", {}).get("bar_end"))


def context_active_entry_signal_time(context: dict | None) -> pd.Timestamp | None:
    """Return the persisted entry signal time from one exit context, if present."""
    if context is None:
        return None
    return parse_timestamp(context.get("strategy_signals", {}).get("active_entry_signal_time"))


def estimate_bars_held(
    *,
    interval: str,
    entry_signal_time: pd.Timestamp | None,
    entry_time: pd.Timestamp,
    exit_time: pd.Timestamp,
    explicit_bars_held: int | None,
) -> int:
    """Estimate bars held when the exit context does not already provide it."""
    if explicit_bars_held is not None and explicit_bars_held > 0:
        return explicit_bars_held

    anchor_time = entry_signal_time or entry_time
    seconds_per_bar = interval_to_minutes(interval) * 60
    if seconds_per_bar <= 0:
        return 1
    held_seconds = max((exit_time - anchor_time).total_seconds(), seconds_per_bar)
    return max(1, int(math.ceil(held_seconds / seconds_per_bar)))


def extract_stop_and_target_prices(
    *,
    side: str,
    entry_context: dict | None,
    exit_context: dict | None,
) -> tuple[float | None, float | None, str | None, str | None]:
    """Resolve stop/target metadata from the entry and exit context rows."""
    entry_signals = {} if entry_context is None else entry_context.get("strategy_signals", {})
    exit_signals = {} if exit_context is None else exit_context.get("strategy_signals", {})
    entry_parameters = {} if entry_context is None else entry_context.get("strategy_parameters", {})
    exit_parameters = {} if exit_context is None else exit_context.get("strategy_parameters", {})

    stop_price = parse_float(exit_signals.get("active_stop_price"))
    take_profit_price = parse_float(exit_signals.get("active_take_profit_price"))
    if side == "long":
        if stop_price is None:
            stop_price = parse_float(entry_signals.get("long_stop_price"))
        if take_profit_price is None:
            take_profit_price = parse_float(entry_signals.get("long_take_profit_price"))
    else:
        if stop_price is None:
            stop_price = parse_float(entry_signals.get("short_stop_price"))
        if take_profit_price is None:
            take_profit_price = parse_float(entry_signals.get("short_take_profit_price"))

    stop_source = exit_signals.get("active_stop_source") or entry_parameters.get("stop_model")
    take_profit_source = exit_parameters.get("take_profit_model") or entry_parameters.get("take_profit_model")
    return stop_price, take_profit_price, stop_source, take_profit_source


def resolve_exit_entry_side(exit_row: pd.Series) -> str | None:
    """Infer which entry side one exit row should close."""
    position_before = str(exit_row.get("position_side_before", "")).strip().lower()
    if position_before in {"long", "short"}:
        return position_before

    intended_action = str(exit_row.get("intended_action", "")).strip().upper()
    if intended_action == "SELL":
        return "long"
    if intended_action == "BUY":
        return "short"
    return None


def pop_deque_index(entry_queue: deque[pd.Series], index: int) -> pd.Series:
    """Remove one queued entry row by index while preserving the remaining order."""
    entry_queue.rotate(-index)
    entry_row = entry_queue.popleft()
    entry_queue.rotate(index)
    return entry_row


def pop_matching_entry_row(
    *,
    entry_queue: deque[pd.Series],
    exit_row: pd.Series,
    exit_time: pd.Timestamp,
) -> pd.Series | None:
    """Select the best open entry for one exit without ever pairing to a future entry."""
    desired_entry_side = resolve_exit_entry_side(exit_row)
    stop_submitted_time = parse_timestamp(exit_row.get("submitted_at"))
    use_stop_anchor = str(exit_row.get("note", "")).strip().lower() == "broker_history_stop_fill"
    candidates: list[tuple[tuple[float, pd.Timestamp, int], int]] = []

    for index, candidate_entry in enumerate(entry_queue):
        entry_time = first_present_timestamp(
            candidate_entry.get("filled_at"),
            candidate_entry.get("submitted_at"),
            candidate_entry.get("timestamp"),
        )
        if entry_time is None or entry_time > exit_time:
            continue
        if desired_entry_side is not None and resolve_trade_side(candidate_entry) != desired_entry_side:
            continue

        if use_stop_anchor and stop_submitted_time is not None:
            priority = (abs((stop_submitted_time - entry_time).total_seconds()), entry_time, index)
        else:
            priority = (0.0, entry_time, index)
        candidates.append((priority, index))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0])
    return pop_deque_index(entry_queue, candidates[0][1])


def build_reconstructed_trade(
    *,
    trade_id: str,
    group_key: tuple[str, str, str],
    entry_row: pd.Series,
    exit_row: pd.Series,
    entry_context: dict | None,
    exit_context: dict | None,
    fallback_strategy_config: dict | None,
) -> ReconstructedTrade | None:
    """Build one completed trade from a matched entry and exit row."""
    symbol = group_key[1]
    strategy_name = group_key[2]
    strategy_version = str(entry_row.get("strategy_version", "") or exit_row.get("strategy_version", "")).strip()
    strategy_config = build_strategy_config_for_trade(
        entry_context=entry_context,
        exit_context=exit_context,
        fallback_strategy_config=fallback_strategy_config,
        symbol=symbol,
        strategy_name=strategy_name,
        strategy_version=strategy_version or "unknown",
    )
    interval = str(strategy_config.get("interval", "5m"))

    entry_time = first_present_timestamp(entry_row.get("filled_at"), entry_row.get("submitted_at"), entry_row.get("timestamp"))
    exit_time = first_present_timestamp(exit_row.get("filled_at"), exit_row.get("submitted_at"), exit_row.get("timestamp"))
    if entry_time is None or exit_time is None or exit_time < entry_time:
        return None

    side = resolve_trade_side(entry_row)
    qty = parse_int(entry_row["filled_qty"], default=parse_int(entry_row["requested_qty"], default=0))
    entry_price = parse_float(entry_row["filled_avg_price"], default=0.0) or 0.0
    exit_price = parse_float(exit_row["filled_avg_price"], default=0.0) or 0.0
    pnl = calculate_trade_pnl(side, qty, entry_price, exit_price)
    notional = abs(entry_price * qty)
    return_pct = (pnl / notional) * 100 if notional else 0.0

    entry_signal_time = (
        context_active_entry_signal_time(exit_context)
        or context_bar_end(entry_context)
        or context_bar_end(exit_context)
        or entry_time
    )
    explicit_bars_held = parse_int(
        None if exit_context is None else exit_context.get("strategy_signals", {}).get("bars_in_trade"),
        default=0,
    )
    bars_held = estimate_bars_held(
        interval=interval,
        entry_signal_time=entry_signal_time,
        entry_time=entry_time,
        exit_time=exit_time,
        explicit_bars_held=explicit_bars_held or None,
    )
    stop_price, take_profit_price, stop_source, take_profit_source = extract_stop_and_target_prices(
        side=side,
        entry_context=entry_context,
        exit_context=exit_context,
    )
    exit_reason = str(exit_row.get("strategy_reason", "")).strip() or "stop_loss"
    note = str(exit_row.get("note", "")).strip() or None

    return ReconstructedTrade(
        trade_id=trade_id,
        account=group_key[0],
        symbol=symbol,
        strategy=strategy_name,
        strategy_version=strategy_version or str(strategy_config.get("version", "")),
        interval=interval,
        side=side,
        qty=qty,
        entry_decision_id=str(entry_row["decision_id"]).strip(),
        exit_decision_id=str(exit_row["decision_id"]).strip(),
        entry_signal_time=entry_signal_time.isoformat(),
        entry_time=entry_time.isoformat(),
        entry_price=round(entry_price, 6),
        exit_time=exit_time.isoformat(),
        exit_price=round(exit_price, 6),
        entry_reason=str(entry_row.get("strategy_reason", "")).strip(),
        exit_reason=exit_reason,
        bars_held=bars_held,
        holding_minutes=round((exit_time - entry_time).total_seconds() / 60, 4),
        pnl=round(pnl, 6),
        return_pct=round(return_pct, 6),
        stop_price=None if stop_price is None else round(stop_price, 6),
        take_profit_price=None if take_profit_price is None else round(take_profit_price, 6),
        stop_source=None if not stop_source else str(stop_source),
        take_profit_source=None if not take_profit_source else str(take_profit_source),
        entry_signal_context_json=coerce_json_text(entry_context),
        exit_signal_context_json=coerce_json_text(exit_context),
        strategy_config_json=coerce_json_text(strategy_config),
        chart_group_key=json.dumps(
            {
                "symbol": symbol,
                "strategy": strategy_name,
                "interval": interval,
                "strategy_config": strategy_config,
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        note=note,
    )


def trade_matches_time_window(
    trade: ReconstructedTrade,
    *,
    start_utc: pd.Timestamp | None,
    end_utc: pd.Timestamp | None,
) -> bool:
    """Return True when the trade overlaps the selected report window."""
    entry_time = pd.Timestamp(trade.entry_time)
    exit_time = pd.Timestamp(trade.exit_time)

    if start_utc is not None and exit_time < start_utc:
        return False
    if end_utc is not None and entry_time >= end_utc:
        return False
    return True


def reconstruct_completed_trades(
    *,
    filled_orders_df: pd.DataFrame,
    context_index: dict[str, dict],
    strategy_config_index: dict[tuple[str, str, str], dict],
    start_utc: pd.Timestamp | None,
    end_utc: pd.Timestamp | None,
) -> tuple[list[ReconstructedTrade], ReconstructionStats]:
    """Pair filled entry and exit rows into completed round-trip paper trades."""
    open_entries: dict[tuple[str, str, str], deque[pd.Series]] = defaultdict(deque)
    completed_trades: list[ReconstructedTrade] = []
    orphan_exit_count = 0
    entry_order_count = 0
    exit_order_count = 0

    for _, row in filled_orders_df.iterrows():
        position_before = str(row["position_side_before"]).strip().lower()
        position_after = str(row["position_side_after_expected"]).strip().lower()
        note = str(row.get("note", "")).strip().lower()
        group_key = (
            str(row["account"]).strip(),
            str(row["symbol"]).strip().upper(),
            str(row["strategy"]).strip(),
        )

        if position_before == "flat" and position_after in {"long", "short"}:
            entry_order_count += 1
            open_entries[group_key].append(row)
            continue

        is_exit_row = (position_before in {"long", "short"} and position_after == "flat") or note == "broker_history_stop_fill"
        if is_exit_row:
            exit_order_count += 1
            exit_time = first_present_timestamp(row.get("filled_at"), row.get("submitted_at"), row.get("timestamp"))
            if exit_time is None or not open_entries[group_key]:
                orphan_exit_count += 1
                continue

            entry_row = pop_matching_entry_row(
                entry_queue=open_entries[group_key],
                exit_row=row,
                exit_time=exit_time,
            )
            if entry_row is None:
                orphan_exit_count += 1
                continue

            entry_context = context_index.get(str(entry_row["decision_id"]).strip())
            exit_context = None if note == "broker_history_stop_fill" else context_index.get(str(row["decision_id"]).strip())
            fallback_strategy_config = strategy_config_index.get(group_key)

            trade = build_reconstructed_trade(
                trade_id=f"T{len(completed_trades) + 1:03d}",
                group_key=group_key,
                entry_row=entry_row,
                exit_row=row,
                entry_context=entry_context,
                exit_context=exit_context,
                fallback_strategy_config=fallback_strategy_config,
            )
            if trade is None:
                orphan_exit_count += 1
                continue
            if trade_matches_time_window(trade, start_utc=start_utc, end_utc=end_utc):
                completed_trades.append(trade)

    dangling_entry_count = sum(len(entries) for entries in open_entries.values())
    stats = ReconstructionStats(
        filled_order_count=int(len(filled_orders_df)),
        entry_order_count=entry_order_count,
        exit_order_count=exit_order_count,
        completed_trade_count=len(completed_trades),
        orphan_exit_count=orphan_exit_count,
        dangling_entry_count=dangling_entry_count,
    )
    return completed_trades, stats


def estimate_initial_equity(
    trades: list[ReconstructedTrade],
    context_index: dict[str, dict],
    cli_initial_equity: float | None,
) -> tuple[float, str]:
    """Resolve the equity base used for overview return metrics."""
    if cli_initial_equity is not None:
        return float(cli_initial_equity), "cli_override"

    earliest_equity_by_account: dict[str, tuple[pd.Timestamp, float]] = {}
    for trade in trades:
        entry_context = context_index.get(trade.entry_decision_id)
        if entry_context is None:
            continue
        account_snapshot = entry_context.get("account", {}) or {}
        account_equity = (
            parse_float(account_snapshot.get("equity"))
            or parse_float(account_snapshot.get("portfolio_value"))
            or parse_float(account_snapshot.get("last_equity"))
        )
        entry_time = parse_timestamp(trade.entry_time)
        if account_equity is None or entry_time is None:
            continue
        current = earliest_equity_by_account.get(trade.account)
        if current is None or entry_time < current[0]:
            earliest_equity_by_account[trade.account] = (entry_time, account_equity)

    if earliest_equity_by_account:
        return (
            round(sum(value for _, value in earliest_equity_by_account.values()), 4),
            "first_context_equity_estimate",
        )

    unique_account_count = len({trade.account for trade in trades}) or 1
    return DEFAULT_FALLBACK_INITIAL_EQUITY * unique_account_count, "fallback_default"


def build_equity_curve(
    *,
    trades_df: pd.DataFrame,
    initial_equity: float,
) -> pd.DataFrame:
    """Build a simple realized-equity curve from completed trades."""
    if trades_df.empty:
        return pd.DataFrame(columns=["timestamp", "equity", "cumulative_pnl", "trade_id"])

    work = trades_df.copy()
    work["exit_timestamp"] = pd.to_datetime(work["exit_time"], utc=True)
    work = work.sort_values(by=["exit_timestamp", "trade_id"]).reset_index(drop=True)
    work["cumulative_pnl"] = work["pnl"].astype(float).cumsum()
    work["equity"] = initial_equity + work["cumulative_pnl"]
    return work.rename(columns={"exit_timestamp": "timestamp"})[
        ["timestamp", "equity", "cumulative_pnl", "trade_id", "symbol", "account", "strategy"]
    ].copy()


def build_paper_trade_summary(
    *,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    selected_filters: dict,
    initial_equity: float,
    initial_equity_source: str,
    reconstruction_stats: ReconstructionStats,
    source_logs: dict,
) -> dict:
    """Build the summary payload saved beside the report artifacts."""
    trade_count = int(len(trades_df))
    wins = int((trades_df["pnl"] > 0).sum()) if trade_count else 0
    losses = int((trades_df["pnl"] < 0).sum()) if trade_count else 0
    long_trades = int((trades_df["side"] == "long").sum()) if trade_count else 0
    short_trades = int((trades_df["side"] == "short").sum()) if trade_count else 0
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()) if trade_count else 0.0
    gross_loss = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()) if trade_count else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    total_pnl = float(trades_df["pnl"].sum()) if trade_count else 0.0
    avg_trade_pnl = float(trades_df["pnl"].mean()) if trade_count else 0.0
    avg_trade_return_pct = float(trades_df["return_pct"].mean()) if trade_count else 0.0
    best_trade_pct = float(trades_df["return_pct"].max()) if trade_count else 0.0
    worst_trade_pct = float(trades_df["return_pct"].min()) if trade_count else 0.0
    avg_holding_minutes = float(trades_df["holding_minutes"].mean()) if trade_count else 0.0
    exit_reason_counts = trades_df["exit_reason"].value_counts().to_dict() if trade_count else {}
    ending_equity = float(equity_df["equity"].iloc[-1]) if not equity_df.empty else initial_equity

    symbol_breakdown = {}
    if trade_count:
        grouped = trades_df.groupby("symbol", dropna=False)
        for symbol, symbol_df in grouped:
            symbol_breakdown[str(symbol)] = {
                "trade_count": int(len(symbol_df)),
                "win_rate_pct": round(float((symbol_df["pnl"] > 0).mean() * 100), 4),
                "total_pnl": round(float(symbol_df["pnl"].sum()), 4),
                "average_trade_return_pct": round(float(symbol_df["return_pct"].mean()), 4),
            }

    account_breakdown = {}
    if trade_count:
        grouped = trades_df.groupby("account", dropna=False)
        for account, account_df in grouped:
            account_breakdown[str(account)] = {
                "trade_count": int(len(account_df)),
                "win_rate_pct": round(float((account_df["pnl"] > 0).mean() * 100), 4),
                "total_pnl": round(float(account_df["pnl"].sum()), 4),
                "symbol_count": int(account_df["symbol"].nunique()),
            }

    strategy_breakdown = {}
    if trade_count:
        grouped = trades_df.groupby("strategy", dropna=False)
        for strategy, strategy_df in grouped:
            strategy_breakdown[str(strategy)] = {
                "trade_count": int(len(strategy_df)),
                "win_rate_pct": round(float((strategy_df["pnl"] > 0).mean() * 100), 4),
                "total_pnl": round(float(strategy_df["pnl"].sum()), 4),
            }

    analysis_start = None if trades_df.empty else pd.to_datetime(trades_df["entry_time"], utc=True).min().isoformat()
    analysis_end = None if trades_df.empty else pd.to_datetime(trades_df["exit_time"], utc=True).max().isoformat()

    return {
        "report_kind": "paper_trade_report",
        "analysis_start": analysis_start,
        "analysis_end": analysis_end,
        "selected_filters": selected_filters,
        "initial_equity": round(initial_equity, 4),
        "initial_equity_source": initial_equity_source,
        "ending_equity": round(ending_equity, 4),
        "total_pnl": round(total_pnl, 4),
        "total_return_pct": round(((ending_equity / initial_equity) - 1) * 100, 4) if initial_equity else None,
        "max_drawdown_pct": round(max_drawdown_pct(equity_df), 4) if not equity_df.empty else 0.0,
        "trade_count": trade_count,
        "win_count": wins,
        "loss_count": losses,
        "win_rate_pct": round((wins / trade_count) * 100, 4) if trade_count else 0.0,
        "long_trade_count": long_trades,
        "short_trade_count": short_trades,
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "profit_factor": round(profit_factor, 4) if profit_factor is not None else None,
        "average_trade_pnl": round(avg_trade_pnl, 4),
        "average_trade_return_pct": round(avg_trade_return_pct, 4),
        "best_trade_return_pct": round(best_trade_pct, 4),
        "worst_trade_return_pct": round(worst_trade_pct, 4),
        "average_holding_minutes": round(avg_holding_minutes, 4),
        "exit_reason_counts": exit_reason_counts,
        "symbol_breakdown": symbol_breakdown,
        "account_breakdown": account_breakdown,
        "strategy_breakdown": strategy_breakdown,
        "reconstruction_stats": asdict(reconstruction_stats),
        "source_logs": source_logs,
        "generated_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }


def build_output_paths(output_dir: str, slug: str, run_timestamp: str) -> tuple[Path, Path, Path, Path]:
    """Return the standard output paths for one paper-trade report run."""
    root = Path(output_dir) / f"{run_timestamp}_{slug}"
    root.mkdir(parents=True, exist_ok=True)
    return (
        root,
        root / "trades.csv",
        root / "summary.json",
        root / "overview_chart.png",
    )


def make_selector_slug(selected_values: set[str], fallback_label: str) -> str:
    """Build one compact slug component from one selected-value set."""
    if not selected_values:
        return fallback_label
    if len(selected_values) == 1:
        return next(iter(selected_values))
    return fallback_label


def make_report_slug(args: argparse.Namespace, trades: list[ReconstructedTrade]) -> str:
    """Build a short filesystem-friendly run slug from the selected trades."""
    accounts = {trade.account for trade in trades}
    account_label = make_selector_slug(accounts, "multi_account")
    return account_label


def build_prepared_data_cache(trades: list[ReconstructedTrade]) -> dict[str, tuple[pd.DataFrame, str]]:
    """Fetch and prepare market data once per unique symbol/config combination."""
    grouped_trades: dict[str, list[ReconstructedTrade]] = defaultdict(list)
    for trade in trades:
        grouped_trades[trade.chart_group_key].append(trade)

    prepared_data_cache: dict[str, tuple[pd.DataFrame, str]] = {}
    for chart_group_key, group_trades in grouped_trades.items():
        group_definition = json.loads(chart_group_key)
        strategy_config = group_definition["strategy_config"]
        interval = str(group_definition["interval"])
        symbol = str(group_definition["symbol"])
        strategy = create_strategy(strategy_config)

        signal_times = [pd.Timestamp(trade.entry_signal_time) for trade in group_trades]
        exit_times = [pd.Timestamp(trade.exit_time) for trade in group_trades]
        start_utc = min(signal_times) - (pd.Timedelta(minutes=interval_to_minutes(interval)) * 24)
        end_utc = max(exit_times) + (pd.Timedelta(minutes=interval_to_minutes(interval)) * 24)
        raw_df = download_data_range(
            symbol=symbol,
            interval=interval,
            start=start_utc.to_pydatetime(),
            end=end_utc.to_pydatetime(),
            warmup_bars=int(strategy.lookback_bars),
        )
        prepared_data_cache[chart_group_key] = (strategy.prepare_dataframe(raw_df), interval)

    return prepared_data_cache


def plot_paper_trade_zoom_charts(
    *,
    prepared_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    output_dir: Path,
    interval: str,
) -> list[Path]:
    """Save one zoomed paper-trade chart per completed trade."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib is not installed, so the trade charts were skipped.")
        return []

    if trades_df.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    def eastern_index(index: pd.Index) -> pd.DatetimeIndex:
        eastern = pd.DatetimeIndex(index)
        if eastern.tz is None:
            eastern = eastern.tz_localize("UTC")
        return eastern.tz_convert(NEW_YORK_TIMEZONE)

    def shade_session_phases(ax, phase_labels: list[str]) -> None:
        start = None
        active_phase = None
        for idx, phase in enumerate(phase_labels):
            if phase == "regular":
                if start is not None and active_phase is not None:
                    ax.axvspan(
                        start - 0.5,
                        idx - 0.5,
                        color=PAPER_REPORT_THEME[f"session_{active_phase}"],
                        alpha=0.34 if active_phase == "overnight" else 0.26,
                        zorder=0,
                    )
                    start = None
                    active_phase = None
                continue
            if start is None:
                start = idx
                active_phase = phase
                continue
            if phase != active_phase:
                ax.axvspan(
                    start - 0.5,
                    idx - 0.5,
                    color=PAPER_REPORT_THEME[f"session_{active_phase}"],
                    alpha=0.34 if active_phase == "overnight" else 0.26,
                    zorder=0,
                )
                start = idx
                active_phase = phase
        if start is not None and active_phase is not None:
            ax.axvspan(
                start - 0.5,
                len(phase_labels) - 0.5,
                color=PAPER_REPORT_THEME[f"session_{active_phase}"],
                alpha=0.34 if active_phase == "overnight" else 0.26,
                zorder=0,
            )

    def draw_day_boundaries(ax, local_index: pd.DatetimeIndex) -> None:
        if len(local_index) < 2:
            return
        local_dates = local_index.date
        for idx in range(1, len(local_dates)):
            if local_dates[idx] != local_dates[idx - 1]:
                ax.axvline(
                    idx - 0.5,
                    color=PAPER_REPORT_THEME["day_boundary"],
                    linewidth=1.2,
                    alpha=0.8,
                    zorder=1,
                )

    def draw_candlesticks(ax, window_df: pd.DataFrame, x_values: list[int]) -> None:
        width = 0.82
        for x_value, (_, row) in zip(x_values, window_df.iterrows()):
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            high_price = float(row["High"])
            low_price = float(row["Low"])
            candle_color = (
                PAPER_REPORT_THEME["candle_up"]
                if close_price >= open_price
                else PAPER_REPORT_THEME["candle_down"]
            )
            ax.vlines(x_value, low_price, high_price, color=candle_color, linewidth=1.5, alpha=0.98, zorder=2)
            body_bottom = min(open_price, close_price)
            body_height = max(abs(close_price - open_price), 0.02)
            ax.add_patch(
                Rectangle(
                    (x_value - (width / 2), body_bottom),
                    width,
                    body_height,
                    facecolor=candle_color,
                    edgecolor=candle_color,
                    linewidth=1.0,
                    alpha=0.9,
                    zorder=3,
                )
            )

    def detect_indicator_kind(window_df: pd.DataFrame) -> str | None:
        if {"MACD", "signal_line", "histogram"}.issubset(window_df.columns):
            return "macd"
        if "RSI" in window_df.columns:
            return "rsi"
        return None

    for _, trade in trades_df.iterrows():
        entry_time = _aligned_timestamp_for_index(trade["entry_time"], prepared_df.index)
        exit_time = _aligned_timestamp_for_index(trade["exit_time"], prepared_df.index)
        entry_signal_time = _aligned_timestamp_for_index(trade["entry_signal_time"], prepared_df.index)
        bars_before_entry, bars_after_exit = trade_chart_padding_bars(
            interval=interval,
            bars_held=int(trade.get("bars_held", 1)),
        )

        entry_loc = int(prepared_df.index.searchsorted(entry_time, side="left"))
        exit_loc = int(prepared_df.index.searchsorted(exit_time, side="left"))
        exit_loc = max(entry_loc, exit_loc)
        start_loc = max(0, entry_loc - bars_before_entry)
        end_loc = min(len(prepared_df), exit_loc + bars_after_exit + 1)
        window_df = prepared_df.iloc[start_loc:end_loc].copy()
        if window_df.empty:
            continue

        x_positions = list(range(len(window_df)))
        local_index = eastern_index(window_df.index)
        minutes_of_day = (local_index.hour * 60) + local_index.minute
        phase_labels = []
        for minute in minutes_of_day:
            if ((9 * 60) + 30) <= minute < (16 * 60):
                phase_labels.append("regular")
            elif ((4 * 60) <= minute < ((9 * 60) + 30)) or ((16 * 60) <= minute < (20 * 60)):
                phase_labels.append("extended")
            else:
                phase_labels.append("overnight")

        signal_x = max(0, min(len(window_df) - 1, int(window_df.index.searchsorted(entry_signal_time, side="left"))))
        entry_x = max(0, min(len(window_df) - 1, int(window_df.index.searchsorted(entry_time, side="left"))))
        exit_x = max(0, min(len(window_df) - 1, int(window_df.index.searchsorted(exit_time, side="left"))))
        indicator_kind = detect_indicator_kind(window_df)

        if indicator_kind is None:
            fig, price_ax = plt.subplots(1, 1, figsize=(14, 5), sharex=True)
            axes = [price_ax]
            indicator_ax = None
        else:
            fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[3, 2])
            price_ax, indicator_ax = axes

        fig.patch.set_facecolor(PAPER_REPORT_THEME["figure_bg"])
        for axis in axes:
            axis.set_facecolor(PAPER_REPORT_THEME["axes_bg"])
            axis.tick_params(colors=PAPER_REPORT_THEME["muted_text"])
            for spine in axis.spines.values():
                spine.set_color(PAPER_REPORT_THEME["spine"])

        shade_session_phases(price_ax, phase_labels)
        draw_day_boundaries(price_ax, local_index)
        if indicator_ax is not None:
            shade_session_phases(indicator_ax, phase_labels)
            draw_day_boundaries(indicator_ax, local_index)

        draw_candlesticks(price_ax, window_df, x_positions)
        if "EMA12" in window_df.columns:
            price_ax.plot(x_positions, window_df["EMA12"], label="EMA12", linewidth=1.1, color=PAPER_REPORT_THEME["ema12"], zorder=4)
        if "EMA26" in window_df.columns:
            price_ax.plot(x_positions, window_df["EMA26"], label="EMA26", linewidth=1.1, color=PAPER_REPORT_THEME["ema26"], zorder=4)
        if "EMA200_high" in window_df.columns:
            price_ax.plot(x_positions, window_df["EMA200_high"], label="EMA200 high", linewidth=1.0, color=PAPER_REPORT_THEME["ema200_high"], zorder=4)
        if "EMA200_close" in window_df.columns:
            price_ax.plot(x_positions, window_df["EMA200_close"], label="EMA200 close", linewidth=1.0, color=PAPER_REPORT_THEME["ema200_close"], zorder=4)
        if "EMA200_low" in window_df.columns:
            price_ax.plot(x_positions, window_df["EMA200_low"], label="EMA200 low", linewidth=1.0, color=PAPER_REPORT_THEME["ema200_low"], zorder=4)
        if "VWAP" in window_df.columns:
            price_ax.plot(x_positions, window_df["VWAP"], label="VWAP", linewidth=1.1, color=PAPER_REPORT_THEME["vwap"], zorder=4)

        stop_price = parse_float(trade.get("stop_price"))
        target_price = parse_float(trade.get("take_profit_price"))
        if stop_price is not None:
            price_ax.axhline(stop_price, color=PAPER_REPORT_THEME["stop"], linestyle="--", linewidth=1.0, alpha=0.85, label="Stop", zorder=1)
        if target_price is not None:
            price_ax.axhline(target_price, color=PAPER_REPORT_THEME["target"], linestyle="--", linewidth=1.0, alpha=0.85, label="Target", zorder=1)

        price_ax.scatter(
            signal_x,
            float(trade["entry_price"]),
            marker="o",
            s=65,
            facecolors="none",
            edgecolors=PAPER_REPORT_THEME["signal"],
            linewidths=1.7,
            label="Signal",
            zorder=8,
        )
        price_ax.scatter(
            entry_x,
            float(trade["entry_price"]),
            marker="D",
            s=100,
            color=PAPER_REPORT_THEME["entry"],
            edgecolors=PAPER_REPORT_THEME["figure_bg"],
            linewidths=1.1,
            label="Entry",
            zorder=9,
        )
        price_ax.scatter(
            exit_x,
            float(trade["exit_price"]),
            marker="X",
            s=115,
            color=PAPER_REPORT_THEME["exit"],
            edgecolors=PAPER_REPORT_THEME["figure_bg"],
            linewidths=0.9,
            label="Exit",
            zorder=10,
        )
        price_ax.set_ylabel("Price", color=PAPER_REPORT_THEME["text"])
        price_ax.set_title(
            f"{trade['symbol']} | {trade['trade_id']} | {trade['side']} | {trade['interval']} | {trade['exit_reason']} | bars held {trade['bars_held']}",
            color=PAPER_REPORT_THEME["text"],
        )
        price_ax.grid(alpha=0.25, color=PAPER_REPORT_THEME["grid"])
        price_legend = price_ax.legend(loc="upper left", ncol=3)
        price_legend.get_frame().set_facecolor(PAPER_REPORT_THEME["axes_bg"])
        price_legend.get_frame().set_edgecolor(PAPER_REPORT_THEME["spine"])
        for text in price_legend.get_texts():
            text.set_color(PAPER_REPORT_THEME["text"])

        if indicator_ax is not None and indicator_kind == "macd":
            indicator_ax.plot(x_positions, window_df["MACD"], label="MACD", linewidth=1.0, color=PAPER_REPORT_THEME["macd"])
            indicator_ax.plot(x_positions, window_df["signal_line"], label="Signal", linewidth=1.0, color=PAPER_REPORT_THEME["signal_line"])
            histogram_values = window_df["histogram"].fillna(0)
            histogram_previous = histogram_values.shift(1).fillna(0)
            bar_colors = []
            for current_value, previous_value in zip(histogram_values, histogram_previous):
                if current_value >= 0:
                    bar_colors.append(
                        PAPER_REPORT_THEME["histogram_pos_strong"]
                        if current_value >= previous_value
                        else PAPER_REPORT_THEME["histogram_pos_soft"]
                    )
                else:
                    bar_colors.append(
                        PAPER_REPORT_THEME["histogram_neg_strong"]
                        if current_value <= previous_value
                        else PAPER_REPORT_THEME["histogram_neg_soft"]
                    )
            indicator_ax.bar(x_positions, window_df["histogram"], label="Histogram", alpha=0.82, width=0.82, color=bar_colors)
            indicator_ax.axhline(0, color=PAPER_REPORT_THEME["signal"], linewidth=0.8, alpha=0.55)
            indicator_ax.set_ylabel("MACD", color=PAPER_REPORT_THEME["text"])

        if indicator_ax is not None and indicator_kind == "rsi":
            indicator_ax.plot(x_positions, window_df["RSI"], label="RSI", linewidth=1.2, color=PAPER_REPORT_THEME["rsi"])
            if "rsi_oversold_level" in window_df.columns:
                indicator_ax.axhline(
                    float(window_df["rsi_oversold_level"].iloc[-1]),
                    color=PAPER_REPORT_THEME["oversold"],
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.85,
                    label="Oversold",
                )
            if "rsi_overbought_level" in window_df.columns:
                indicator_ax.axhline(
                    float(window_df["rsi_overbought_level"].iloc[-1]),
                    color=PAPER_REPORT_THEME["overbought"],
                    linestyle="--",
                    linewidth=0.9,
                    alpha=0.85,
                    label="Overbought",
                )
            indicator_ax.axhline(50, color=PAPER_REPORT_THEME["signal"], linewidth=0.8, alpha=0.35)
            indicator_ax.set_ylim(0, 100)
            indicator_ax.set_ylabel("RSI", color=PAPER_REPORT_THEME["text"])

        if indicator_ax is not None:
            indicator_ax.set_xlabel("Time (EST/EDT)", color=PAPER_REPORT_THEME["text"])
            indicator_ax.grid(alpha=0.25, color=PAPER_REPORT_THEME["grid"])
            indicator_legend = indicator_ax.legend(loc="upper left")
            indicator_legend.get_frame().set_facecolor(PAPER_REPORT_THEME["axes_bg"])
            indicator_legend.get_frame().set_edgecolor(PAPER_REPORT_THEME["spine"])
            for text in indicator_legend.get_texts():
                text.set_color(PAPER_REPORT_THEME["text"])

        if len(window_df) > 1:
            tick_count = min(6, len(window_df))
            tick_positions = sorted(set(int(round(i)) for i in np.linspace(0, len(window_df) - 1, tick_count)))
        else:
            tick_positions = [0]
        tick_labels = [pd.Timestamp(local_index[int(position)]).strftime("%m-%d %H:%M") for position in tick_positions]
        target_axis = price_ax if indicator_ax is None else indicator_ax
        target_axis.set_xticks(tick_positions)
        target_axis.set_xticklabels(tick_labels, rotation=0, color=PAPER_REPORT_THEME["muted_text"])

        fig.tight_layout()
        output_path = output_dir / (
            f"{trade['trade_id']}_{trade['side']}_{float(trade['return_pct']):+.2f}pct_{trade['exit_reason']}.png"
        )
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def plot_overview_chart(
    *,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary: dict,
    output_path: Path,
) -> Path | None:
    """Save the overview equity chart for one paper-trade report run."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so the overview chart was skipped.")
        return None

    if trades_df.empty or equity_df.empty:
        return None

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=True, height_ratios=[3, 2])
    equity_ax, pnl_ax = axes
    fig.patch.set_facecolor(PAPER_REPORT_THEME["figure_bg"])

    for axis in axes:
        axis.set_facecolor(PAPER_REPORT_THEME["axes_bg"])
        axis.tick_params(colors=PAPER_REPORT_THEME["muted_text"])
        axis.grid(alpha=0.25, color=PAPER_REPORT_THEME["grid"])
        for spine in axis.spines.values():
            spine.set_color(PAPER_REPORT_THEME["spine"])

    equity_times = pd.to_datetime(equity_df["timestamp"], utc=True)
    trade_exit_times = pd.to_datetime(trades_df["exit_time"], utc=True)
    trade_pnls = trades_df["pnl"].astype(float)
    bar_colors = [
        PAPER_REPORT_THEME["pnl_gain"] if pnl_value >= 0 else PAPER_REPORT_THEME["pnl_loss"]
        for pnl_value in trade_pnls
    ]

    equity_ax.plot(equity_times, equity_df["equity"], color=PAPER_REPORT_THEME["equity"], linewidth=1.8, label="Equity")
    equity_ax.scatter(
        equity_times,
        equity_df["equity"],
        color=PAPER_REPORT_THEME["equity"],
        s=18,
        alpha=0.9,
    )
    equity_ax.set_ylabel("Equity", color=PAPER_REPORT_THEME["text"])
    equity_ax.set_title(
        (
            f"Paper Trade Report | Trades {summary['trade_count']} | Win rate {summary['win_rate_pct']:.2f}% | "
            f"PnL {summary['total_pnl']:.2f} | Return {summary['total_return_pct']:.2f}%"
        ),
        color=PAPER_REPORT_THEME["text"],
    )
    legend = equity_ax.legend(loc="upper left")
    legend.get_frame().set_facecolor(PAPER_REPORT_THEME["axes_bg"])
    legend.get_frame().set_edgecolor(PAPER_REPORT_THEME["spine"])
    for text in legend.get_texts():
        text.set_color(PAPER_REPORT_THEME["text"])

    pnl_ax.bar(trade_exit_times, trade_pnls, color=bar_colors, width=0.01, alpha=0.9)
    pnl_ax.axhline(0, color=PAPER_REPORT_THEME["signal"], linewidth=0.8, alpha=0.6)
    pnl_ax.set_ylabel("Trade PnL", color=PAPER_REPORT_THEME["text"])
    pnl_ax.set_xlabel("Exit Time", color=PAPER_REPORT_THEME["text"])

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    pnl_ax.xaxis.set_major_locator(locator)
    pnl_ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def print_summary(
    summary: dict,
    *,
    trades_csv_path: Path,
    summary_json_path: Path,
    overview_chart_path: Path | None,
) -> None:
    """Print a concise terminal summary after a paper-trade report run."""
    filters = summary["selected_filters"]
    print("Paper trade report")
    print(f"Account filter: {filters['account'] or 'all'}")
    print(f"Strategy filter: {filters['strategy'] or 'all'}")
    print(f"Symbol filter: {filters['symbol'] or 'all'}")
    print(f"Range: {summary['analysis_start']} -> {summary['analysis_end']}")
    print()
    print(f"Trades: {summary['trade_count']}")
    print(f"Win rate: {summary['win_rate_pct']:.2f}%")
    print(f"Total PnL: {summary['total_pnl']:.2f}")
    if summary["total_return_pct"] is not None:
        print(f"Total return: {summary['total_return_pct']:.2f}%")
    print(f"Ending equity: {summary['ending_equity']:.2f}")
    print(f"Max drawdown: {summary['max_drawdown_pct']:.2f}%")
    print(f"Long trades: {summary['long_trade_count']}")
    print(f"Short trades: {summary['short_trade_count']}")
    if summary["profit_factor"] is not None:
        print(f"Profit factor: {summary['profit_factor']:.2f}")
    print()
    print(f"Trades CSV: {trades_csv_path}")
    print(f"Summary JSON: {summary_json_path}")
    if overview_chart_path is not None:
        print(f"Overview chart: {overview_chart_path}")
    trade_chart_dir = summary.get("trade_chart_dir")
    trade_chart_count = summary.get("trade_chart_count", 0)
    if trade_chart_dir:
        print(f"Trade charts: {trade_chart_dir} ({trade_chart_count})")


def run_report(args: argparse.Namespace) -> Path:
    """Build one paper-trade report and return its output root."""
    config = load_config(args.config)
    strategy_config_index = build_strategy_config_index(config)
    account_symbol_strategy_index = build_account_symbol_strategy_index(config)
    trade_log_df = load_trade_log(TRADE_LOG_PATH)
    filled_orders_df = filter_filled_orders(
        trade_log_df,
        account=args.account,
        strategy=args.strategy,
        symbol=args.symbol,
    )
    if filled_orders_df.empty:
        raise ValueError("No filled paper-trade rows matched the selected filters.")

    context_index = load_notable_context_index(NOTABLE_CONTEXT_LOG_PATH)
    start_utc = None if args.start is None else normalize_filter_timestamp(args.start, end_boundary=False)
    end_utc = None if args.end is None else normalize_filter_timestamp(args.end, end_boundary=True)
    if not args.skip_broker_history:
        try:
            broker_history_supplements_df = fetch_broker_history_supplements(
                config=config,
                filtered_orders_df=filled_orders_df,
                account_symbol_strategy_index=account_symbol_strategy_index,
                start_utc=start_utc,
                end_utc=end_utc,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not fetch Alpaca broker history. "
                "Retry with network access or rerun with --skip-broker-history for a local-log-only report."
            ) from exc
        if not broker_history_supplements_df.empty:
            filled_orders_df = pd.concat([filled_orders_df, broker_history_supplements_df], ignore_index=True, sort=False)
            filled_orders_df = filled_orders_df.sort_values(
                by=["filled_timestamp", "submitted_timestamp", "log_timestamp", "decision_id"],
                na_position="last",
            ).reset_index(drop=True)
    reconstructed_trades, reconstruction_stats = reconstruct_completed_trades(
        filled_orders_df=filled_orders_df,
        context_index=context_index,
        strategy_config_index=strategy_config_index,
        start_utc=start_utc,
        end_utc=end_utc,
    )
    if not reconstructed_trades:
        raise ValueError("No completed paper trades matched the selected filters.")

    initial_equity, initial_equity_source = estimate_initial_equity(
        reconstructed_trades,
        context_index,
        args.initial_equity,
    )
    trades_df = pd.DataFrame((asdict(trade) for trade in reconstructed_trades))
    trades_df["entry_time"] = pd.to_datetime(trades_df["entry_time"], utc=True)
    trades_df["exit_time"] = pd.to_datetime(trades_df["exit_time"], utc=True)
    trades_df["entry_signal_time"] = pd.to_datetime(trades_df["entry_signal_time"], utc=True)
    trades_df = trades_df.sort_values(by=["exit_time", "trade_id"]).reset_index(drop=True)
    equity_df = build_equity_curve(trades_df=trades_df, initial_equity=initial_equity)

    selected_filters = {
        "account": args.account,
        "strategy": args.strategy,
        "symbol": args.symbol.upper() if args.symbol else None,
        "start": args.start,
        "end": args.end,
        "chart_enabled": not args.no_chart,
        "broker_history_enabled": not args.skip_broker_history,
    }
    summary = build_paper_trade_summary(
        trades_df=trades_df,
        equity_df=equity_df,
        selected_filters=selected_filters,
        initial_equity=initial_equity,
        initial_equity_source=initial_equity_source,
        reconstruction_stats=reconstruction_stats,
        source_logs={
            "trade_log_path": str(TRADE_LOG_PATH),
            "notable_strategy_context_log_path": str(NOTABLE_CONTEXT_LOG_PATH),
        },
    )

    slug = make_report_slug(args, reconstructed_trades)
    run_timestamp = make_run_timestamp()
    run_root, trades_csv_path, summary_json_path, overview_chart_path = build_output_paths(
        args.output_dir,
        slug,
        run_timestamp,
    )

    csv_ready_df = trades_df.copy()
    for column in ("entry_time", "exit_time", "entry_signal_time"):
        csv_ready_df[column] = csv_ready_df[column].dt.strftime("%Y-%m-%dT%H:%M:%S%z")
        csv_ready_df[column] = csv_ready_df[column].str.replace("+0000", "Z", regex=False)
    csv_ready_df.to_csv(trades_csv_path, index=False)

    saved_overview_chart_path = None
    saved_trade_chart_paths: list[Path] = []
    if not args.no_chart:
        saved_overview_chart_path = plot_overview_chart(
            trades_df=trades_df,
            equity_df=equity_df,
            summary=summary,
            output_path=overview_chart_path,
        )
        prepared_data_cache = build_prepared_data_cache(reconstructed_trades)
        grouped_trade_rows: dict[str, list[dict]] = defaultdict(list)
        for trade in reconstructed_trades:
            grouped_trade_rows[trade.chart_group_key].append(asdict(trade))
        for chart_group_key, trade_rows in grouped_trade_rows.items():
            prepared_df, interval = prepared_data_cache[chart_group_key]
            group_trades_df = pd.DataFrame(trade_rows)
            saved_trade_chart_paths.extend(
                plot_paper_trade_zoom_charts(
                    prepared_df=prepared_df,
                    trades_df=group_trades_df,
                    output_dir=run_root / "trade_charts",
                    interval=interval,
                )
            )

    summary["trade_chart_count"] = len(saved_trade_chart_paths)
    summary["trade_chart_dir"] = None if not saved_trade_chart_paths else str(run_root / "trade_charts")
    summary["overview_chart_path"] = None if saved_overview_chart_path is None else str(saved_overview_chart_path)
    summary["trades_csv_path"] = str(trades_csv_path)
    summary["summary_json_path"] = str(summary_json_path)
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print_summary(
        summary,
        trades_csv_path=trades_csv_path,
        summary_json_path=summary_json_path,
        overview_chart_path=saved_overview_chart_path,
    )
    return run_root


def main() -> None:
    """Run the paper-trade report CLI."""
    args = parse_args()
    run_report(args)


if __name__ == "__main__":
    main()
