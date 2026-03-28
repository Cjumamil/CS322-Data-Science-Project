"""Builds stable decision log rows and flexible strategy context entries."""

from typing import Optional

from trading_bot.logging_utils import log_decision, log_strategy_context, make_event_id


def serialize_value(value) -> str:
    """Convert SDK values into simple log-friendly strings."""
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value)


def build_strategy_metadata(strategy_name: str, strategy_version: str, interval: str) -> dict:
    """Describe the currently active strategy as one configurable option."""
    return {
        "name": strategy_name,
        "version": strategy_version,
        "interval": interval,
    }


def build_decision_payload(
    *,
    bot_version: str,
    symbol: str,
    strategy_metadata: dict,
    latest_close: float,
    entry_price,
    buying_power: float,
    in_position: bool,
    force_test_trade: bool,
    force_direction: str,
    action: str,
    reason: str,
    qty: int = 0,
    market_open: Optional[bool] = None,
) -> dict:
    """Build a flat, strategy-agnostic decision log row."""
    return {
        "bot_version": bot_version,
        "symbol": symbol,
        "strategy": strategy_metadata["name"],
        "strategy_version": strategy_metadata["version"],
        "interval": strategy_metadata["interval"],
        "action": action,
        "reason": reason,
        "latest_close": latest_close,
        "entry_price": entry_price,
        "buying_power": buying_power,
        "in_position": in_position,
        "qty": qty,
        "market_is_open": market_open,
        "force_test_trade": force_test_trade,
        "force_direction": force_direction,
    }


def build_strategy_context_payload(
    *,
    decision_id: str,
    bot_version: str,
    symbol: str,
    strategy_metadata: dict,
    action: str,
    reason: str,
    market_open: Optional[bool],
    qty: int,
    position,
    entry_price,
    buying_power: float,
    account_status,
    latest_close: float,
    latest_signal_value: float,
    lookback_bars: int,
    raw_bar_count: int,
    strategy_row_count: int,
    bar_start,
    bar_end,
    strategy_parameters: dict,
    strategy_signals: dict,
    stop_loss_pct: float,
    take_profit_pct: float,
    risk_fraction_of_buying_power: float,
    max_position_qty: int,
    live_state: dict,
) -> dict:
    """Build flexible strategy/context data without forcing CSV schema changes."""
    return {
        "decision_id": decision_id,
        "bot_version": bot_version,
        "symbol": symbol,
        "strategy": strategy_metadata["name"],
        "strategy_version": strategy_metadata["version"],
        "interval": strategy_metadata["interval"],
        "action": action,
        "reason": reason,
        "market_is_open": market_open,
        "position": {
            "in_position": position is not None,
            "qty": float(position.qty) if position is not None else 0,
            "entry_price": entry_price,
        },
        "account": {
            "buying_power": buying_power,
            "status": serialize_value(account_status),
        },
        "market_data": {
            "latest_close": latest_close,
            "latest_signal_value": latest_signal_value,
            "bar_interval": strategy_metadata["interval"],
            "lookback_bars": lookback_bars,
            "raw_bar_count": raw_bar_count,
            "strategy_row_count": strategy_row_count,
            "bar_start": bar_start.isoformat(),
            "bar_end": bar_end.isoformat(),
        },
        "strategy_metadata": strategy_metadata,
        "strategy_parameters": {
            **strategy_parameters,
            "stop_loss_pct": stop_loss_pct,
            "take_profit_pct": take_profit_pct,
            "risk_fraction_of_buying_power": risk_fraction_of_buying_power,
            "max_position_qty": max_position_qty,
        },
        "strategy_signals": strategy_signals,
        "broker_state": {
            "working_order_count": len(live_state["working_orders"]),
            "blocking_order_count": len(live_state["blocking_orders"]),
            "protective_stop_order_id": serialize_value(
                getattr(live_state["protective_stop_order"], "id", "")
            ),
            "protective_stop_price": serialize_value(
                getattr(live_state["protective_stop_order"], "stop_price", "")
            ),
        },
        "decision": {
            "qty": qty,
        },
    }


def log_decision_event(*, decision_row: dict, strategy_context_row: dict) -> str:
    """Write the stable decision log and flexible strategy-context log together."""
    decision_id = make_event_id()
    row = dict(decision_row)
    context = dict(strategy_context_row)
    row["decision_id"] = decision_id
    context["decision_id"] = decision_id
    log_decision(row)
    log_strategy_context(context)
    return decision_id
