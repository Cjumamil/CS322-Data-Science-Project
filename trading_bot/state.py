"""Maintains lightweight session helpers and live broker state snapshots."""

from typing import Any

from trading_bot.broker import (
    get_order_by_id,
    get_order_status,
    get_position,
    get_position_qty,
    get_position_side,
    get_protective_stop_order,
    get_stop_orders,
    get_working_orders,
    is_final_order_status,
)


def get_symbol_session_state(session_state: dict, symbol: str) -> dict:
    """Return lightweight in-memory helpers for the current process only.

    Alpaca state is the operational source of truth. This local state is
    kept only for short-lived logging metadata and same-process guards.
    """
    if symbol not in session_state:
        session_state[symbol] = {
            "last_submitted_action": None,
            "last_strategy_event_key": None,
            "pending_order_id": None,
            "pending_order_action": None,
            "pending_order_reason": None,
            "pending_requested_qty": None,
            "pending_decision_price": None,
            "pending_decision_id": None,
            "pending_position_side_before": "flat",
            "pending_position_side_after": "flat",
            "pending_entry_risk_plan": None,
            "pending_signal_time": None,
            "active_entry_signal_time": None,
            "active_stop_price": None,
            "active_take_profit_price": None,
            "active_stop_source": None,
            "active_risk_reward_multiple": None,
            "active_stop_distance": None,
            "active_stop_distance_frac_of_price": None,
            "active_max_stop_distance_frac_of_price": None,
        }
    return session_state[symbol]


def should_prevent_duplicate_trade(session_state: dict, symbol: str, action: str, event_key) -> bool:
    """Block repeated submissions for the same strategy event.

    This is only a secondary guard after broker position/order checks.
    """
    symbol_state = get_symbol_session_state(session_state, symbol)

    if symbol_state["last_strategy_event_key"] != event_key and symbol_state["pending_order_id"] is None:
        symbol_state["last_strategy_event_key"] = event_key
        symbol_state["last_submitted_action"] = None

    return symbol_state["last_submitted_action"] == action


def record_submitted_action(
    session_state: dict,
    symbol: str,
    action: str,
    event_key,
    order_id: str,
    reason: str,
    requested_qty: int,
    decision_price: float,
    decision_id: str,
    position_side_before: str,
    position_side_after: str,
    entry_risk_plan: dict | None = None,
    signal_time: str | None = None,
) -> None:
    """Remember the latest submitted order so the loop does not spam duplicates."""
    symbol_state = get_symbol_session_state(session_state, symbol)
    symbol_state["last_submitted_action"] = action
    symbol_state["last_strategy_event_key"] = event_key
    symbol_state["pending_order_id"] = order_id
    symbol_state["pending_order_action"] = action
    symbol_state["pending_order_reason"] = reason
    symbol_state["pending_requested_qty"] = requested_qty
    symbol_state["pending_decision_price"] = decision_price
    symbol_state["pending_decision_id"] = decision_id
    symbol_state["pending_position_side_before"] = position_side_before
    symbol_state["pending_position_side_after"] = position_side_after
    symbol_state["pending_entry_risk_plan"] = entry_risk_plan
    symbol_state["pending_signal_time"] = signal_time


def clear_pending_order(session_state: dict, symbol: str) -> None:
    """Clear the pending order fields after Alpaca reaches a final state."""
    symbol_state = get_symbol_session_state(session_state, symbol)
    symbol_state["pending_order_id"] = None
    symbol_state["pending_order_action"] = None
    symbol_state["pending_order_reason"] = None
    symbol_state["pending_requested_qty"] = None
    symbol_state["pending_decision_price"] = None
    symbol_state["pending_decision_id"] = None
    symbol_state["pending_position_side_before"] = "flat"
    symbol_state["pending_position_side_after"] = "flat"
    symbol_state["pending_entry_risk_plan"] = None
    symbol_state["pending_signal_time"] = None


def set_active_exit_levels(session_state: dict, symbol: str, exit_levels: dict | None) -> None:
    """Remember the active stop-loss and take-profit prices for one live trade."""
    symbol_state = get_symbol_session_state(session_state, symbol)
    if exit_levels is None:
        symbol_state["active_stop_price"] = None
        symbol_state["active_take_profit_price"] = None
        symbol_state["active_stop_source"] = None
        symbol_state["active_risk_reward_multiple"] = None
        symbol_state["active_entry_signal_time"] = None
        symbol_state["active_stop_distance"] = None
        symbol_state["active_stop_distance_frac_of_price"] = None
        symbol_state["active_max_stop_distance_frac_of_price"] = None
        return

    symbol_state["active_stop_price"] = exit_levels.get("stop_price")
    symbol_state["active_take_profit_price"] = exit_levels.get("take_profit_price")
    symbol_state["active_stop_source"] = exit_levels.get("stop_source")
    symbol_state["active_risk_reward_multiple"] = exit_levels.get("risk_reward_multiple")
    symbol_state["active_entry_signal_time"] = exit_levels.get("entry_signal_time")
    symbol_state["active_stop_distance"] = exit_levels.get("stop_distance")
    symbol_state["active_stop_distance_frac_of_price"] = exit_levels.get("stop_distance_frac_of_price")
    symbol_state["active_max_stop_distance_frac_of_price"] = exit_levels.get("max_stop_distance_frac_of_price")


def clear_active_exit_levels(session_state: dict, symbol: str) -> None:
    """Forget the current trade's exit levels after the position is gone."""
    set_active_exit_levels(session_state, symbol, None)


def get_active_exit_levels(session_state: dict, symbol: str) -> dict | None:
    """Return any remembered stop-loss and take-profit levels for a live trade."""
    symbol_state = get_symbol_session_state(session_state, symbol)
    stop_price = symbol_state["active_stop_price"]
    take_profit_price = symbol_state["active_take_profit_price"]

    if stop_price is None and take_profit_price is None:
        return None

    return {
        "stop_price": stop_price,
        "take_profit_price": take_profit_price,
        "stop_source": symbol_state["active_stop_source"],
        "risk_reward_multiple": symbol_state["active_risk_reward_multiple"],
        "entry_signal_time": symbol_state["active_entry_signal_time"],
        "stop_distance": symbol_state["active_stop_distance"],
        "stop_distance_frac_of_price": symbol_state["active_stop_distance_frac_of_price"],
        "max_stop_distance_frac_of_price": symbol_state["active_max_stop_distance_frac_of_price"],
    }


def expected_protective_stop_side(position_side: str) -> str | None:
    """Return the stop side that would protect the current position."""
    if position_side == "long":
        return "sell"
    if position_side == "short":
        return "buy"
    return None


def get_live_broker_state(trading_client, symbol: str) -> dict[str, Any]:
    """Collect the live Alpaca state used for operational decisions."""
    position = get_position(trading_client, symbol)
    working_orders = get_working_orders(trading_client, symbol)
    position_side = get_position_side(position)
    position_qty = get_position_qty(position)
    stop_orders = get_stop_orders(trading_client, symbol)
    stop_order_ids = {str(getattr(stop_order, "id", "")) for stop_order in stop_orders}
    protective_stop_order = get_protective_stop_order(
        trading_client,
        symbol,
        expected_side=expected_protective_stop_side(position_side),
    )
    blocking_orders = [
        order
        for order in working_orders
        if str(getattr(order, "id", "")) not in stop_order_ids
    ]
    return {
        "position": position,
        "position_side": position_side,
        "position_qty": position_qty,
        "signed_position_qty": position_qty if position_side == "long" else -position_qty if position_side == "short" else 0,
        "is_long_position": position_side == "long",
        "is_short_position": position_side == "short",
        "is_flat": position_side == "flat",
        "in_position": position_side in {"long", "short"},
        "entry_price": float(position.avg_entry_price) if position is not None else None,
        "working_orders": working_orders,
        "stop_orders": stop_orders,
        "protective_stop_order": protective_stop_order,
        "blocking_orders": blocking_orders,
    }


def describe_order(order, serialize_value, get_order_status_fn=get_order_status) -> str:
    """Return a short readable summary of an Alpaca order."""
    side = serialize_value(getattr(order, "side", ""))
    order_type = serialize_value(getattr(order, "order_type", getattr(order, "type", "")))
    status = get_order_status_fn(order)
    order_id = serialize_value(getattr(order, "id", ""))
    return f"{side} {order_type} {status} ({order_id})"


def refresh_pending_order(trading_client, session_state: dict, symbol: str, live_state: dict) -> tuple[str, Any | None]:
    """Check whether a locally tracked order is still pending or now final."""
    symbol_state = get_symbol_session_state(session_state, symbol)
    pending_order_id = symbol_state["pending_order_id"]

    if pending_order_id is None:
        return "none", None

    order = get_order_by_id(trading_client, pending_order_id)
    if order is None:
        broker_has_pending = any(str(getattr(item, "id", "")) == pending_order_id for item in live_state["blocking_orders"])
        if not broker_has_pending:
            print("Clearing stale local pending-order tracking because Alpaca shows no matching working order.")
            clear_pending_order(session_state, symbol)
            return "cleared", None
        return "unknown", None

    status = get_order_status(order)
    print(f"Pending order {pending_order_id} status: {status}")

    if is_final_order_status(status):
        return "finalized", order

    return "pending", order
