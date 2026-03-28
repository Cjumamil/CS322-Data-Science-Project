"""Maintains lightweight session helpers and live broker state snapshots."""

from typing import Any

from trading_bot.broker import get_order_by_id, get_order_status, get_position, get_protective_stop_order, get_working_orders, is_final_order_status


def get_symbol_session_state(session_state: dict, symbol: str) -> dict:
    """Return lightweight in-memory helpers for the current process only.

    Alpaca state is the operational source of truth. This local state is
    kept only for short-lived logging metadata and same-process guards.
    """
    if symbol not in session_state:
        session_state[symbol] = {
            "last_submitted_action": None,
            "last_crossover": None,
            "pending_order_id": None,
            "pending_order_action": None,
            "pending_order_reason": None,
            "pending_requested_qty": None,
            "pending_decision_price": None,
            "pending_decision_id": None,
        }
    return session_state[symbol]


def should_prevent_duplicate_trade(session_state: dict, symbol: str, action: str, event_key) -> bool:
    """Block repeated submissions for the same strategy event.

    This is only a secondary guard after broker position/order checks.
    """
    symbol_state = get_symbol_session_state(session_state, symbol)

    if symbol_state["last_crossover"] != event_key and symbol_state["pending_order_id"] is None:
        symbol_state["last_crossover"] = event_key
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
) -> None:
    """Remember the latest submitted order so the loop does not spam duplicates."""
    symbol_state = get_symbol_session_state(session_state, symbol)
    symbol_state["last_submitted_action"] = action
    symbol_state["last_crossover"] = event_key
    symbol_state["pending_order_id"] = order_id
    symbol_state["pending_order_action"] = action
    symbol_state["pending_order_reason"] = reason
    symbol_state["pending_requested_qty"] = requested_qty
    symbol_state["pending_decision_price"] = decision_price
    symbol_state["pending_decision_id"] = decision_id


def clear_pending_order(session_state: dict, symbol: str) -> None:
    """Clear the pending order fields after Alpaca reaches a final state."""
    symbol_state = get_symbol_session_state(session_state, symbol)
    symbol_state["pending_order_id"] = None
    symbol_state["pending_order_action"] = None
    symbol_state["pending_order_reason"] = None
    symbol_state["pending_requested_qty"] = None
    symbol_state["pending_decision_price"] = None
    symbol_state["pending_decision_id"] = None


def get_live_broker_state(trading_client, symbol: str) -> dict[str, Any]:
    """Collect the live Alpaca state used for operational decisions."""
    position = get_position(trading_client, symbol)
    working_orders = get_working_orders(trading_client, symbol)
    protective_stop_order = get_protective_stop_order(trading_client, symbol)
    blocking_orders = [
        order
        for order in working_orders
        if protective_stop_order is None or str(getattr(order, "id", "")) != str(getattr(protective_stop_order, "id", ""))
    ]
    return {
        "position": position,
        "in_position": position is not None,
        "entry_price": float(position.avg_entry_price) if position is not None else None,
        "working_orders": working_orders,
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
