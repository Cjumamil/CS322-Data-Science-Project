"""Handles order lifecycle, protective orders, and trade execution flow."""

from alpaca.trading.enums import OrderSide

from trading_bot.broker import cancel_order_by_id, get_protective_stop_order, get_order_status, is_final_order_status, submit_market_order, submit_stop_order, wait_for_order_final_state
from trading_bot.logging_utils import log_trade
from trading_bot.state import clear_pending_order, describe_order, get_symbol_session_state, record_submitted_action, refresh_pending_order, should_prevent_duplicate_trade


def build_trade_payload(
    *,
    order,
    decision_id: str,
    bot_version: str,
    symbol: str,
    strategy_name: str,
    strategy_version: str,
    intended_action: str,
    strategy_reason: str,
    requested_qty: int,
    decision_price: float,
    serialize_value,
) -> dict:
    """Build one flat trade log row from the final Alpaca order state."""
    return {
        "decision_id": decision_id,
        "bot_version": bot_version,
        "symbol": serialize_value(getattr(order, "symbol", symbol)),
        "strategy": strategy_name,
        "strategy_version": strategy_version,
        "intended_action": intended_action,
        "strategy_reason": strategy_reason,
        "order_id": serialize_value(getattr(order, "id", "")),
        "client_order_id": serialize_value(getattr(order, "client_order_id", "")),
        "order_type": serialize_value(getattr(order, "order_type", "")),
        "requested_qty": requested_qty,
        "filled_qty": serialize_value(getattr(order, "filled_qty", "")),
        "final_status": get_order_status(order),
        "decision_price": decision_price,
        "filled_avg_price": serialize_value(getattr(order, "filled_avg_price", "")),
        "submitted_at": serialize_value(getattr(order, "submitted_at", "")),
        "filled_at": serialize_value(getattr(order, "filled_at", "")),
        "canceled_at": serialize_value(getattr(order, "canceled_at", "")),
        "failed_at": serialize_value(getattr(order, "failed_at", "")),
        "expired_at": serialize_value(getattr(order, "expired_at", "")),
        "note": "",
    }


def place_protective_stop_after_entry(
    *,
    trading_client,
    symbol: str,
    order,
    stop_loss_pct: float,
    enable_broker_side_stop_loss: bool,
) -> None:
    """Place a simple stop-loss order after a long entry fills."""
    if not enable_broker_side_stop_loss:
        return

    filled_qty = getattr(order, "filled_qty", None)
    filled_avg_price = getattr(order, "filled_avg_price", None)
    status = get_order_status(order)

    if status != "filled" or not filled_qty or not filled_avg_price:
        return

    try:
        existing_stop = get_protective_stop_order(trading_client, symbol)
        if existing_stop is not None:
            print("Protective stop-loss order already exists at Alpaca. Skipping duplicate stop placement.")
            return

        qty = int(float(filled_qty))
        stop_price = round(float(filled_avg_price) * (1 - stop_loss_pct), 2)
        submit_stop_order(trading_client, symbol, OrderSide.SELL, qty, stop_price)
        print(f"Protective stop-loss order submitted at {stop_price:.2f}.")
    except Exception as exc:
        print(f"Could not place protective stop-loss order: {exc}")


def cancel_protective_stop_if_needed(trading_client, symbol: str) -> None:
    """Cancel any live broker-side protective stop before a manual sell."""
    order = get_protective_stop_order(trading_client, symbol)
    if order is None:
        return

    order_id = str(getattr(order, "id", ""))
    if order_id and cancel_order_by_id(trading_client, order_id):
        print("Canceled existing protective stop before manual sell.")


def finalize_order_logging(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    order,
    bot_version: str,
    strategy_name: str,
    strategy_version: str,
    stop_loss_pct: float,
    enable_broker_side_stop_loss: bool,
    serialize_value,
) -> None:
    """Log the final order state once Alpaca finishes processing the order."""
    symbol_state = get_symbol_session_state(session_state, symbol)
    log_trade(
        build_trade_payload(
            order=order,
            decision_id=symbol_state["pending_decision_id"],
            bot_version=bot_version,
            symbol=symbol,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            intended_action=symbol_state["pending_order_action"],
            strategy_reason=symbol_state["pending_order_reason"],
            requested_qty=symbol_state["pending_requested_qty"],
            decision_price=symbol_state["pending_decision_price"],
            serialize_value=serialize_value,
        )
    )

    if symbol_state["pending_order_action"] == "BUY":
        place_protective_stop_after_entry(
            trading_client=trading_client,
            symbol=symbol,
            order=order,
            stop_loss_pct=stop_loss_pct,
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
        )

    clear_pending_order(session_state, symbol)


def submit_and_track_order(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    action: str,
    side: OrderSide,
    qty: int,
    event_key,
    reason: str,
    decision_price: float,
    decision_id: str,
    bot_version: str,
    strategy_name: str,
    strategy_version: str,
    order_status_poll_seconds: int,
    order_status_timeout_seconds: int,
    stop_loss_pct: float,
    enable_broker_side_stop_loss: bool,
    serialize_value,
) -> None:
    """Submit an order, then poll briefly so trade logs reflect final fill data when possible."""
    order = submit_market_order(trading_client, symbol, side, qty)
    print(f"{action} ORDER SUBMITTED:")
    print(order)

    record_submitted_action(
        session_state,
        symbol,
        action,
        event_key,
        order_id=str(getattr(order, "id")),
        reason=reason,
        requested_qty=qty,
        decision_price=decision_price,
        decision_id=decision_id,
    )

    final_order = wait_for_order_final_state(
        trading_client,
        str(getattr(order, "id")),
        poll_seconds=order_status_poll_seconds,
        timeout_seconds=order_status_timeout_seconds,
    )

    if final_order is not None and is_final_order_status(get_order_status(final_order)):
        finalize_order_logging(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            order=final_order,
            bot_version=bot_version,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            stop_loss_pct=stop_loss_pct,
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
            serialize_value=serialize_value,
        )
    else:
        print("Order is still pending. It will be checked again on the next cycle.")


def handle_pending_order_state(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    current_market_open: bool,
    live_state: dict,
    log_decision_event,
    bot_version: str,
    strategy_name: str,
    strategy_version: str,
    stop_loss_pct: float,
    enable_broker_side_stop_loss: bool,
    serialize_value,
) -> bool:
    """Handle unresolved broker orders before looking at fresh signals."""
    pending_result, pending_order = refresh_pending_order(trading_client, session_state, symbol, live_state)
    if pending_result == "pending":
        print("Existing order is still pending. New trade decisions are paused.")
        log_decision_event("HOLD", "pending_order_not_final", market_open=current_market_open)
        return True
    if pending_result == "finalized":
        finalize_order_logging(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            order=pending_order,
            bot_version=bot_version,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            stop_loss_pct=stop_loss_pct,
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
            serialize_value=serialize_value,
        )
        print("Previous order finalized. Waiting for the next cycle before taking a new action.")
        return True
    if pending_result == "unknown":
        print("Could not refresh the existing order. New trade decisions are paused.")
        log_decision_event("HOLD", "pending_order_lookup_failed", market_open=current_market_open)
        return True
    if live_state["blocking_orders"]:
        print("Broker reports working non-protective order(s) for this symbol. New trade decisions are paused.")
        for order in live_state["blocking_orders"]:
            print(f"Working order: {describe_order(order, serialize_value)}")
        log_decision_event("HOLD", "broker_open_order_exists", market_open=current_market_open)
        return True
    return False


def handle_exit_triggers(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    latest_close: float,
    event_key,
    live_state: dict,
    should_exit_position,
    stop_loss_pct: float,
    take_profit_pct: float,
    log_decision_event,
    bot_version: str,
    strategy_exit_reason: str,
    strategy_name: str,
    strategy_version: str,
    order_status_poll_seconds: int,
    order_status_timeout_seconds: int,
    enable_broker_side_stop_loss: bool,
    serialize_value,
) -> bool:
    """Handle position exits using live broker state for size and basis."""
    if not live_state["in_position"]:
        return False

    position = live_state["position"]
    entry_price = live_state["entry_price"]
    qty = int(float(position.qty))

    exit_reason = should_exit_position(
        latest_close,
        entry_price,
        stop_loss_pct,
        take_profit_pct,
    )
    if exit_reason is None:
        return False

    if should_prevent_duplicate_trade(session_state, symbol, "SELL", event_key):
        print("Duplicate SELL prevented for the current strategy event.")
        log_decision_event("HOLD", "duplicate_sell_prevented", qty=qty, market_open=True)
        return True

    cancel_protective_stop_if_needed(trading_client, symbol)
    decision_id = log_decision_event("SELL", exit_reason, qty=qty, market_open=True)
    submit_and_track_order(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        action="SELL",
        side=OrderSide.SELL,
        qty=qty,
        event_key=event_key,
        reason=strategy_exit_reason if strategy_exit_reason else exit_reason,
        decision_price=latest_close,
        decision_id=decision_id,
        bot_version=bot_version,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        order_status_poll_seconds=order_status_poll_seconds,
        order_status_timeout_seconds=order_status_timeout_seconds,
        stop_loss_pct=stop_loss_pct,
        enable_broker_side_stop_loss=enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    )
    return True


def handle_entry_signals(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    latest_close: float,
    event_key,
    buying_power: float,
    live_state: dict,
    calculate_order_qty,
    strategy_entry_signal: bool,
    strategy_exit_signal: bool,
    strategy_entry_reason: str,
    strategy_exit_reason: str,
    risk_fraction_of_buying_power: float,
    max_position_qty: int,
    log_decision_event,
    bot_version: str,
    strategy_name: str,
    strategy_version: str,
    order_status_poll_seconds: int,
    order_status_timeout_seconds: int,
    stop_loss_pct: float,
    enable_broker_side_stop_loss: bool,
    serialize_value,
) -> bool:
    """Handle strategy-driven entries and exits for the active strategy."""
    in_position = live_state["in_position"]
    position = live_state["position"]

    if strategy_entry_signal and not in_position:
        if live_state["protective_stop_order"] is not None:
            print("A broker-side protective stop still exists without a live position. New BUY is paused.")
            log_decision_event("HOLD", "stale_protective_stop_exists", market_open=True)
            return True

        if should_prevent_duplicate_trade(session_state, symbol, "BUY", event_key):
            print("Duplicate BUY prevented for the current strategy event.")
            log_decision_event("HOLD", "duplicate_buy_prevented", market_open=True)
            return True

        qty = calculate_order_qty(
            latest_close,
            buying_power,
            risk_fraction_of_buying_power,
            max_position_qty,
        )
        decision_id = log_decision_event("BUY", strategy_entry_reason, qty=qty, market_open=True)
        submit_and_track_order(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            action="BUY",
            side=OrderSide.BUY,
            qty=qty,
            event_key=event_key,
            reason=strategy_entry_reason,
            decision_price=latest_close,
            decision_id=decision_id,
            bot_version=bot_version,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            order_status_poll_seconds=order_status_poll_seconds,
            order_status_timeout_seconds=order_status_timeout_seconds,
            stop_loss_pct=stop_loss_pct,
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
            serialize_value=serialize_value,
        )
        return True

    if strategy_exit_signal and in_position:
        if should_prevent_duplicate_trade(session_state, symbol, "SELL", event_key):
            print("Duplicate SELL prevented for the current strategy event.")
            log_decision_event("HOLD", "duplicate_sell_prevented", market_open=True)
            return True

        qty = int(float(position.qty))
        cancel_protective_stop_if_needed(trading_client, symbol)
        decision_id = log_decision_event("SELL", strategy_exit_reason, qty=qty, market_open=True)
        submit_and_track_order(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            action="SELL",
            side=OrderSide.SELL,
            qty=qty,
            event_key=event_key,
            reason=strategy_exit_reason,
            decision_price=latest_close,
            decision_id=decision_id,
            bot_version=bot_version,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            order_status_poll_seconds=order_status_poll_seconds,
            order_status_timeout_seconds=order_status_timeout_seconds,
            stop_loss_pct=stop_loss_pct,
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
            serialize_value=serialize_value,
        )
        return True

    return False
