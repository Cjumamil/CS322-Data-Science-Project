"""Handles order lifecycle, protective orders, and direction-aware execution flow."""

from alpaca.trading.enums import OrderSide

from trading_bot.broker import (
    cancel_order_by_id,
    get_order_status,
    get_protective_stop_order,
    get_stop_orders,
    is_final_order_status,
    submit_market_order,
    submit_stop_order,
    wait_for_order_final_state,
)
from trading_bot.logging_utils import log_trade
from trading_bot.risk import (
    build_exit_levels_for_live_position,
    build_exit_levels_from_entry_plan,
    exit_action_for_position_side,
)
from trading_bot.state import (
    clear_active_exit_levels,
    clear_pending_order,
    describe_order,
    expected_protective_stop_side,
    get_active_exit_levels,
    get_symbol_session_state,
    record_submitted_action,
    refresh_pending_order,
    set_active_exit_levels,
    should_prevent_duplicate_trade,
)


ACTION_TO_ORDER_SIDE = {
    "BUY": OrderSide.BUY,
    "SELL": OrderSide.SELL,
}


def action_to_order_side(action: str) -> OrderSide:
    """Translate a simple bot action into the Alpaca order side enum."""
    return ACTION_TO_ORDER_SIDE[action]


def resulting_position_side(position_side_before: str, action: str) -> str:
    """Return the expected position side after the requested market order."""
    if position_side_before == "flat":
        return "long" if action == "BUY" else "short"
    if position_side_before == "long" and action == "SELL":
        return "flat"
    if position_side_before == "short" and action == "BUY":
        return "flat"
    return position_side_before


def build_trade_payload(
    *,
    order,
    decision_id: str,
    bot_version: str,
    account_name: str,
    symbol: str,
    strategy_name: str,
    strategy_version: str,
    intended_action: str,
    position_side_before: str,
    position_side_after_expected: str,
    strategy_reason: str,
    requested_qty: int,
    decision_price: float,
    serialize_value,
) -> dict:
    """Build one flat trade log row from the final Alpaca order state."""
    return {
        "decision_id": decision_id,
        "bot_version": bot_version,
        "account": account_name,
        "symbol": serialize_value(getattr(order, "symbol", symbol)),
        "strategy": strategy_name,
        "strategy_version": strategy_version,
        "intended_action": intended_action,
        "position_side_before": position_side_before,
        "position_side_after_expected": position_side_after_expected,
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


def summarize_final_order(order, serialize_value) -> str:
    """Return a one-line summary of the final broker order state."""
    symbol = serialize_value(getattr(order, "symbol", ""))
    side = serialize_value(getattr(order, "side", ""))
    status = get_order_status(order)
    filled_qty = serialize_value(getattr(order, "filled_qty", ""))
    requested_qty = serialize_value(getattr(order, "qty", ""))
    filled_avg_price = serialize_value(getattr(order, "filled_avg_price", ""))
    return (
        f"symbol={symbol} side={side} status={status} "
        f"filled_qty={filled_qty or '?'} requested_qty={requested_qty or '?'} "
        f"filled_avg_price={filled_avg_price or '?'}"
    )


def place_protective_stop_after_entry(
    *,
    trading_client,
    symbol: str,
    order,
    resulting_position_side: str,
    stop_price: float | None,
    enable_broker_side_stop_loss: bool,
) -> None:
    """Place a mirrored protective stop after a new long or short entry fills."""
    if not enable_broker_side_stop_loss or resulting_position_side not in {"long", "short"}:
        return

    filled_qty = getattr(order, "filled_qty", None)
    filled_avg_price = getattr(order, "filled_avg_price", None)
    status = get_order_status(order)

    if status != "filled" or not filled_qty or not filled_avg_price:
        return

    expected_stop_side = expected_protective_stop_side(resulting_position_side)
    if expected_stop_side is None:
        return

    try:
        existing_stop = get_protective_stop_order(
            trading_client,
            symbol,
            expected_side=expected_stop_side,
        )
        if existing_stop is not None:
            print("Protective stop-loss order already exists at Alpaca. Skipping duplicate stop placement.")
            return

        qty = int(float(filled_qty))
        if stop_price is None:
            return

        submit_stop_order(
            trading_client,
            symbol,
            action_to_order_side("SELL" if expected_stop_side == "sell" else "BUY"),
            qty,
            stop_price,
        )
        print(
            "Protective stop-loss order submitted "
            f"for {resulting_position_side} position at {stop_price:.2f}."
        )
    except Exception as exc:
        print(f"Could not place protective stop-loss order: {exc}")


def cancel_stop_orders_if_needed(trading_client, symbol: str) -> None:
    """Cancel all live stop orders for a symbol before a manual flattening trade."""
    for order in get_stop_orders(trading_client, symbol):
        order_id = str(getattr(order, "id", ""))
        if order_id and cancel_order_by_id(trading_client, order_id):
            print(f"Canceled existing stop order {order_id} before manual exit.")


def reconcile_missing_protective_stop(
    *,
    trading_client,
    symbol: str,
    live_state: dict,
    active_exit_levels: dict | None,
    enable_broker_side_stop_loss: bool,
) -> str:
    """Ensure an already-open position has a broker-side protective stop when possible."""
    if not enable_broker_side_stop_loss:
        return "broker_stop_disabled"

    position_side = live_state["position_side"]
    position_qty = live_state["position_qty"]
    if position_side not in {"long", "short"} or position_qty <= 0:
        return "not_in_position"

    if live_state["protective_stop_order"] is not None:
        return "protective_stop_present"

    if live_state["stop_orders"]:
        print("Open stop orders exist, but none matches the live position. Protective stop reconciliation skipped.")
        return "unmatched_open_stop_orders"

    if active_exit_levels is None:
        print("Could not resolve exit levels for protective stop reconciliation.")
        return "missing_exit_levels"

    stop_price = active_exit_levels.get("stop_price")
    if stop_price is None:
        print("Protective stop reconciliation skipped because no stop price was available.")
        return "missing_stop_price"

    expected_stop_side = expected_protective_stop_side(position_side)
    if expected_stop_side is None:
        return "invalid_position_side"

    try:
        submit_stop_order(
            trading_client,
            symbol,
            action_to_order_side("SELL" if expected_stop_side == "sell" else "BUY"),
            position_qty,
            float(stop_price),
        )
        print(
            "Reconciled missing protective stop-loss order "
            f"for {symbol}: side={expected_stop_side}, qty={position_qty}, stop={float(stop_price):.2f}."
        )
        return "protective_stop_reconciled"
    except Exception as exc:
        print(f"Could not reconcile protective stop-loss order: {exc}")
        return "protective_stop_reconcile_failed"


def finalize_order_logging(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    state_symbol: str | None,
    order,
    bot_version: str,
    account_name: str,
    strategy_name: str,
    strategy_version: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    enable_broker_side_stop_loss: bool,
    serialize_value,
) -> None:
    """Log the final order state once Alpaca finishes processing the order."""
    session_symbol = state_symbol or symbol
    symbol_state = get_symbol_session_state(session_state, session_symbol)
    status = get_order_status(order)
    log_trade(
        build_trade_payload(
            order=order,
            decision_id=symbol_state["pending_decision_id"],
            bot_version=bot_version,
            account_name=account_name,
            symbol=symbol,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            intended_action=symbol_state["pending_order_action"],
            position_side_before=symbol_state["pending_position_side_before"],
            position_side_after_expected=symbol_state["pending_position_side_after"],
            strategy_reason=symbol_state["pending_order_reason"],
            requested_qty=symbol_state["pending_requested_qty"],
            decision_price=symbol_state["pending_decision_price"],
            serialize_value=serialize_value,
        )
    )

    if (
        symbol_state["pending_position_side_before"] == "flat"
        and symbol_state["pending_position_side_after"] in {"long", "short"}
        and status == "filled"
    ):
        exit_levels = None
        filled_avg_price = getattr(order, "filled_avg_price", None)
        if filled_avg_price not in {None, ""}:
            exit_levels = build_exit_levels_from_entry_plan(
                entry_price=float(filled_avg_price),
                position_side=symbol_state["pending_position_side_after"],
                stop_loss_pct=stop_loss_pct,
                take_profit_pct=take_profit_pct,
                entry_risk_plan=symbol_state["pending_entry_risk_plan"],
            )
            if exit_levels is not None:
                exit_levels["entry_signal_time"] = symbol_state["pending_signal_time"]
        set_active_exit_levels(session_state, session_symbol, exit_levels)
        place_protective_stop_after_entry(
            trading_client=trading_client,
            symbol=symbol,
            order=order,
            resulting_position_side=symbol_state["pending_position_side_after"],
            stop_price=None if exit_levels is None else exit_levels["stop_price"],
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
        )
    elif symbol_state["pending_position_side_after"] == "flat" and status == "filled":
        clear_active_exit_levels(session_state, session_symbol)

    clear_pending_order(session_state, session_symbol)


def submit_and_track_order(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    action: str,
    qty: int,
    event_key,
    reason: str,
    decision_price: float,
    decision_id: str,
    position_side_before: str,
    position_side_after: str,
    bot_version: str,
    account_name: str,
    strategy_name: str,
    strategy_version: str,
    order_status_poll_seconds: int,
    order_status_timeout_seconds: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    enable_broker_side_stop_loss: bool,
    serialize_value,
    state_symbol: str | None = None,
    entry_risk_plan: dict | None = None,
    signal_time: str | None = None,
) -> None:
    """Submit an order, then poll briefly so trade logs reflect final fill data when possible."""
    session_symbol = state_symbol or symbol
    order = submit_market_order(trading_client, symbol, action_to_order_side(action), qty)
    print(f"Submitted {action} market order for {symbol}: qty={qty}")

    record_submitted_action(
        session_state,
        session_symbol,
        action,
        event_key,
        order_id=str(getattr(order, "id")),
        reason=reason,
        requested_qty=qty,
        decision_price=decision_price,
        decision_id=decision_id,
        position_side_before=position_side_before,
        position_side_after=position_side_after,
        entry_risk_plan=entry_risk_plan,
        signal_time=signal_time,
    )

    final_order = wait_for_order_final_state(
        trading_client,
        str(getattr(order, "id")),
        poll_seconds=order_status_poll_seconds,
        timeout_seconds=order_status_timeout_seconds,
    )

    if final_order is not None and is_final_order_status(get_order_status(final_order)):
        print(f"Order final state: {summarize_final_order(final_order, serialize_value)}")
        finalize_order_logging(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            state_symbol=session_symbol,
            order=final_order,
            bot_version=bot_version,
            account_name=account_name,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
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
    account_name: str,
    strategy_name: str,
    strategy_version: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    enable_broker_side_stop_loss: bool,
    serialize_value,
    state_symbol: str | None = None,
) -> tuple[str, str] | None:
    """Handle unresolved broker orders before looking at fresh signals."""
    session_symbol = state_symbol or symbol
    pending_result, pending_order = refresh_pending_order(trading_client, session_state, session_symbol, live_state)
    if pending_result == "pending":
        print("Existing order is still pending. New trade decisions are paused.")
        log_decision_event("HOLD", "pending_order_not_final", market_open=current_market_open)
        return "HOLD", "pending_order_not_final"
    if pending_result == "finalized":
        finalize_order_logging(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            state_symbol=session_symbol,
            order=pending_order,
            bot_version=bot_version,
            account_name=account_name,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            stop_loss_pct=stop_loss_pct,
            take_profit_pct=take_profit_pct,
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
            serialize_value=serialize_value,
        )
        print("Previous order finalized. Waiting for the next cycle before taking a new action.")
        return "HOLD", "previous_order_finalized_wait_next_cycle"
    if pending_result == "unknown":
        print("Could not refresh the existing order. New trade decisions are paused.")
        log_decision_event("HOLD", "pending_order_lookup_failed", market_open=current_market_open)
        return "HOLD", "pending_order_lookup_failed"
    if live_state["blocking_orders"]:
        print("Broker reports working non-protective order(s) for this symbol. New trade decisions are paused.")
        for order in live_state["blocking_orders"]:
            print(f"Working order: {describe_order(order, serialize_value)}")
        log_decision_event("HOLD", "broker_open_order_exists", market_open=current_market_open)
        return "HOLD", "broker_open_order_exists"
    return None


def handle_flatten_before_close(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    latest_close: float,
    event_key,
    live_state: dict,
    log_decision_event,
    bot_version: str,
    account_name: str,
    strategy_name: str,
    strategy_version: str,
    order_status_poll_seconds: int,
    order_status_timeout_seconds: int,
    stop_loss_pct: float,
    take_profit_pct: float,
    enable_broker_side_stop_loss: bool,
    serialize_value,
    state_symbol: str | None = None,
) -> tuple[str, str] | None:
    """Flatten any live position before the configured market close cutoff."""
    session_symbol = state_symbol or symbol
    position_side = live_state["position_side"]
    if position_side == "flat":
        log_decision_event("HOLD", "flatten_window_no_new_entries", market_open=True)
        return "HOLD", "flatten_window_no_new_entries"

    action = exit_action_for_position_side(position_side)
    qty = live_state["position_qty"]
    if action is None or qty <= 0:
        return False

    if should_prevent_duplicate_trade(session_state, session_symbol, action, event_key):
        print(f"Duplicate {action} prevented for the current strategy event.")
        log_decision_event("HOLD", f"duplicate_{action.lower()}_prevented", qty=qty, market_open=True)
        return "HOLD", f"duplicate_{action.lower()}_prevented"

    cancel_stop_orders_if_needed(trading_client, symbol)
    decision_id = log_decision_event(action, "flatten_before_close", qty=qty, market_open=True)
    submit_and_track_order(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        state_symbol=session_symbol,
        action=action,
        qty=qty,
        event_key=event_key,
        reason="flatten_before_close",
        decision_price=latest_close,
        decision_id=decision_id,
        position_side_before=position_side,
        position_side_after="flat",
        bot_version=bot_version,
        account_name=account_name,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        order_status_poll_seconds=order_status_poll_seconds,
        order_status_timeout_seconds=order_status_timeout_seconds,
        stop_loss_pct=stop_loss_pct,
        take_profit_pct=take_profit_pct,
        enable_broker_side_stop_loss=enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    )
    return action, "flatten_before_close"


def handle_exit_triggers(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    latest_close: float,
    event_key,
    live_state: dict,
    should_exit_position,
    strategy,
    risk_settings,
    log_decision_event,
    bot_version: str,
    account_name: str,
    strategy_exit_reason: str,
    position_context,
    strategy_name: str,
    strategy_version: str,
    order_status_poll_seconds: int,
    order_status_timeout_seconds: int,
    enable_broker_side_stop_loss: bool,
    serialize_value,
    state_symbol: str | None = None,
) -> tuple[str, str] | None:
    """Handle position exits using live broker state for size and basis."""
    session_symbol = state_symbol or symbol
    position_side = live_state["position_side"]
    if position_side == "flat":
        return None

    qty = live_state["position_qty"]
    entry_price = live_state["entry_price"]
    exit_levels = build_exit_levels_for_live_position(
        entry_price=entry_price,
        position_side=position_side,
        stop_loss_pct=risk_settings.stop_loss_pct,
        take_profit_pct=risk_settings.take_profit_pct,
        protective_stop_order=live_state["protective_stop_order"],
        active_exit_levels=get_active_exit_levels(session_state, session_symbol),
        risk_reward_multiple=strategy.risk_reward_multiple(),
    )
    if exit_levels is None:
        print("Could not resolve stop-loss / take-profit levels for the live position.")
        return None

    exit_reason = should_exit_position(
        latest_close,
        position_side,
        exit_levels["stop_price"],
        exit_levels["take_profit_price"],
    )
    if exit_reason is None:
        return None

    action = exit_action_for_position_side(position_side)
    if action is None:
        return None

    if should_prevent_duplicate_trade(session_state, session_symbol, action, event_key):
        print(f"Duplicate {action} prevented for the current strategy event.")
        log_decision_event("HOLD", f"duplicate_{action.lower()}_prevented", qty=qty, market_open=True)
        return "HOLD", f"duplicate_{action.lower()}_prevented"

    cancel_stop_orders_if_needed(trading_client, symbol)
    decision_id = log_decision_event(action, exit_reason, qty=qty, market_open=True)
    submit_and_track_order(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        state_symbol=session_symbol,
        action=action,
        qty=qty,
        event_key=event_key,
        reason=exit_reason,
        decision_price=latest_close,
        decision_id=decision_id,
        position_side_before=position_side,
        position_side_after="flat",
        bot_version=bot_version,
        account_name=account_name,
        strategy_name=strategy_name,
        strategy_version=strategy_version,
        order_status_poll_seconds=order_status_poll_seconds,
        order_status_timeout_seconds=order_status_timeout_seconds,
        stop_loss_pct=risk_settings.stop_loss_pct,
        take_profit_pct=risk_settings.take_profit_pct,
        enable_broker_side_stop_loss=enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    )
    return action, exit_reason


def handle_strategy_actions(
    *,
    trading_client,
    session_state: dict,
    symbol: str,
    latest_close: float,
    event_key,
    buying_power: float,
    live_state: dict,
    calculate_order_qty,
    strategy,
    latest_row,
    strategy_entry_action: str | None,
    strategy_exit_action: str | None,
    strategy_entry_reason: str,
    strategy_exit_reason: str,
    risk_settings,
    log_decision_event,
    bot_version: str,
    account_name: str,
    strategy_name: str,
    strategy_version: str,
    order_status_poll_seconds: int,
    order_status_timeout_seconds: int,
    enable_broker_side_stop_loss: bool,
    serialize_value,
    state_symbol: str | None = None,
    signal_time: str | None = None,
) -> tuple[str, str] | None:
    """Handle direction-aware strategy entries and exits for the active strategy."""
    session_symbol = state_symbol or symbol
    position_side = live_state["position_side"]
    qty = live_state["position_qty"]

    if position_side == "flat":
        if strategy_entry_action not in {"BUY", "SELL"}:
            return None

        if live_state["stop_orders"]:
            print("One or more stop orders still exist without a live position. New entry is paused.")
            log_decision_event("HOLD", "stale_protective_stop_exists", market_open=True)
            return "HOLD", "stale_protective_stop_exists"

        if should_prevent_duplicate_trade(session_state, session_symbol, strategy_entry_action, event_key):
            print(f"Duplicate {strategy_entry_action} prevented for the current strategy event.")
            log_decision_event(
                "HOLD",
                f"duplicate_{strategy_entry_action.lower()}_prevented",
                market_open=True,
            )
            return "HOLD", f"duplicate_{strategy_entry_action.lower()}_prevented"

        entry_qty = calculate_order_qty(
            latest_close,
            buying_power,
            risk_settings.risk_fraction_of_buying_power,
            risk_settings.max_position_qty,
        )
        resulting_side = resulting_position_side(position_side, strategy_entry_action)
        entry_risk_plan = strategy.build_entry_risk_plan(latest_row, resulting_side, risk_settings)
        preview_exit_levels = build_exit_levels_from_entry_plan(
            entry_price=latest_close,
            position_side=resulting_side,
            stop_loss_pct=risk_settings.stop_loss_pct,
            take_profit_pct=risk_settings.take_profit_pct,
            entry_risk_plan=entry_risk_plan,
        )
        if preview_exit_levels is None:
            print("Entry skipped because the strategy could not build valid exit levels.")
            rejection_reason = "invalid_entry_risk_plan"
            if entry_risk_plan is not None and entry_risk_plan.get("rejection_reason"):
                rejection_reason = str(entry_risk_plan["rejection_reason"])
            log_decision_event("HOLD", rejection_reason, market_open=True)
            return "HOLD", rejection_reason

        decision_id = log_decision_event(strategy_entry_action, strategy_entry_reason, qty=entry_qty, market_open=True)
        submit_and_track_order(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            state_symbol=session_symbol,
            action=strategy_entry_action,
            qty=entry_qty,
            event_key=event_key,
            reason=strategy_entry_reason,
            decision_price=latest_close,
            decision_id=decision_id,
            position_side_before="flat",
            position_side_after=resulting_side,
            bot_version=bot_version,
            account_name=account_name,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            order_status_poll_seconds=order_status_poll_seconds,
            order_status_timeout_seconds=order_status_timeout_seconds,
            stop_loss_pct=risk_settings.stop_loss_pct,
            take_profit_pct=risk_settings.take_profit_pct,
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
            serialize_value=serialize_value,
            entry_risk_plan=entry_risk_plan,
            signal_time=signal_time,
        )
        return strategy_entry_action, strategy_entry_reason

    if position_side == "long" and strategy_exit_action == "SELL":
        if should_prevent_duplicate_trade(session_state, session_symbol, "SELL", event_key):
            print("Duplicate SELL prevented for the current strategy event.")
            log_decision_event("HOLD", "duplicate_sell_prevented", market_open=True)
            return "HOLD", "duplicate_sell_prevented"

        cancel_stop_orders_if_needed(trading_client, symbol)
        decision_id = log_decision_event("SELL", strategy_exit_reason, qty=qty, market_open=True)
        submit_and_track_order(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            state_symbol=session_symbol,
            action="SELL",
            qty=qty,
            event_key=event_key,
            reason=strategy_exit_reason,
            decision_price=latest_close,
            decision_id=decision_id,
            position_side_before="long",
            position_side_after="flat",
            bot_version=bot_version,
            account_name=account_name,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            order_status_poll_seconds=order_status_poll_seconds,
            order_status_timeout_seconds=order_status_timeout_seconds,
            stop_loss_pct=risk_settings.stop_loss_pct,
            take_profit_pct=risk_settings.take_profit_pct,
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
            serialize_value=serialize_value,
        )
        return "SELL", strategy_exit_reason

    if position_side == "short" and strategy_exit_action == "BUY":
        if should_prevent_duplicate_trade(session_state, session_symbol, "BUY", event_key):
            print("Duplicate BUY prevented for the current strategy event.")
            log_decision_event("HOLD", "duplicate_buy_prevented", market_open=True)
            return "HOLD", "duplicate_buy_prevented"

        cancel_stop_orders_if_needed(trading_client, symbol)
        decision_id = log_decision_event("BUY", strategy_exit_reason, qty=qty, market_open=True)
        submit_and_track_order(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            state_symbol=session_symbol,
            action="BUY",
            qty=qty,
            event_key=event_key,
            reason=strategy_exit_reason,
            decision_price=latest_close,
            decision_id=decision_id,
            position_side_before="short",
            position_side_after="flat",
            bot_version=bot_version,
            account_name=account_name,
            strategy_name=strategy_name,
            strategy_version=strategy_version,
            order_status_poll_seconds=order_status_poll_seconds,
            order_status_timeout_seconds=order_status_timeout_seconds,
            stop_loss_pct=risk_settings.stop_loss_pct,
            take_profit_pct=risk_settings.take_profit_pct,
            enable_broker_side_stop_loss=enable_broker_side_stop_loss,
            serialize_value=serialize_value,
        )
        return "BUY", strategy_exit_reason

    return None
