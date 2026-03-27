"""Runs the trading bot workflow from data download through order decisions."""

import time
from typing import Optional

from alpaca.trading.enums import OrderSide

from trading_bot.broker import (
    cancel_order_by_id,
    connect_alpaca,
    get_account,
    get_order_by_id,
    get_order_status,
    get_position,
    is_final_order_status,
    market_is_open,
    submit_market_order,
    submit_stop_order,
    wait_for_order_final_state,
)
from trading_bot.data import download_data
from trading_bot.logging_utils import log_decision, log_trade
from trading_bot.risk import calculate_order_qty, should_exit_position
from trading_bot.strategy import build_strategy_dataframe, calculate_backtest_metrics


# ============================
# SETTINGS
# ============================
SYMBOL = "AAPL"
FAST_WINDOW = 20
SLOW_WINDOW = 50

# Bar interval controls the size of each candle used by the strategy.
INTERVAL = "5m"

# Lookback controls how many recent bars we download for indicator math.
# SMA-50 needs at least 50 bars, but a larger buffer gives cleaner context
# and avoids making every cycle depend on the bare minimum amount of history.
LOOKBACK_BARS = 180

# Poll frequency controls how often the bot wakes up and reevaluates.
RUN_CONTINUOUSLY = True
POLL_INTERVAL_SECONDS = 60

# After an order is submitted, poll Alpaca for a short time to see
# whether it reaches a final status in the same cycle.
ORDER_STATUS_POLL_SECONDS = 2
ORDER_STATUS_TIMEOUT_SECONDS = 30

# Risk / position controls
MAX_POSITION_QTY = 5
RISK_FRACTION_OF_BUYING_POWER = 0.02
STOP_LOSS_PCT = 0.03
TAKE_PROFIT_PCT = 0.06

# Broker-side protection is placed after a BUY fills. The loop-based exit
# logic still stays in place as a fallback if the protective order is not
# active yet or needs cleanup.
ENABLE_BROKER_SIDE_STOP_LOSS = True

# Test mode
FORCE_TEST_TRADE = False
FORCE_DIRECTION = "SELL"


def print_metrics(metrics: dict) -> None:
    """Print a small summary of backtest metrics for the latest run."""
    print("\nBacktest metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def _get_symbol_session_state(session_state: dict, symbol: str) -> dict:
    """Return the in-memory state we use for loop safety and order tracking."""
    if symbol not in session_state:
        session_state[symbol] = {
            "last_submitted_action": None,
            "last_crossover": None,
            "pending_order_id": None,
            "pending_order_action": None,
            "pending_order_reason": None,
            "pending_requested_qty": None,
            "pending_decision_price": None,
            "protective_stop_order_id": None,
        }
    return session_state[symbol]


def _should_prevent_duplicate_trade(session_state: dict, symbol: str, action: str, crossover: float) -> bool:
    """Block repeated submissions for the same crossover event."""
    symbol_state = _get_symbol_session_state(session_state, symbol)

    # Once a new crossover appears, the previous duplicate guard can reset.
    if symbol_state["last_crossover"] != crossover and symbol_state["pending_order_id"] is None:
        symbol_state["last_crossover"] = crossover
        symbol_state["last_submitted_action"] = None

    return symbol_state["last_submitted_action"] == action


def _record_submitted_action(
    session_state: dict,
    symbol: str,
    action: str,
    crossover: float,
    order_id: str,
    reason: str,
    requested_qty: int,
    decision_price: float,
) -> None:
    """Remember the latest submitted order so the loop does not spam duplicates."""
    symbol_state = _get_symbol_session_state(session_state, symbol)
    symbol_state["last_submitted_action"] = action
    symbol_state["last_crossover"] = crossover
    symbol_state["pending_order_id"] = order_id
    symbol_state["pending_order_action"] = action
    symbol_state["pending_order_reason"] = reason
    symbol_state["pending_requested_qty"] = requested_qty
    symbol_state["pending_decision_price"] = decision_price


def _clear_pending_order(session_state: dict, symbol: str) -> None:
    """Clear the pending order fields after Alpaca reaches a final state."""
    symbol_state = _get_symbol_session_state(session_state, symbol)
    symbol_state["pending_order_id"] = None
    symbol_state["pending_order_action"] = None
    symbol_state["pending_order_reason"] = None
    symbol_state["pending_requested_qty"] = None
    symbol_state["pending_decision_price"] = None


def _set_protective_stop_order(session_state: dict, symbol: str, order_id: str) -> None:
    """Remember the active broker-side stop order for this symbol."""
    symbol_state = _get_symbol_session_state(session_state, symbol)
    symbol_state["protective_stop_order_id"] = order_id


def _clear_protective_stop_order(session_state: dict, symbol: str) -> None:
    """Clear the stored protective stop reference."""
    symbol_state = _get_symbol_session_state(session_state, symbol)
    symbol_state["protective_stop_order_id"] = None


def _serialize_value(value) -> str:
    """Convert SDK values into simple CSV-friendly strings."""
    if value is None:
        return ""
    if hasattr(value, "value"):
        value = value.value
    return str(value)


def _build_trade_payload(order, intended_action: str, strategy_reason: str, requested_qty: int, decision_price: float) -> dict:
    """Build one flat trade log row from the final Alpaca order state."""
    return {
        "symbol": _serialize_value(getattr(order, "symbol", SYMBOL)),
        "intended_action": intended_action,
        "strategy_reason": strategy_reason,
        "order_id": _serialize_value(getattr(order, "id", "")),
        "client_order_id": _serialize_value(getattr(order, "client_order_id", "")),
        "order_type": _serialize_value(getattr(order, "order_type", "")),
        "requested_qty": requested_qty,
        "filled_qty": _serialize_value(getattr(order, "filled_qty", "")),
        "final_status": get_order_status(order),
        "decision_price": decision_price,
        "filled_avg_price": _serialize_value(getattr(order, "filled_avg_price", "")),
        "submitted_at": _serialize_value(getattr(order, "submitted_at", "")),
        "filled_at": _serialize_value(getattr(order, "filled_at", "")),
        "canceled_at": _serialize_value(getattr(order, "canceled_at", "")),
        "failed_at": _serialize_value(getattr(order, "failed_at", "")),
        "expired_at": _serialize_value(getattr(order, "expired_at", "")),
        "note": "",
    }


def _place_protective_stop_after_entry(trading_client, session_state: dict, symbol: str, order) -> None:
    """Place a simple stop-loss order after a long entry fills.

    This gives the position broker-side downside protection even if the
    next polling cycle is late. The loop-based exit checks still remain
    as a backup and for take-profit handling.
    """
    if not ENABLE_BROKER_SIDE_STOP_LOSS:
        return

    filled_qty = getattr(order, "filled_qty", None)
    filled_avg_price = getattr(order, "filled_avg_price", None)
    status = get_order_status(order)

    if status != "filled" or not filled_qty or not filled_avg_price:
        return

    try:
        qty = int(float(filled_qty))
        stop_price = round(float(filled_avg_price) * (1 - STOP_LOSS_PCT), 2)
        protective_order = submit_stop_order(trading_client, symbol, OrderSide.SELL, qty, stop_price)
        _set_protective_stop_order(session_state, symbol, str(getattr(protective_order, "id")))
        print(f"Protective stop-loss order submitted at {stop_price:.2f}.")
    except Exception as exc:
        print(f"Could not place protective stop-loss order: {exc}")


def _cancel_protective_stop_if_needed(trading_client, session_state: dict, symbol: str) -> None:
    """Cancel any leftover protective stop before submitting a manual sell."""
    symbol_state = _get_symbol_session_state(session_state, symbol)
    protective_order_id = symbol_state["protective_stop_order_id"]

    if not protective_order_id:
        return

    order = get_order_by_id(trading_client, protective_order_id)
    if order is None:
        return

    status = get_order_status(order)
    if is_final_order_status(status):
        _clear_protective_stop_order(session_state, symbol)
        return

    if cancel_order_by_id(trading_client, protective_order_id):
        print("Canceled existing protective stop before manual sell.")
        _clear_protective_stop_order(session_state, symbol)


def _finalize_order_logging(trading_client, session_state: dict, symbol: str, order) -> None:
    """Log the final order state once Alpaca finishes processing the order."""
    symbol_state = _get_symbol_session_state(session_state, symbol)
    log_trade(
        _build_trade_payload(
            order,
            intended_action=symbol_state["pending_order_action"],
            strategy_reason=symbol_state["pending_order_reason"],
            requested_qty=symbol_state["pending_requested_qty"],
            decision_price=symbol_state["pending_decision_price"],
        )
    )

    if symbol_state["pending_order_action"] == "BUY":
        _place_protective_stop_after_entry(trading_client, session_state, symbol, order)
    elif symbol_state["pending_order_action"] == "SELL" and get_order_status(order) == "filled":
        _clear_protective_stop_order(session_state, symbol)

    _clear_pending_order(session_state, symbol)


def _refresh_pending_order(trading_client, session_state: dict, symbol: str) -> str:
    """Check whether a previously submitted order is still pending or now final."""
    symbol_state = _get_symbol_session_state(session_state, symbol)
    pending_order_id = symbol_state["pending_order_id"]

    if pending_order_id is None:
        return "none"

    order = get_order_by_id(trading_client, pending_order_id)
    if order is None:
        return "unknown"

    status = get_order_status(order)
    print(f"Pending order {pending_order_id} status: {status}")

    if is_final_order_status(status):
        _finalize_order_logging(trading_client, session_state, symbol, order)
        return "finalized"

    return "pending"


def _submit_and_track_order(
    trading_client,
    session_state: dict,
    symbol: str,
    action: str,
    side: OrderSide,
    qty: int,
    crossover: float,
    reason: str,
    decision_price: float,
) -> None:
    """Submit an order, then poll briefly so trade logs reflect final fill data when possible."""
    order = submit_market_order(trading_client, symbol, side, qty)
    print(f"{action} ORDER SUBMITTED:")
    print(order)

    _record_submitted_action(
        session_state,
        symbol,
        action,
        crossover,
        order_id=str(getattr(order, "id")),
        reason=reason,
        requested_qty=qty,
        decision_price=decision_price,
    )

    final_order = wait_for_order_final_state(
        trading_client,
        str(getattr(order, "id")),
        poll_seconds=ORDER_STATUS_POLL_SECONDS,
        timeout_seconds=ORDER_STATUS_TIMEOUT_SECONDS,
    )

    if final_order is not None and is_final_order_status(get_order_status(final_order)):
        _finalize_order_logging(trading_client, session_state, symbol, final_order)
    else:
        # If the timeout expires, the bot keeps the order marked as pending
        # and will re-check it next cycle rather than guessing the outcome.
        print("Order is still pending. It will be checked again on the next cycle.")


def _handle_pending_order_state(trading_client, session_state: dict, decision_payload, current_market_open: bool) -> bool:
    """Handle unresolved orders before looking at fresh entry or exit signals."""
    pending_result = _refresh_pending_order(trading_client, session_state, SYMBOL)
    if pending_result == "pending":
        print("Existing order is still pending. New trade decisions are paused.")
        log_decision(decision_payload("HOLD", "pending_order_not_final", market_open=current_market_open))
        return True
    if pending_result == "finalized":
        print("Previous order finalized. Waiting for the next cycle before taking a new action.")
        return True
    if pending_result == "unknown":
        print("Could not refresh the existing order. New trade decisions are paused.")
        log_decision(decision_payload("HOLD", "pending_order_lookup_failed", market_open=current_market_open))
        return True
    return False


def _handle_exit_triggers(
    trading_client,
    session_state: dict,
    latest_close: float,
    latest_cross: float,
    in_position: bool,
    entry_price: Optional[float],
    position,
    decision_payload,
) -> bool:
    """Handle position exit logic separately from entry signal generation."""
    if not in_position:
        return False

    qty = int(float(position.qty))

    # These exit checks still depend on the latest observed bar, so they
    # are slower than true real-time order management. The broker-side
    # stop-loss helps cover part of that gap for downside protection.
    exit_reason = should_exit_position(
        latest_close,
        entry_price,
        STOP_LOSS_PCT,
        TAKE_PROFIT_PCT,
    )
    if exit_reason is None:
        return False

    if _should_prevent_duplicate_trade(session_state, SYMBOL, "SELL", latest_cross):
        print("Duplicate SELL prevented for current crossover event.")
        log_decision(decision_payload("HOLD", "duplicate_sell_prevented", qty=qty, market_open=True))
        return True

    _cancel_protective_stop_if_needed(trading_client, session_state, SYMBOL)
    log_decision(decision_payload("SELL", exit_reason, qty=qty, market_open=True))
    _submit_and_track_order(
        trading_client,
        session_state,
        SYMBOL,
        action="SELL",
        side=OrderSide.SELL,
        qty=qty,
        crossover=latest_cross,
        reason=exit_reason,
        decision_price=latest_close,
    )
    return True


def _handle_entry_signals(
    trading_client,
    session_state: dict,
    latest_close: float,
    latest_cross: float,
    buying_power: float,
    in_position: bool,
    position,
    decision_payload,
) -> bool:
    """Handle new entry or crossover-exit signals after exit checks are done."""
    # We trade on crossover events, not on a signal staying bullish or bearish.
    if latest_cross == 1.0 and not in_position:
        if _should_prevent_duplicate_trade(session_state, SYMBOL, "BUY", latest_cross):
            print("Duplicate BUY prevented for current crossover event.")
            log_decision(decision_payload("HOLD", "duplicate_buy_prevented", market_open=True))
            return True

        qty = calculate_order_qty(
            latest_close,
            buying_power,
            RISK_FRACTION_OF_BUYING_POWER,
            MAX_POSITION_QTY,
        )
        log_decision(decision_payload("BUY", "sma_crossover", qty=qty, market_open=True))
        _submit_and_track_order(
            trading_client,
            session_state,
            SYMBOL,
            action="BUY",
            side=OrderSide.BUY,
            qty=qty,
            crossover=latest_cross,
            reason="sma_crossover",
            decision_price=latest_close,
        )
        return True

    if latest_cross == -1.0 and in_position:
        if _should_prevent_duplicate_trade(session_state, SYMBOL, "SELL", latest_cross):
            print("Duplicate SELL prevented for current crossover event.")
            log_decision(decision_payload("HOLD", "duplicate_sell_prevented", market_open=True))
            return True

        qty = int(float(position.qty))
        _cancel_protective_stop_if_needed(trading_client, session_state, SYMBOL)
        log_decision(decision_payload("SELL", "sma_crossover", qty=qty, market_open=True))
        _submit_and_track_order(
            trading_client,
            session_state,
            SYMBOL,
            action="SELL",
            side=OrderSide.SELL,
            qty=qty,
            crossover=latest_cross,
            reason="sma_crossover",
            decision_price=latest_close,
        )
        return True

    return False


def run_cycle(session_state: dict) -> None:
    """Run one full bot cycle: data, signals, account checks, and decisions."""
    print("\nDownloading market data from Alpaca...")
    try:
        raw_df = download_data(SYMBOL, INTERVAL, LOOKBACK_BARS)
    except ValueError as exc:
        print(exc)
        return
    except Exception as exc:
        print(f"Error downloading data: {exc}")
        return

    df = build_strategy_dataframe(raw_df, FAST_WINDOW, SLOW_WINDOW)

    if len(df) < 2:
        print("Not enough data to compute strategy.")
        return

    print("\nRecent strategy data:")
    print(df[["Close", "SMA_FAST", "SMA_SLOW", "signal", "crossover"]].tail(8))

    metrics = calculate_backtest_metrics(df)
    print_metrics(metrics)

    latest = df.iloc[-1]
    latest_close = float(latest["Close"])
    latest_cross = float(latest["crossover"])

    print(f"\nLatest close: {latest_close:.2f}")
    print(f"Latest crossover: {latest_cross}")

    try:
        trading_client = connect_alpaca()
    except ValueError as exc:
        print(exc)
        return

    account = get_account(trading_client)
    if account is None:
        print("Could not retrieve account. Exiting.")
        return

    buying_power = float(account.buying_power)
    print(f"Account status: {account.status}")
    print(f"Buying power: {buying_power:.2f}")

    position = get_position(trading_client, SYMBOL)
    in_position = position is not None
    entry_price = float(position.avg_entry_price) if in_position else None
    print(f"Currently holding {SYMBOL}: {in_position}")

    current_market_open = market_is_open(trading_client)

    def decision_payload(action: str, reason: str, qty: int = 0, market_open: Optional[bool] = None) -> dict:
        """Build a flat decision log row with strategy and account context."""
        return {
            "symbol": SYMBOL,
            "strategy": "sma_crossover",
            "action": action,
            "reason": reason,
            "latest_close": latest_close,
            "entry_price": entry_price,
            "buying_power": buying_power,
            "in_position": in_position,
            "qty": qty,
            "sma_fast": latest["SMA_FAST"],
            "sma_slow": latest["SMA_SLOW"],
            "signal": latest["signal"],
            "crossover": latest["crossover"],
            "market_is_open": market_open,
            "force_test_trade": FORCE_TEST_TRADE,
            "force_direction": FORCE_DIRECTION,
        }

    if _handle_pending_order_state(trading_client, session_state, decision_payload, current_market_open):
        return

    if FORCE_TEST_TRADE:
        print("\nFORCE_TEST_TRADE is ON")

        if not current_market_open:
            print("Market is closed. Forced order not sent.")
            log_decision(decision_payload("HOLD", "market_closed_force_mode", market_open=False))
            return

        if FORCE_DIRECTION.upper() == "BUY":
            if in_position:
                print("Already in a position. Force BUY skipped.")
                log_decision(decision_payload("HOLD", "force_buy_skipped_in_position", market_open=True))
                return

            qty = calculate_order_qty(
                latest_close,
                buying_power,
                RISK_FRACTION_OF_BUYING_POWER,
                MAX_POSITION_QTY,
            )
            log_decision(decision_payload("BUY", "forced_test_trade", qty=qty, market_open=True))
            _submit_and_track_order(
                trading_client,
                session_state,
                SYMBOL,
                action="BUY",
                side=OrderSide.BUY,
                qty=qty,
                crossover=latest_cross,
                reason="forced_test_trade",
                decision_price=latest_close,
            )
            return

        if FORCE_DIRECTION.upper() == "SELL":
            if not in_position:
                print("No open position. Force SELL skipped.")
                log_decision(decision_payload("HOLD", "force_sell_skipped_no_position", market_open=True))
                return

            qty = int(float(position.qty))
            _cancel_protective_stop_if_needed(trading_client, session_state, SYMBOL)
            log_decision(decision_payload("SELL", "forced_test_trade", qty=qty, market_open=True))
            _submit_and_track_order(
                trading_client,
                session_state,
                SYMBOL,
                action="SELL",
                side=OrderSide.SELL,
                qty=qty,
                crossover=latest_cross,
                reason="forced_test_trade",
                decision_price=latest_close,
            )
            return

        print("FORCE_DIRECTION must be BUY or SELL.")
        log_decision(decision_payload("HOLD", "invalid_force_direction", market_open=True))
        return

    if not current_market_open:
        print("Market is closed. No live order submitted.")
        log_decision(decision_payload("HOLD", "market_closed", market_open=False))
        return

    if _handle_exit_triggers(
        trading_client,
        session_state,
        latest_close,
        latest_cross,
        in_position,
        entry_price,
        position,
        decision_payload,
    ):
        return

    if _handle_entry_signals(
        trading_client,
        session_state,
        latest_close,
        latest_cross,
        buying_power,
        in_position,
        position,
        decision_payload,
    ):
        return

    print("No trade this cycle.")
    log_decision(decision_payload("HOLD", "no_trade", market_open=True))


def run() -> None:
    """Run the bot once or keep polling until the user stops it."""
    session_state = {}

    if not RUN_CONTINUOUSLY:
        run_cycle(session_state)
        return

    print(f"Starting continuous mode. Polling every {POLL_INTERVAL_SECONDS} seconds.")

    try:
        while True:
            run_cycle(session_state)
            print(f"Sleeping for {POLL_INTERVAL_SECONDS} seconds...")
            time.sleep(POLL_INTERVAL_SECONDS)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")


if __name__ == "__main__":
    run()
