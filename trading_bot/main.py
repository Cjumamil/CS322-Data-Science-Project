"""Runs the trading bot workflow from data download through order decisions."""

import time
import pandas as pd

from trading_bot.broker import connect_alpaca, get_account, get_market_clock, market_is_open
from trading_bot.config import BotConfig, SymbolAssignment, load_config
from trading_bot.data import download_data
from trading_bot.decision_logging import (
    build_decision_payload,
    build_strategy_context_payload,
    build_strategy_metadata,
    log_decision_event,
    serialize_value,
)
from trading_bot.execution import (
    handle_flatten_before_close,
    handle_exit_triggers,
    handle_pending_order_state,
    submit_and_track_order,
    cancel_stop_orders_if_needed,
    handle_strategy_actions,
)
from trading_bot.risk import (
    build_exit_levels_for_live_position,
    build_exit_levels_from_entry_plan,
    calculate_order_qty,
    is_within_flatten_before_close_window,
    should_exit_position,
)
from trading_bot.strategies.base import PositionContext
from trading_bot.state import get_active_exit_levels, get_live_broker_state


def print_metrics(metrics: dict) -> None:
    """Print a small summary of backtest metrics for the latest run."""
    print("\nBacktest metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def build_live_position_context(df, live_state: dict, active_exit_levels: dict | None) -> PositionContext | None:
    """Build position context for strategy-aware exits from live bot state."""
    position_side = live_state["position_side"]
    if position_side not in {"long", "short"}:
        return None

    entry_signal_time = None if active_exit_levels is None else active_exit_levels.get("entry_signal_time")
    bars_in_trade = None
    if entry_signal_time:
        try:
            entry_signal_timestamp = pd.Timestamp(entry_signal_time)
            bars_in_trade = int((df.index > entry_signal_timestamp).sum())
        except Exception:
            bars_in_trade = None

    return PositionContext(
        side=position_side,
        entry_price=live_state["entry_price"],
        bars_in_trade=bars_in_trade,
        entry_signal_time=entry_signal_time,
    )


def run_symbol_cycle(session_state: dict, trading_client, config: BotConfig, symbol_config: SymbolAssignment) -> None:
    """Run one full bot cycle for a single configured symbol."""
    # Get the current symbol and its assigned strategy from the config.
    symbol = symbol_config.ticker
    strategy = symbol_config.strategy
    strategy_metadata = build_strategy_metadata(
        strategy.name,
        strategy.version,
        strategy.interval,
    )

    # Download recent market data needed to evaluate the strategy.
    print(f"\n=== {symbol} | {strategy.name}@{strategy.version} ===")
    print("\nDownloading market data from Alpaca...")
    try:
        raw_df = download_data(symbol, strategy.interval, strategy.lookback_bars)
    except ValueError as exc:
        print(exc)
        return
    except Exception as exc:
        print(f"Error downloading data: {exc}")
        return

    min_required_bars = strategy.min_required_bars()
    print(f"Downloaded {len(raw_df)} raw bars for {symbol} at {strategy.interval}.")

    # Stop this cycle if Alpaca returned no market data.
    if raw_df.empty:
        print("Alpaca returned no bars for this request.")
        return

    # Record the time range covered by the downloaded bars.
    first_bar_timestamp = raw_df.index.min()
    last_bar_timestamp = raw_df.index.max()
    print(
        "Bar time range: "
        f"{first_bar_timestamp.isoformat()} -> {last_bar_timestamp.isoformat()}"
    )

    # Prepare the raw price data so the strategy can evaluate its signals.
    df = strategy.prepare_dataframe(raw_df)
    print(
        "Prepared "
        f"{len(df)} strategy rows after warmup "
        f"(minimum raw bars needed: {min_required_bars})."
    )

    # Stop this cycle if there are not enough prepared rows to evaluate the strategy.
    if len(df) < 2:
        print(
            "Not enough data to compute strategy. "
            f"Need at least {min_required_bars} raw bars for {strategy.name}, "
            f"but received {len(raw_df)}."
        )
        return

    # Show the most recent strategy values and summary performance metrics.
    print("\nRecent strategy data:")
    print(df[strategy.recent_display_columns()].tail(8))

    metrics = strategy.calculate_backtest_metrics(df)
    print_metrics(metrics)

    # Pull the latest row so the bot can inspect the current signal state.
    latest = df.iloc[-1]
    latest_close = float(latest["Close"])
    latest_event_key = strategy.event_key(latest)
    latest_signal_value = strategy.latest_signal_value(latest)

    print(f"\nLatest close: {latest_close:.2f}")
    print(f"{strategy.latest_signal_label()}: {latest_signal_value}")

    # Get the account so position sizing can use live buying power.
    account = get_account(trading_client)
    if account is None:
        print("Could not retrieve account. Exiting.")
        return

    buying_power = float(account.buying_power)
    print(f"Account status: {account.status}")
    print(f"Buying power: {buying_power:.2f}")

    # Check the symbol's current live broker state before making a new decision.
    live_state = get_live_broker_state(trading_client, symbol)
    in_position = live_state["in_position"]
    position_side = live_state["position_side"]
    position_qty = live_state["position_qty"]
    entry_price = live_state["entry_price"]
    active_exit_levels = None
    if position_side in {"long", "short"} and entry_price is not None:
        active_exit_levels = build_exit_levels_for_live_position(
            entry_price=entry_price,
            position_side=position_side,
            stop_loss_pct=config.risk.stop_loss_pct,
            take_profit_pct=config.risk.take_profit_pct,
            protective_stop_order=live_state["protective_stop_order"],
            active_exit_levels=get_active_exit_levels(session_state, symbol),
            risk_reward_multiple=strategy.risk_reward_multiple(),
        )
    position_context = build_live_position_context(df, live_state, active_exit_levels)
    strategy_entry_action = strategy.entry_action(latest)
    strategy_exit_action = strategy.exit_action(latest, position_context)
    strategy_exit_reason = strategy.exit_reason(latest, position_context)
    print(f"Current position for {symbol}: {position_side} (qty={position_qty})")
    if live_state["protective_stop_order"] is not None:
        stop_price = serialize_value(getattr(live_state["protective_stop_order"], "stop_price", ""))
        stop_side = serialize_value(getattr(live_state["protective_stop_order"], "side", ""))
        print(f"Protective stop currently active at Alpaca: side={stop_side}, stop={stop_price}")
    elif live_state["stop_orders"]:
        print(f"Open stop orders at Alpaca without an active matching position stop: {len(live_state['stop_orders'])}")
    if live_state["blocking_orders"]:
        print(f"Non-protective working orders at Alpaca: {len(live_state['blocking_orders'])}")

    # Check whether the market is currently open before allowing live orders.
    market_clock = get_market_clock(trading_client)
    current_market_open = bool(market_clock.is_open) if market_clock is not None else market_is_open(trading_client)
    flatten_window_active = is_within_flatten_before_close_window(
        None if market_clock is None else getattr(market_clock, "timestamp", None),
        flatten_before_close=config.risk.flatten_before_close,
        flatten_minutes_before_close=config.risk.flatten_minutes_before_close,
    )

    # Build the decision and strategy-context log rows for this cycle.
    def log_decision_for_cycle(action: str, reason: str, qty: int = 0, market_open=None) -> str:
        decision_row = build_decision_payload(
            bot_version=config.bot.version,
            symbol=symbol,
            strategy_metadata=strategy_metadata,
            latest_close=latest_close,
            entry_price=entry_price,
            buying_power=buying_power,
            in_position=in_position,
            position_side=position_side,
            position_qty=position_qty,
            force_test_trade=config.execution.force_test_trade,
            force_direction=config.execution.force_direction,
            action=action,
            reason=reason,
            qty=qty,
            market_open=market_open,
        )
        strategy_context_row = build_strategy_context_payload(
            decision_id="",
            bot_version=config.bot.version,
            symbol=symbol,
            strategy_metadata=strategy_metadata,
            action=action,
            reason=reason,
            market_open=market_open,
            qty=qty,
            entry_price=entry_price,
            buying_power=buying_power,
            account_status=account.status,
            latest_close=latest_close,
            latest_signal_value=latest_signal_value,
            lookback_bars=strategy.lookback_bars,
            raw_bar_count=len(raw_df),
            strategy_row_count=len(df),
            bar_start=first_bar_timestamp,
            bar_end=last_bar_timestamp,
            strategy_parameters=strategy.build_strategy_parameters(),
            strategy_signals={
                **strategy.build_strategy_signals(latest, position_context),
                "resolved_entry_action": strategy_entry_action,
                "resolved_exit_action": strategy_exit_action,
                "resolved_exit_reason": strategy_exit_reason,
                "active_stop_price": None if active_exit_levels is None else active_exit_levels["stop_price"],
                "active_take_profit_price": None if active_exit_levels is None else active_exit_levels["take_profit_price"],
                "active_stop_source": None if active_exit_levels is None else active_exit_levels["stop_source"],
                "active_risk_reward_multiple": None
                if active_exit_levels is None
                else active_exit_levels["risk_reward_multiple"],
                "active_stop_distance": None
                if active_exit_levels is None
                else active_exit_levels["stop_distance"],
                "active_stop_distance_frac_of_price": None
                if active_exit_levels is None
                else active_exit_levels["stop_distance_frac_of_price"],
                "active_max_stop_distance_frac_of_price": None
                if active_exit_levels is None
                else active_exit_levels["max_stop_distance_frac_of_price"],
                "active_entry_signal_time": None
                if active_exit_levels is None
                else active_exit_levels["entry_signal_time"],
            },
            stop_loss_pct=config.risk.stop_loss_pct,
            take_profit_pct=config.risk.take_profit_pct,
            risk_fraction_of_buying_power=config.risk.risk_fraction_of_buying_power,
            max_position_qty=config.risk.max_position_qty,
            flatten_before_close=config.risk.flatten_before_close,
            flatten_minutes_before_close=config.risk.flatten_minutes_before_close,
            live_state=live_state,
        )
        return log_decision_event(decision_row=decision_row, strategy_context_row=strategy_context_row)

    # Resolve any previously submitted order before evaluating fresh trade signals.
    if handle_pending_order_state(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        current_market_open=current_market_open,
        live_state=live_state,
        log_decision_event=log_decision_for_cycle,
        bot_version=config.bot.version,
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        stop_loss_pct=config.risk.stop_loss_pct,
        take_profit_pct=config.risk.take_profit_pct,
        enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    ):
        return

    if current_market_open and flatten_window_active:
        print("Inside the configured pre-close flatten window.")
        if handle_flatten_before_close(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            latest_close=latest_close,
            event_key=latest_event_key,
            live_state=live_state,
            log_decision_event=log_decision_for_cycle,
            bot_version=config.bot.version,
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            order_status_poll_seconds=config.execution.order_status_poll_seconds,
            order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
            stop_loss_pct=config.risk.stop_loss_pct,
            take_profit_pct=config.risk.take_profit_pct,
            enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
            serialize_value=serialize_value,
        ):
            return

    # Force-test mode bypasses normal strategy logic and submits the configured action.
    if config.execution.force_test_trade:
        print("\nFORCE_TEST_TRADE is ON")

        # Forced orders still respect market hours.
        if not current_market_open:
            print("Market is closed. Forced order not sent.")
            log_decision_for_cycle("HOLD", "market_closed_force_mode", market_open=False)
            return

        # In force BUY mode, enter a position if one is not already open.
        if config.execution.force_direction.upper() == "BUY":
            if position_side == "long":
                print("Already in a long position. Force BUY skipped.")
                log_decision_for_cycle("HOLD", "force_buy_skipped_long_position", market_open=True)
                return

            if position_side == "flat":
                if live_state["stop_orders"]:
                    print("Flat symbol still has stop orders at Alpaca. Forced BUY paused.")
                    log_decision_for_cycle("HOLD", "stale_protective_stop_exists", market_open=True)
                    return
                qty = calculate_order_qty(
                    latest_close,
                    buying_power,
                    config.risk.risk_fraction_of_buying_power,
                    config.risk.max_position_qty,
                )
                resulting_side = "long"
                entry_risk_plan = strategy.build_entry_risk_plan(latest, resulting_side, config.risk)
                if build_exit_levels_from_entry_plan(
                    entry_price=latest_close,
                    position_side=resulting_side,
                    stop_loss_pct=config.risk.stop_loss_pct,
                    take_profit_pct=config.risk.take_profit_pct,
                    entry_risk_plan=entry_risk_plan,
                ) is None:
                    print("Forced BUY skipped because the strategy could not build valid exit levels.")
                    log_decision_for_cycle(
                        "HOLD",
                        "invalid_entry_risk_plan"
                        if entry_risk_plan is None or not entry_risk_plan.get("rejection_reason")
                        else str(entry_risk_plan["rejection_reason"]),
                        market_open=True,
                    )
                    return
            else:
                qty = position_qty
                resulting_side = "flat"
                entry_risk_plan = None
                cancel_stop_orders_if_needed(trading_client, symbol)

            decision_id = log_decision_for_cycle("BUY", "forced_test_trade", qty=qty, market_open=True)
            submit_and_track_order(
                trading_client=trading_client,
                session_state=session_state,
                symbol=symbol,
                action="BUY",
                qty=qty,
                event_key=latest_event_key,
                reason="forced_test_trade",
                decision_price=latest_close,
                decision_id=decision_id,
                position_side_before=position_side,
                position_side_after=resulting_side,
                bot_version=config.bot.version,
                strategy_name=strategy.name,
                strategy_version=strategy.version,
                order_status_poll_seconds=config.execution.order_status_poll_seconds,
                order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
                stop_loss_pct=config.risk.stop_loss_pct,
                take_profit_pct=config.risk.take_profit_pct,
                enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
                serialize_value=serialize_value,
                entry_risk_plan=entry_risk_plan,
                signal_time=latest.name.isoformat() if hasattr(latest.name, "isoformat") else str(latest.name),
            )
            return

        # In force SELL mode, close the current position if one exists.
        if config.execution.force_direction.upper() == "SELL":
            if position_side == "short":
                print("Already in a short position. Force SELL skipped.")
                log_decision_for_cycle("HOLD", "force_sell_skipped_short_position", market_open=True)
                return

            if position_side == "flat":
                if live_state["stop_orders"]:
                    print("Flat symbol still has stop orders at Alpaca. Forced SELL paused.")
                    log_decision_for_cycle("HOLD", "stale_protective_stop_exists", market_open=True)
                    return
                qty = calculate_order_qty(
                    latest_close,
                    buying_power,
                    config.risk.risk_fraction_of_buying_power,
                    config.risk.max_position_qty,
                )
                resulting_side = "short"
                entry_risk_plan = strategy.build_entry_risk_plan(latest, resulting_side, config.risk)
                if build_exit_levels_from_entry_plan(
                    entry_price=latest_close,
                    position_side=resulting_side,
                    stop_loss_pct=config.risk.stop_loss_pct,
                    take_profit_pct=config.risk.take_profit_pct,
                    entry_risk_plan=entry_risk_plan,
                ) is None:
                    print("Forced SELL skipped because the strategy could not build valid exit levels.")
                    log_decision_for_cycle(
                        "HOLD",
                        "invalid_entry_risk_plan"
                        if entry_risk_plan is None or not entry_risk_plan.get("rejection_reason")
                        else str(entry_risk_plan["rejection_reason"]),
                        market_open=True,
                    )
                    return
            else:
                qty = position_qty
                resulting_side = "flat"
                entry_risk_plan = None
                cancel_stop_orders_if_needed(trading_client, symbol)

            decision_id = log_decision_for_cycle("SELL", "forced_test_trade", qty=qty, market_open=True)
            submit_and_track_order(
                trading_client=trading_client,
                session_state=session_state,
                symbol=symbol,
                action="SELL",
                qty=qty,
                event_key=latest_event_key,
                reason="forced_test_trade",
                decision_price=latest_close,
                decision_id=decision_id,
                position_side_before=position_side,
                position_side_after=resulting_side,
                bot_version=config.bot.version,
                strategy_name=strategy.name,
                strategy_version=strategy.version,
                order_status_poll_seconds=config.execution.order_status_poll_seconds,
                order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
                stop_loss_pct=config.risk.stop_loss_pct,
                take_profit_pct=config.risk.take_profit_pct,
                enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
                serialize_value=serialize_value,
                entry_risk_plan=entry_risk_plan,
                signal_time=latest.name.isoformat() if hasattr(latest.name, "isoformat") else str(latest.name),
            )
            return

        print("FORCE_DIRECTION must be BUY or SELL.")
        log_decision_for_cycle("HOLD", "invalid_force_direction", market_open=True)
        return

    # Skip live trading when the market is closed.
    if not current_market_open:
        print("Market is closed. No live order submitted.")
        log_decision_for_cycle("HOLD", "market_closed", market_open=False)
        return

    # Check risk-based exit conditions first so an existing position can be
    # closed before considering any new entry on this cycle.
    if handle_exit_triggers(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        latest_close=latest_close,
        event_key=latest_event_key,
        live_state=live_state,
        should_exit_position=should_exit_position,
        strategy=strategy,
        risk_settings=config.risk,
        log_decision_event=log_decision_for_cycle,
        bot_version=config.bot.version,
        strategy_exit_reason=strategy_exit_reason,
        position_context=position_context,
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        order_status_poll_seconds=config.execution.order_status_poll_seconds,
        order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
        enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    ):
        return

    # If no risk-based exit was needed, check whether the strategy signals
    # call for entering or exiting a position for this symbol.
    if handle_strategy_actions(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        latest_close=latest_close,
        event_key=latest_event_key,
        buying_power=buying_power,
        live_state=live_state,
        calculate_order_qty=calculate_order_qty,
        strategy=strategy,
        latest_row=latest,
        strategy_entry_action=strategy_entry_action,
        strategy_exit_action=strategy_exit_action,
        strategy_entry_reason=strategy.entry_reason(),
        strategy_exit_reason=strategy_exit_reason,
        risk_settings=config.risk,
        log_decision_event=log_decision_for_cycle,
        bot_version=config.bot.version,
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        order_status_poll_seconds=config.execution.order_status_poll_seconds,
        order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
        enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        serialize_value=serialize_value,
        signal_time=latest.name.isoformat() if hasattr(latest.name, "isoformat") else str(latest.name),
    ):
        return

    # If no action was needed, record a HOLD decision for this cycle.
    print("No trade this cycle.")
    log_decision_for_cycle("HOLD", "no_trade", market_open=True)


def run_cycle(session_state: dict, trading_client, config: BotConfig) -> None:
    """Run one full bot cycle for every configured symbol."""
    # Run one symbol cycle for each configured ticker.
    for symbol_config in config.symbols:
        run_symbol_cycle(session_state, trading_client, config, symbol_config)


def run() -> None:
    """Run the bot once or keep polling until the user stops it."""
    config = load_config()
    session_state = {}

    # Connect to Alpaca using the configured trading mode.
    try:
        trading_client = connect_alpaca(paper=config.bot.paper_trading)
    except ValueError as exc:
        print(exc)
        return

    # In single-run mode, execute one cycle and exit.
    if not config.bot.run_continuously:
        run_cycle(session_state, trading_client, config)
        return

    print(f"Starting continuous mode. Polling every {config.bot.poll_interval_seconds} seconds.")
    print(f"Bot version: {config.bot.version}")

    # In continuous mode, keep running the bot and wait between polling cycles.
    try:
        while True:
            run_cycle(session_state, trading_client, config)
            print(f"Sleeping for {config.bot.poll_interval_seconds} seconds...")
            time.sleep(config.bot.poll_interval_seconds)
    except KeyboardInterrupt:  # Stop the bot gracefully when the user presses Ctrl+C.
        print("\nBot stopped by user.")


if __name__ == "__main__":
    run()
