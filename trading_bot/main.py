"""Runs the trading bot workflow from data download through order decisions."""

import time

from alpaca.trading.enums import OrderSide

from trading_bot.broker import connect_alpaca, get_account, market_is_open
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
    cancel_protective_stop_if_needed,
    handle_entry_signals,
    handle_exit_triggers,
    handle_pending_order_state,
    submit_and_track_order,
)
from trading_bot.risk import calculate_order_qty, should_exit_position
from trading_bot.state import get_live_broker_state


def print_metrics(metrics: dict) -> None:
    """Print a small summary of backtest metrics for the latest run."""
    print("\nBacktest metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")


def run_symbol_cycle(session_state: dict, trading_client, config: BotConfig, symbol_config: SymbolAssignment) -> None:
    """Run one full bot cycle for a single configured symbol."""
    symbol = symbol_config.ticker
    strategy = symbol_config.strategy
    strategy_metadata = build_strategy_metadata(
        strategy.name,
        strategy.version,
        strategy.interval,
    )

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

    if raw_df.empty:
        print("Alpaca returned no bars for this request.")
        return

    first_bar_timestamp = raw_df.index.min()
    last_bar_timestamp = raw_df.index.max()
    print(
        "Bar time range: "
        f"{first_bar_timestamp.isoformat()} -> {last_bar_timestamp.isoformat()}"
    )

    df = strategy.prepare_dataframe(raw_df)
    print(
        "Prepared "
        f"{len(df)} strategy rows after warmup "
        f"(minimum raw bars needed: {min_required_bars})."
    )

    if len(df) < 2:
        print(
            "Not enough data to compute strategy. "
            f"Need at least {min_required_bars} raw bars for {strategy.name}, "
            f"but received {len(raw_df)}."
        )
        return

    print("\nRecent strategy data:")
    print(df[strategy.recent_display_columns()].tail(8))

    metrics = strategy.calculate_backtest_metrics(df)
    print_metrics(metrics)

    latest = df.iloc[-1]
    latest_close = float(latest["Close"])
    latest_event_key = strategy.event_key(latest)
    latest_signal_value = strategy.latest_signal_value(latest)

    print(f"\nLatest close: {latest_close:.2f}")
    print(f"{strategy.latest_signal_label()}: {latest_signal_value}")

    account = get_account(trading_client)
    if account is None:
        print("Could not retrieve account. Exiting.")
        return

    buying_power = float(account.buying_power)
    print(f"Account status: {account.status}")
    print(f"Buying power: {buying_power:.2f}")

    live_state = get_live_broker_state(trading_client, symbol)
    position = live_state["position"]
    in_position = live_state["in_position"]
    entry_price = live_state["entry_price"]
    print(f"Currently holding {symbol}: {in_position}")
    if live_state["protective_stop_order"] is not None:
        stop_price = serialize_value(getattr(live_state["protective_stop_order"], "stop_price", ""))
        print(f"Protective stop currently active at Alpaca: {stop_price}")
    if live_state["blocking_orders"]:
        print(f"Non-protective working orders at Alpaca: {len(live_state['blocking_orders'])}")

    current_market_open = market_is_open(trading_client)

    def log_decision_for_cycle(action: str, reason: str, qty: int = 0, market_open=None) -> str:
        decision_row = build_decision_payload(
            bot_version=config.bot.version,
            symbol=symbol,
            strategy_metadata=strategy_metadata,
            latest_close=latest_close,
            entry_price=entry_price,
            buying_power=buying_power,
            in_position=in_position,
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
            position=position,
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
            strategy_signals=strategy.build_strategy_signals(latest),
            stop_loss_pct=config.risk.stop_loss_pct,
            take_profit_pct=config.risk.take_profit_pct,
            risk_fraction_of_buying_power=config.risk.risk_fraction_of_buying_power,
            max_position_qty=config.risk.max_position_qty,
            live_state=live_state,
        )
        return log_decision_event(decision_row=decision_row, strategy_context_row=strategy_context_row)

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
        enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    ):
        return

    if config.execution.force_test_trade:
        print("\nFORCE_TEST_TRADE is ON")

        if not current_market_open:
            print("Market is closed. Forced order not sent.")
            log_decision_for_cycle("HOLD", "market_closed_force_mode", market_open=False)
            return

        if config.execution.force_direction.upper() == "BUY":
            if in_position:
                print("Already in a position. Force BUY skipped.")
                log_decision_for_cycle("HOLD", "force_buy_skipped_in_position", market_open=True)
                return

            qty = calculate_order_qty(
                latest_close,
                buying_power,
                config.risk.risk_fraction_of_buying_power,
                config.risk.max_position_qty,
            )
            decision_id = log_decision_for_cycle("BUY", "forced_test_trade", qty=qty, market_open=True)
            submit_and_track_order(
                trading_client=trading_client,
                session_state=session_state,
                symbol=symbol,
                action="BUY",
                side=OrderSide.BUY,
                qty=qty,
                event_key=latest_event_key,
                reason="forced_test_trade",
                decision_price=latest_close,
                decision_id=decision_id,
                bot_version=config.bot.version,
                strategy_name=strategy.name,
                strategy_version=strategy.version,
                order_status_poll_seconds=config.execution.order_status_poll_seconds,
                order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
                stop_loss_pct=config.risk.stop_loss_pct,
                enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
                serialize_value=serialize_value,
            )
            return

        if config.execution.force_direction.upper() == "SELL":
            if not in_position:
                print("No open position. Force SELL skipped.")
                log_decision_for_cycle("HOLD", "force_sell_skipped_no_position", market_open=True)
                return

            qty = int(float(position.qty))
            cancel_protective_stop_if_needed(trading_client, symbol)
            decision_id = log_decision_for_cycle("SELL", "forced_test_trade", qty=qty, market_open=True)
            submit_and_track_order(
                trading_client=trading_client,
                session_state=session_state,
                symbol=symbol,
                action="SELL",
                side=OrderSide.SELL,
                qty=qty,
                event_key=latest_event_key,
                reason="forced_test_trade",
                decision_price=latest_close,
                decision_id=decision_id,
                bot_version=config.bot.version,
                strategy_name=strategy.name,
                strategy_version=strategy.version,
                order_status_poll_seconds=config.execution.order_status_poll_seconds,
                order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
                stop_loss_pct=config.risk.stop_loss_pct,
                enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
                serialize_value=serialize_value,
            )
            return

        print("FORCE_DIRECTION must be BUY or SELL.")
        log_decision_for_cycle("HOLD", "invalid_force_direction", market_open=True)
        return

    if not current_market_open:
        print("Market is closed. No live order submitted.")
        log_decision_for_cycle("HOLD", "market_closed", market_open=False)
        return

    if handle_exit_triggers(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        latest_close=latest_close,
        event_key=latest_event_key,
        live_state=live_state,
        should_exit_position=should_exit_position,
        stop_loss_pct=config.risk.stop_loss_pct,
        take_profit_pct=config.risk.take_profit_pct,
        log_decision_event=log_decision_for_cycle,
        bot_version=config.bot.version,
        strategy_exit_reason=strategy.exit_reason(),
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        order_status_poll_seconds=config.execution.order_status_poll_seconds,
        order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
        enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    ):
        return

    if handle_entry_signals(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        latest_close=latest_close,
        event_key=latest_event_key,
        buying_power=buying_power,
        live_state=live_state,
        calculate_order_qty=calculate_order_qty,
        strategy_entry_signal=strategy.should_enter_long(latest),
        strategy_exit_signal=strategy.should_exit_long(latest),
        strategy_entry_reason=strategy.entry_reason(),
        strategy_exit_reason=strategy.exit_reason(),
        risk_fraction_of_buying_power=config.risk.risk_fraction_of_buying_power,
        max_position_qty=config.risk.max_position_qty,
        log_decision_event=log_decision_for_cycle,
        bot_version=config.bot.version,
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        order_status_poll_seconds=config.execution.order_status_poll_seconds,
        order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
        stop_loss_pct=config.risk.stop_loss_pct,
        enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    ):
        return

    print("No trade this cycle.")
    log_decision_for_cycle("HOLD", "no_trade", market_open=True)


def run_cycle(session_state: dict, trading_client, config: BotConfig) -> None:
    """Run one full bot cycle for every configured symbol."""
    for symbol_config in config.symbols:
        run_symbol_cycle(session_state, trading_client, config, symbol_config)


def run() -> None:
    """Run the bot once or keep polling until the user stops it."""
    config = load_config()
    session_state = {}

    try:
        trading_client = connect_alpaca(paper=config.bot.paper_trading)
    except ValueError as exc:
        print(exc)
        return

    if not config.bot.run_continuously:
        run_cycle(session_state, trading_client, config)
        return

    print(f"Starting continuous mode. Polling every {config.bot.poll_interval_seconds} seconds.")
    print(f"Bot version: {config.bot.version}")

    try:
        while True:
            run_cycle(session_state, trading_client, config)
            print(f"Sleeping for {config.bot.poll_interval_seconds} seconds...")
            time.sleep(config.bot.poll_interval_seconds)
    except KeyboardInterrupt:
        print("\nBot stopped by user.")


if __name__ == "__main__":
    run()
