"""Runs the trading bot workflow from data download through order decisions."""

import pandas as pd
import time

from trading_bot.broker import (
    connect_alpaca,
    get_account,
    get_all_positions,
    get_asset,
    get_asset_flags,
    get_market_clock,
    get_position_market_value,
    get_working_orders,
    market_is_open,
)
from trading_bot.config import AccountAssignment, BotConfig, SymbolAssignment, load_config
from trading_bot.data import download_data
from trading_bot.decision_logging import (
    build_decision_payload,
    build_strategy_context_payload,
    build_strategy_metadata,
    log_decision_event,
    serialize_value,
)
from trading_bot.incident_logging import sync_incident_state
from trading_bot.logging_utils import configure_logging
from trading_bot.execution import (
    handle_flatten_before_close,
    handle_exit_triggers,
    handle_pending_order_state,
    submit_and_track_order,
    cancel_stop_orders_if_needed,
    handle_strategy_actions,
    reconcile_missing_protective_stop,
)
from trading_bot.risk import (
    build_exit_levels_for_live_position,
    build_exit_levels_from_entry_plan,
    calculate_order_qty,
    is_within_flatten_before_close_window,
    should_exit_position,
)
from trading_bot.strategies.base import PositionContext
from trading_bot.state import (
    clear_active_exit_levels,
    get_active_exit_levels,
    get_live_broker_state,
    load_session_state,
    set_active_exit_levels,
)


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


def summarize_stop_state(
    live_state: dict,
    active_exit_levels: dict | None,
    reconciliation_status: str | None = None,
) -> str:
    """Return a short stop summary for console output."""
    protective_stop_order = live_state["protective_stop_order"]
    if protective_stop_order is not None:
        stop_side = serialize_value(getattr(protective_stop_order, "side", ""))
        stop_price = serialize_value(getattr(protective_stop_order, "stop_price", ""))
        return f"broker:{stop_side}@{stop_price}"

    if live_state["stop_orders"]:
        return f"unmatched_open_stops={len(live_state['stop_orders'])}"

    if active_exit_levels is not None and active_exit_levels.get("stop_price") is not None:
        label = "reconciled" if reconciliation_status == "protective_stop_reconciled" else "derived"
        return f"{label}@{float(active_exit_levels['stop_price']):.2f}"

    return "none"


def format_symbol_label(symbol: str, width: int) -> str:
    """Pad symbols to a fixed width so console summaries line up cleanly."""
    return f"{symbol:<{width}}"


def build_state_symbol(state_namespace: str, symbol: str) -> str:
    """Namespace local session state so accounts can trade the same ticker independently."""
    return symbol if not state_namespace else f"{state_namespace}:{symbol}"


def print_cycle_summary(results: list[dict]) -> None:
    """Print a compact multi-symbol summary after each full cycle."""
    if not results:
        return

    account_width = max(len(str(result["account"])) for result in results)
    symbol_width = max(len(str(result["symbol"])) for result in results)
    print("\nCycle summary:")
    for result in results:
        print(
            f"{format_symbol_label(str(result['account']), account_width)} | "
            f"{format_symbol_label(str(result['symbol']), symbol_width)} | {result['strategy']} | close={result['latest_close']} | "
            f"signal={result['signal']} | position={result['position']} | market={result['market']} | "
            f"decision={result['decision']}:{result['reason']} | stop={result['stop']}"
        )


def current_portfolio_exposure_value(trading_client) -> float:
    """Return the absolute market value currently deployed across open positions."""
    return sum(get_position_market_value(position) for position in get_all_positions(trading_client))


def projected_portfolio_exposure_fraction(
    *,
    trading_client,
    buying_power: float,
    latest_close: float,
    entry_qty: int,
) -> tuple[float, float, float]:
    """Estimate current and projected exposure as fractions of buying power."""
    current_exposure_value = current_portfolio_exposure_value(trading_client)
    projected_exposure_value = current_exposure_value + (latest_close * entry_qty)
    denominator = buying_power if buying_power > 0 else 1.0
    return (
        current_exposure_value,
        current_exposure_value / denominator,
        projected_exposure_value / denominator,
    )


def evaluate_entry_safeguards(
    *,
    trading_client,
    symbol: str,
    action: str,
    latest_close: float,
    entry_qty: int,
    buying_power: float,
    execution_settings,
    risk_settings,
) -> tuple[str | None, dict]:
    """Return an optional block reason plus asset/exposure metadata for a candidate entry."""
    asset = get_asset(trading_client, symbol)
    asset_flags = get_asset_flags(asset)
    current_exposure_value, current_exposure_fraction, projected_exposure_fraction = projected_portfolio_exposure_fraction(
        trading_client=trading_client,
        buying_power=buying_power,
        latest_close=latest_close,
        entry_qty=entry_qty,
    )
    metadata = {
        "asset_flags": asset_flags,
        "current_exposure_value": current_exposure_value,
        "current_exposure_fraction": current_exposure_fraction,
        "projected_exposure_fraction": projected_exposure_fraction,
    }

    if not asset_flags["tradable"]:
        return "asset_not_tradable", metadata

    if action == "SELL":
        if not asset_flags["shortable"]:
            return "asset_not_shortable", metadata
        if execution_settings.require_easy_to_borrow_for_short_entries and not asset_flags["easy_to_borrow"]:
            return "asset_not_easy_to_borrow", metadata

    exposure_cap = risk_settings.max_total_position_fraction_of_buying_power
    if exposure_cap > 0 and projected_exposure_fraction > exposure_cap:
        return "portfolio_exposure_cap_reached", metadata

    return None, metadata


def print_preflight_report(
    trading_client,
    account_config: AccountAssignment,
    session_state: dict,
) -> None:
    """Print a startup preflight summary before the live trading loop begins."""
    print(f"\nPreflight: {account_config.name}")
    symbol_width = max(len(symbol_config.ticker) for symbol_config in account_config.symbols) if account_config.symbols else 1
    all_working_orders = get_working_orders(trading_client)

    account = get_account(trading_client)
    if account is None:
        print("Account: unavailable")
    else:
        buying_power = float(account.buying_power)
        print(
            "Account: "
            f"status={serialize_value(account.status)} | buying_power={buying_power:.2f} | "
            f"trading_blocked={bool(getattr(account, 'trading_blocked', False))}"
        )

    market_clock = get_market_clock(trading_client)
    if market_clock is None:
        print("Market: unavailable")
    else:
        print(
            "Market: "
            f"is_open={bool(getattr(market_clock, 'is_open', False))} | "
            f"timestamp={serialize_value(getattr(market_clock, 'timestamp', ''))}"
        )

    for symbol_config in account_config.symbols:
        symbol = symbol_config.ticker
        state_symbol = build_state_symbol(account_config.state_namespace, symbol)
        strategy = symbol_config.strategy
        live_state = get_live_broker_state(trading_client, symbol, all_working_orders=all_working_orders)
        if live_state["is_flat"]:
            clear_active_exit_levels(session_state, state_symbol)
        asset_flags = get_asset_flags(get_asset(trading_client, symbol))
        stop_summary = summarize_stop_state(
            live_state,
            get_active_exit_levels(session_state, state_symbol),
        )
        position_label = "flat"
        if not live_state["is_flat"]:
            position_label = f"{live_state['position_side']} x{live_state['position_qty']}"
        print(
            f"{format_symbol_label(symbol, symbol_width)} | {strategy.name}@{strategy.version} | "
            f"position={position_label} | "
            f"tradable={asset_flags['tradable']} | shortable={asset_flags['shortable']} | "
            f"etb={asset_flags['easy_to_borrow']} | stop={stop_summary} | "
            f"working_orders={len(live_state['working_orders'])}"
        )


def run_symbol_cycle(
    session_state: dict,
    trading_client,
    config: BotConfig,
    account_config: AccountAssignment,
    symbol_config: SymbolAssignment,
) -> dict:
    """Run one full bot cycle for a single configured symbol."""
    # Get the current symbol and its assigned strategy from the config.
    symbol = symbol_config.ticker
    state_symbol = build_state_symbol(account_config.state_namespace, symbol)
    strategy = symbol_config.strategy
    strategy_metadata = build_strategy_metadata(
        strategy.name,
        strategy.version,
        strategy.interval,
    )
    summary = {
        "account": account_config.name,
        "symbol": symbol,
        "strategy": f"{strategy.name}@{strategy.version}",
        "latest_close": "?",
        "signal": "?",
        "position": "unknown",
        "market": "unknown",
        "decision": "",
        "reason": "",
        "stop": "unknown",
    }
    incident_context = {
        "account_name": account_config.name,
        "symbol": symbol,
        "latest_close": None,
        "signal": None,
        "market_is_open": None,
        "buying_power": None,
        "position_side": "unknown",
        "position_qty": 0,
        "entry_price": None,
        "stop_summary": "unknown",
        "asset_flags": None,
        "current_exposure_value": None,
        "current_exposure_fraction": None,
        "projected_exposure_fraction": None,
    }

    def finish(action: str, reason: str, **updates) -> dict:
        summary.update(updates)
        summary["decision"] = action
        summary["reason"] = reason
        sync_incident_state(
            account_name=account_config.name,
            symbol=symbol,
            action=action,
            reason=reason,
            bot_version=config.bot.version,
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            interval=strategy.interval,
            context=incident_context,
        )
        print(f"Decision: {action} | reason={reason}")
        return summary

    # Download recent market data needed to evaluate the strategy.
    print(f"\n=== {account_config.name} | {symbol} | {strategy.name}@{strategy.version} ===")
    print("Downloading market data from Alpaca...")
    try:
        raw_df = download_data(
            symbol,
            strategy.interval,
            strategy.lookback_bars,
            api_key_env=account_config.api_key_env,
            secret_key_env=account_config.secret_key_env,
            fallback_api_key_env=account_config.fallback_api_key_env,
            fallback_secret_key_env=account_config.fallback_secret_key_env,
        )
    except ValueError as exc:
        print(exc)
        return finish("ERROR", "data_download_value_error")
    except Exception as exc:
        print(f"Error downloading data: {exc}")
        return finish("ERROR", "data_download_failed")

    min_required_bars = strategy.min_required_bars()

    # Stop this cycle if Alpaca returned no market data.
    if raw_df.empty:
        print("Alpaca returned no bars for this request.")
        return finish("ERROR", "no_market_data")

    # Record the time range covered by the downloaded bars.
    first_bar_timestamp = raw_df.index.min()
    last_bar_timestamp = raw_df.index.max()

    # Prepare the raw price data so the strategy can evaluate its signals.
    df = strategy.prepare_dataframe(raw_df)
    print(
        "Data ready: "
        f"raw={len(raw_df)} | prepared={len(df)} | interval={strategy.interval} | "
        f"bars={first_bar_timestamp.isoformat()} -> {last_bar_timestamp.isoformat()}"
    )

    # Stop this cycle if there are not enough prepared rows to evaluate the strategy.
    if len(df) < 2:
        print(
            "Not enough data to compute strategy. "
            f"Need at least {min_required_bars} raw bars for {strategy.name}, "
            f"but received {len(raw_df)}."
        )
        return finish("ERROR", "not_enough_data")

    # Pull the latest row so the bot can inspect the current signal state.
    latest = df.iloc[-1]
    latest_close = float(latest["Close"])
    latest_event_key = strategy.event_key(latest)
    latest_signal_value = strategy.latest_signal_value(latest)
    summary["latest_close"] = f"{latest_close:.2f}"
    summary["signal"] = str(latest_signal_value)
    incident_context["latest_close"] = latest_close
    incident_context["signal"] = serialize_value(latest_signal_value)

    # Get the account so position sizing can use live buying power.
    account = get_account(trading_client)
    if account is None:
        print("Could not retrieve account. Exiting.")
        return finish("ERROR", "account_lookup_failed")

    buying_power = float(account.buying_power)
    incident_context["buying_power"] = buying_power
    exposure_metadata = {
        "asset_flags": {
            "tradable": False,
            "shortable": False,
            "easy_to_borrow": False,
            "marginable": False,
            "fractionable": False,
        },
        "current_exposure_value": current_portfolio_exposure_value(trading_client),
        "current_exposure_fraction": 0.0,
        "projected_exposure_fraction": 0.0,
    }

    # Check the symbol's current live broker state before making a new decision.
    live_state = get_live_broker_state(trading_client, symbol)
    in_position = live_state["in_position"]
    position_side = live_state["position_side"]
    position_qty = live_state["position_qty"]
    entry_price = live_state["entry_price"]
    incident_context["position_side"] = position_side
    incident_context["position_qty"] = position_qty
    incident_context["entry_price"] = entry_price
    if live_state["is_flat"]:
        clear_active_exit_levels(session_state, state_symbol)
    active_exit_levels = None
    stop_reconciliation_status = None
    if position_side in {"long", "short"} and entry_price is not None:
        active_exit_levels = build_exit_levels_for_live_position(
            entry_price=entry_price,
            position_side=position_side,
            stop_loss_pct=config.risk.stop_loss_pct,
            take_profit_pct=config.risk.take_profit_pct,
            protective_stop_order=live_state["protective_stop_order"],
            active_exit_levels=get_active_exit_levels(session_state, state_symbol),
            risk_reward_multiple=strategy.risk_reward_multiple(),
        )
        if active_exit_levels is not None:
            set_active_exit_levels(session_state, state_symbol, active_exit_levels)
        stop_reconciliation_status = reconcile_missing_protective_stop(
            trading_client=trading_client,
            symbol=symbol,
            live_state=live_state,
            active_exit_levels=active_exit_levels,
            enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        )
        if stop_reconciliation_status == "protective_stop_reconciled":
            live_state = get_live_broker_state(trading_client, symbol)
            in_position = live_state["in_position"]
            position_side = live_state["position_side"]
            position_qty = live_state["position_qty"]
            entry_price = live_state["entry_price"]
            active_exit_levels = build_exit_levels_for_live_position(
                entry_price=entry_price,
                position_side=position_side,
                stop_loss_pct=config.risk.stop_loss_pct,
                take_profit_pct=config.risk.take_profit_pct,
                protective_stop_order=live_state["protective_stop_order"],
                active_exit_levels=get_active_exit_levels(session_state, state_symbol),
                risk_reward_multiple=strategy.risk_reward_multiple(),
            )
            if active_exit_levels is not None:
                set_active_exit_levels(session_state, state_symbol, active_exit_levels)

    position_context = build_live_position_context(df, live_state, active_exit_levels)
    strategy_entry_action = strategy.entry_action(latest)
    strategy_exit_action = strategy.exit_action(latest, position_context)
    strategy_exit_reason = strategy.exit_reason(latest, position_context)

    # Check whether the market is currently open before allowing live orders.
    market_clock = get_market_clock(trading_client)
    current_market_open = bool(market_clock.is_open) if market_clock is not None else market_is_open(trading_client)
    flatten_window_active = is_within_flatten_before_close_window(
        None if market_clock is None else getattr(market_clock, "timestamp", None),
        flatten_before_close=config.risk.flatten_before_close,
        flatten_minutes_before_close=config.risk.flatten_minutes_before_close,
    )
    summary["position"] = "flat" if not in_position else f"{position_side} x{position_qty}"
    summary["market"] = "open" if current_market_open else "closed"
    summary["stop"] = summarize_stop_state(live_state, active_exit_levels, stop_reconciliation_status)
    incident_context["market_is_open"] = current_market_open
    incident_context["stop_summary"] = summary["stop"]
    print(
        "Snapshot: "
        f"close={latest_close:.2f} | signal={latest_signal_value} | position={summary['position']} | "
        f"market={summary['market']} | stop={summary['stop']} | "
        f"buying_power={buying_power:.2f}"
    )
    if live_state["blocking_orders"]:
        print(f"Non-protective working orders at Alpaca: {len(live_state['blocking_orders'])}")

    # Build the decision and strategy-context log rows for this cycle.
    def log_decision_for_cycle(action: str, reason: str, qty: int = 0, market_open=None) -> str:
        decision_row = build_decision_payload(
            bot_version=config.bot.version,
            account_name=account_config.name,
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
            account_name=account_config.name,
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

    def guard_new_entry(action: str, qty: int, market_open: bool) -> dict | None:
        nonlocal exposure_metadata
        block_reason, exposure_metadata = evaluate_entry_safeguards(
            trading_client=trading_client,
            symbol=symbol,
            action=action,
            latest_close=latest_close,
            entry_qty=qty,
            buying_power=buying_power,
            execution_settings=config.execution,
            risk_settings=config.risk,
        )
        if block_reason is None:
            return None

        incident_context["asset_flags"] = exposure_metadata["asset_flags"]
        incident_context["current_exposure_value"] = exposure_metadata["current_exposure_value"]
        incident_context["current_exposure_fraction"] = exposure_metadata["current_exposure_fraction"]
        incident_context["projected_exposure_fraction"] = exposure_metadata["projected_exposure_fraction"]
        asset_flags = exposure_metadata["asset_flags"]
        print(
            "Entry blocked: "
            f"reason={block_reason} | tradable={asset_flags['tradable']} | "
            f"shortable={asset_flags['shortable']} | etb={asset_flags['easy_to_borrow']} | "
            f"projected_exposure={exposure_metadata['projected_exposure_fraction']:.4f}"
        )
        log_decision_for_cycle("HOLD", block_reason, market_open=market_open)
        return finish("HOLD", block_reason)

    # Resolve any previously submitted order before evaluating fresh trade signals.
    pending_state_result = handle_pending_order_state(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        state_symbol=state_symbol,
        current_market_open=current_market_open,
        live_state=live_state,
        log_decision_event=log_decision_for_cycle,
        bot_version=config.bot.version,
        account_name=account_config.name,
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        stop_loss_pct=config.risk.stop_loss_pct,
        take_profit_pct=config.risk.take_profit_pct,
        enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    )
    if pending_state_result is not None:
        return finish(*pending_state_result)

    if current_market_open and flatten_window_active:
        print("Inside the configured pre-close flatten window.")
        flatten_result = handle_flatten_before_close(
            trading_client=trading_client,
            session_state=session_state,
            symbol=symbol,
            state_symbol=state_symbol,
            latest_close=latest_close,
            event_key=latest_event_key,
            live_state=live_state,
            log_decision_event=log_decision_for_cycle,
            bot_version=config.bot.version,
            account_name=account_config.name,
            strategy_name=strategy.name,
            strategy_version=strategy.version,
            order_status_poll_seconds=config.execution.order_status_poll_seconds,
            order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
            stop_loss_pct=config.risk.stop_loss_pct,
            take_profit_pct=config.risk.take_profit_pct,
            enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
            serialize_value=serialize_value,
        )
        if flatten_result is not None:
            return finish(*flatten_result)

    # Force-test mode bypasses normal strategy logic and submits the configured action.
    if config.execution.force_test_trade:
        print("\nFORCE_TEST_TRADE is ON")

        # Forced orders still respect market hours.
        if not current_market_open:
            print("Market is closed. Forced order not sent.")
            log_decision_for_cycle("HOLD", "market_closed_force_mode", market_open=False)
            return finish("HOLD", "market_closed_force_mode")

        # In force BUY mode, enter a position if one is not already open.
        if config.execution.force_direction.upper() == "BUY":
            if position_side == "long":
                print("Already in a long position. Force BUY skipped.")
                log_decision_for_cycle("HOLD", "force_buy_skipped_long_position", market_open=True)
                return finish("HOLD", "force_buy_skipped_long_position")

            if position_side == "flat":
                if live_state["stop_orders"]:
                    print("Flat symbol still has stop orders at Alpaca. Forced BUY paused.")
                    log_decision_for_cycle("HOLD", "stale_protective_stop_exists", market_open=True)
                    return finish("HOLD", "stale_protective_stop_exists")
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
                    return finish(
                        "HOLD",
                        "invalid_entry_risk_plan"
                        if entry_risk_plan is None or not entry_risk_plan.get("rejection_reason")
                        else str(entry_risk_plan["rejection_reason"]),
                    )
                blocked_entry = guard_new_entry("BUY", qty, True)
                if blocked_entry is not None:
                    return blocked_entry
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
                state_symbol=state_symbol,
                action="BUY",
                qty=qty,
                event_key=latest_event_key,
                reason="forced_test_trade",
                decision_price=latest_close,
                decision_id=decision_id,
                position_side_before=position_side,
                position_side_after=resulting_side,
                bot_version=config.bot.version,
                account_name=account_config.name,
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
            return finish("BUY", "forced_test_trade")

        # In force SELL mode, close the current position if one exists.
        if config.execution.force_direction.upper() == "SELL":
            if position_side == "short":
                print("Already in a short position. Force SELL skipped.")
                log_decision_for_cycle("HOLD", "force_sell_skipped_short_position", market_open=True)
                return finish("HOLD", "force_sell_skipped_short_position")

            if position_side == "flat":
                if live_state["stop_orders"]:
                    print("Flat symbol still has stop orders at Alpaca. Forced SELL paused.")
                    log_decision_for_cycle("HOLD", "stale_protective_stop_exists", market_open=True)
                    return finish("HOLD", "stale_protective_stop_exists")
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
                    return finish(
                        "HOLD",
                        "invalid_entry_risk_plan"
                        if entry_risk_plan is None or not entry_risk_plan.get("rejection_reason")
                        else str(entry_risk_plan["rejection_reason"]),
                    )
                blocked_entry = guard_new_entry("SELL", qty, True)
                if blocked_entry is not None:
                    return blocked_entry
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
                state_symbol=state_symbol,
                action="SELL",
                qty=qty,
                event_key=latest_event_key,
                reason="forced_test_trade",
                decision_price=latest_close,
                decision_id=decision_id,
                position_side_before=position_side,
                position_side_after=resulting_side,
                bot_version=config.bot.version,
                account_name=account_config.name,
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
            return finish("SELL", "forced_test_trade")

        print("FORCE_DIRECTION must be BUY or SELL.")
        log_decision_for_cycle("HOLD", "invalid_force_direction", market_open=True)
        return finish("HOLD", "invalid_force_direction")

    # Skip live trading when the market is closed.
    if not current_market_open:
        print("Market is closed. No live order submitted.")
        log_decision_for_cycle("HOLD", "market_closed", market_open=False)
        return finish("HOLD", "market_closed")

    # Check risk-based exit conditions first so an existing position can be
    # closed before considering any new entry on this cycle.
    exit_trigger_result = handle_exit_triggers(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        state_symbol=state_symbol,
        latest_close=latest_close,
        event_key=latest_event_key,
        live_state=live_state,
        should_exit_position=should_exit_position,
        strategy=strategy,
        risk_settings=config.risk,
        log_decision_event=log_decision_for_cycle,
        bot_version=config.bot.version,
        account_name=account_config.name,
        strategy_exit_reason=strategy_exit_reason,
        position_context=position_context,
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        order_status_poll_seconds=config.execution.order_status_poll_seconds,
        order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
        enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        serialize_value=serialize_value,
    )
    if exit_trigger_result is not None:
        return finish(*exit_trigger_result)

    if position_side == "flat" and strategy_entry_action in {"BUY", "SELL"}:
        entry_qty = calculate_order_qty(
            latest_close,
            buying_power,
            config.risk.risk_fraction_of_buying_power,
            config.risk.max_position_qty,
        )
        blocked_entry = guard_new_entry(strategy_entry_action, entry_qty, True)
        if blocked_entry is not None:
            return blocked_entry

    # If no risk-based exit was needed, check whether the strategy signals
    # call for entering or exiting a position for this symbol.
    strategy_action_result = handle_strategy_actions(
        trading_client=trading_client,
        session_state=session_state,
        symbol=symbol,
        state_symbol=state_symbol,
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
        account_name=account_config.name,
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        order_status_poll_seconds=config.execution.order_status_poll_seconds,
        order_status_timeout_seconds=config.execution.order_status_timeout_seconds,
        enable_broker_side_stop_loss=config.execution.enable_broker_side_stop_loss,
        serialize_value=serialize_value,
        signal_time=latest.name.isoformat() if hasattr(latest.name, "isoformat") else str(latest.name),
    )
    if strategy_action_result is not None:
        return finish(*strategy_action_result)

    # If no action was needed, record a HOLD decision for this cycle.
    print("No trade this cycle.")
    log_decision_for_cycle("HOLD", "no_trade", market_open=True)
    return finish("HOLD", "no_trade")


def run_cycle(session_state: dict, account_clients: list[tuple[AccountAssignment, object]], config: BotConfig) -> list[dict]:
    """Run one full bot cycle for every configured account and symbol."""
    results = []
    for account_config, trading_client in account_clients:
        for symbol_config in account_config.symbols:
            try:
                results.append(run_symbol_cycle(session_state, trading_client, config, account_config, symbol_config))
            except Exception as exc:
                print(
                    f"\n=== {account_config.name} | {symbol_config.ticker} | "
                    f"{symbol_config.strategy.name}@{symbol_config.strategy.version} ==="
                )
                print(f"Unhandled symbol-cycle error: {exc}")
                results.append(
                    {
                        "account": account_config.name,
                        "symbol": symbol_config.ticker,
                        "strategy": f"{symbol_config.strategy.name}@{symbol_config.strategy.version}",
                        "latest_close": "?",
                        "signal": "?",
                        "position": "unknown",
                        "market": "unknown",
                        "decision": "ERROR",
                        "reason": "symbol_cycle_exception",
                        "stop": "unknown",
                    }
                )
                sync_incident_state(
                    account_name=account_config.name,
                    symbol=symbol_config.ticker,
                    action="ERROR",
                    reason="symbol_cycle_exception",
                    bot_version=config.bot.version,
                    strategy_name=symbol_config.strategy.name,
                    strategy_version=symbol_config.strategy.version,
                    interval=symbol_config.strategy.interval,
                    context={"exception_message": str(exc)},
                )
    print_cycle_summary(results)
    return results


def run() -> None:
    """Run the bot once or keep polling until the user stops it."""
    config = load_config()
    configure_logging(
        enable_full_decision_log=config.logging.enable_full_decision_log,
        enable_full_strategy_context_log=config.logging.enable_full_strategy_context_log,
        enable_notable_decision_log=config.logging.enable_notable_decision_log,
        enable_notable_strategy_context_log=config.logging.enable_notable_strategy_context_log,
    )
    session_state = load_session_state()

    account_clients = []
    for account_config in config.accounts:
        try:
            trading_client = connect_alpaca(
                paper=account_config.paper_trading,
                api_key_env=account_config.api_key_env,
                secret_key_env=account_config.secret_key_env,
                fallback_api_key_env=account_config.fallback_api_key_env,
                fallback_secret_key_env=account_config.fallback_secret_key_env,
            )
        except ValueError as exc:
            print(f"Skipping account {account_config.name}: {exc}")
            continue
        account_clients.append((account_config, trading_client))
        print_preflight_report(trading_client, account_config, session_state)

    if not account_clients:
        print("No configured accounts could connect. Exiting.")
        return

    if config.bot.preflight_only:
        print("Preflight-only mode is enabled. Exiting without running a trading cycle.")
        return

    # In single-run mode, execute one cycle and exit.
    if not config.bot.run_continuously:
        run_cycle(session_state, account_clients, config)
        return

    print(f"Starting continuous mode. Polling every {config.bot.poll_interval_seconds} seconds.")
    print(f"Bot version: {config.bot.version}")

    # In continuous mode, keep running the bot and wait between polling cycles.
    try:
        while True:
            run_cycle(session_state, account_clients, config)
            print(f"Sleeping for {config.bot.poll_interval_seconds} seconds...")
            time.sleep(config.bot.poll_interval_seconds)
    except KeyboardInterrupt:  # Stop the bot gracefully when the user presses Ctrl+C.
        print("\nBot stopped by user.")


if __name__ == "__main__":
    run()
