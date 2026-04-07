"""Offline historical backtesting entry point for configured strategies."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

from trading_bot.config import RiskSettings, load_config, load_raw_config
from trading_bot.risk import (
    assess_stop_distance,
    build_exit_levels_from_entry_plan,
    calculate_order_qty,
    exit_action_for_position_side,
)
from trading_bot.strategies.base import PositionContext
from trading_bot.strategies import create_strategy, default_strategy_config, list_supported_strategies


NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
LOCAL_BACKTEST_OUTPUT_DIR = "backtests"
SHARED_BACKTEST_OUTPUT_DIR = "shared_backtests"


@dataclass
class PendingOrder:
    action: str
    reason: str
    signal_time: datetime
    entry_risk_plan: dict | None = None


@dataclass
class SimulatedPosition:
    side: str
    qty: int
    entry_time: datetime
    entry_signal_time: datetime
    entry_price: float
    entry_reason: str
    entry_bar_number: int
    stop_price: float
    take_profit_price: float
    stop_source: str
    risk_reward_multiple: float | None


@dataclass
class TradeRecord:
    trade_id: str
    symbol: str
    strategy: str
    interval: str
    side: str
    qty: int
    entry_signal_time: str
    entry_time: str
    entry_price: float
    exit_time: str
    exit_price: float
    exit_reason: str
    bars_held: int
    pnl: float
    return_pct: float
    stop_price: float
    take_profit_price: float
    stop_distance: float
    stop_distance_frac_of_price: float
    stop_source: str
    risk_reward_multiple: float | None
    time_stop_triggered: bool
    macd_failure_triggered: bool


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for offline backtests."""
    parser = argparse.ArgumentParser(description="Run an offline historical backtest.")
    parser.add_argument("--symbol", help="Ticker symbol to backtest, for example MSFT.")
    parser.add_argument(
        "--strategy",
        help="Strategy name to use. If omitted, reuse the symbol's configured strategy from bot_config.toml.",
    )
    parser.add_argument("--start", help="Inclusive analysis start date or datetime.")
    parser.add_argument("--end", help="Inclusive end date or exclusive end datetime.")
    parser.add_argument("--interval", help="Optional bar interval override, for example 5m.")
    parser.add_argument("--config", default="bot_config.toml", help="Path to the bot TOML config.")
    parser.add_argument(
        "--output-dir",
        help="Optional output directory override for saved backtest artifacts.",
    )
    parser.add_argument(
        "--shared",
        action="store_true",
        help="Save this run under the team-shared backtest folder instead of the local ignored folder.",
    )
    parser.add_argument(
        "--initial-equity",
        type=float,
        default=100_000.0,
        help="Starting equity used for position sizing during the backtest.",
    )
    parser.add_argument(
        "--no-chart",
        action="store_true",
        help="Skip chart generation even if matplotlib is available.",
    )
    parser.add_argument(
        "--list-strategies",
        action="store_true",
        help="Print supported strategies and exit.",
    )
    args = parser.parse_args()
    if not args.list_strategies:
        missing = [
            name
            for name in ("symbol", "start", "end")
            if getattr(args, name) in {None, ""}
        ]
        if missing:
            parser.error("the following arguments are required: " + ", ".join(f"--{name}" for name in missing))
    return args


def resolve_output_dir(args: argparse.Namespace) -> str:
    """Resolve the final output directory for one backtest run."""
    if args.output_dir:
        return args.output_dir
    if args.shared:
        return SHARED_BACKTEST_OUTPUT_DIR
    return LOCAL_BACKTEST_OUTPUT_DIR


def parse_window_timestamp(value: str, *, is_end: bool) -> datetime:
    """Parse a date or datetime string in New York time for analysis windows."""
    text = value.strip()

    if "T" not in text and " " not in text:
        parsed_date = date.fromisoformat(text)
        if is_end:
            return datetime.combine(parsed_date + timedelta(days=1), time.min, tzinfo=NEW_YORK_TIMEZONE)
        return datetime.combine(parsed_date, time.min, tzinfo=NEW_YORK_TIMEZONE)

    parsed_datetime = datetime.fromisoformat(text)
    if parsed_datetime.tzinfo is None:
        parsed_datetime = parsed_datetime.replace(tzinfo=NEW_YORK_TIMEZONE)
    else:
        parsed_datetime = parsed_datetime.astimezone(NEW_YORK_TIMEZONE)
    return parsed_datetime


def to_utc(timestamp: datetime) -> datetime:
    """Convert a timezone-aware timestamp into UTC."""
    return timestamp.astimezone(timezone.utc)


def resolve_strategy_config(raw_config: dict, symbol: str, strategy_name: str | None, interval_override: str | None) -> dict:
    """Resolve one strategy config using symbol-specific config first, then strategy defaults."""
    symbols = raw_config.get("symbols", [])
    symbol_entry = next((entry for entry in symbols if str(entry.get("ticker", "")).upper() == symbol.upper()), None)

    if strategy_name is None:
        if symbol_entry is None:
            raise ValueError(
                f"No configured symbol entry found for {symbol}. Pass --strategy explicitly or add the symbol to bot_config.toml."
            )
        strategy_config = dict(symbol_entry["strategy"])
        strategy_name = str(strategy_config["name"])
    else:
        strategy_config = default_strategy_config(strategy_name)

        symbol_strategy = None
        if symbol_entry is not None:
            candidate = symbol_entry.get("strategy", {})
            if str(candidate.get("name", "")) == strategy_name:
                symbol_strategy = candidate

        matching_strategy = next(
            (
                entry["strategy"]
                for entry in symbols
                if str(entry.get("strategy", {}).get("name", "")) == strategy_name
            ),
            None,
        )

        if matching_strategy is not None:
            strategy_config.update(matching_strategy)
        if symbol_strategy is not None:
            strategy_config.update(symbol_strategy)

    if strategy_name is None:
        raise ValueError("Could not resolve a strategy to backtest.")

    strategy_config["name"] = strategy_name
    if interval_override is not None:
        strategy_config["interval"] = interval_override

    return strategy_config


def format_timestamp(timestamp: datetime) -> str:
    """Return an ISO timestamp string."""
    return timestamp.isoformat()


def calculate_trade_pnl(side: str, qty: int, entry_price: float, exit_price: float) -> float:
    """Return realized PnL for one long or short trade."""
    if side == "long":
        return (exit_price - entry_price) * qty
    return (entry_price - exit_price) * qty


def intrabar_exit_price(
    position: SimulatedPosition,
    bar: pd.Series,
) -> tuple[float | None, str | None]:
    """Return an intrabar stop-loss or take-profit fill if one was hit."""
    low_price = float(bar["Low"])
    high_price = float(bar["High"])

    if position.side == "long":
        stop_hit = low_price <= position.stop_price
        target_hit = high_price >= position.take_profit_price
        if stop_hit and target_hit:
            return position.stop_price, "stop_loss"
        if stop_hit:
            return position.stop_price, "stop_loss"
        if target_hit:
            return position.take_profit_price, "take_profit"
        return None, None

    stop_hit = high_price >= position.stop_price
    target_hit = low_price <= position.take_profit_price
    if stop_hit and target_hit:
        return position.stop_price, "stop_loss"
    if stop_hit:
        return position.stop_price, "stop_loss"
    if target_hit:
        return position.take_profit_price, "take_profit"
    return None, None


def unrealized_pnl(position: SimulatedPosition | None, mark_price: float) -> float:
    """Return unrealized PnL at a given mark price."""
    if position is None:
        return 0.0
    return calculate_trade_pnl(position.side, position.qty, position.entry_price, mark_price)


def build_backtest_position_context(position: SimulatedPosition, current_bar_number: int) -> PositionContext:
    """Build strategy exit context from the simulated open position."""
    return PositionContext(
        side=position.side,
        entry_price=position.entry_price,
        bars_in_trade=max(1, current_bar_number - position.entry_bar_number + 1),
        entry_signal_time=format_timestamp(position.entry_signal_time),
    )


def close_position(
    *,
    trade_number: int,
    position: SimulatedPosition,
    exit_time: datetime,
    exit_price: float,
    exit_reason: str,
    strategy_name: str,
    interval: str,
    symbol: str,
    current_bar_number: int,
) -> TradeRecord:
    """Build one trade record when a simulated position closes."""
    pnl = calculate_trade_pnl(position.side, position.qty, position.entry_price, exit_price)
    notional = position.entry_price * position.qty
    return_pct = (pnl / notional) * 100 if notional else 0.0
    return TradeRecord(
        trade_id=f"T{trade_number:03d}",
        symbol=symbol,
        strategy=strategy_name,
        interval=interval,
        side=position.side,
        qty=position.qty,
        entry_signal_time=format_timestamp(position.entry_signal_time),
        entry_time=format_timestamp(position.entry_time),
        entry_price=round(position.entry_price, 4),
        exit_time=format_timestamp(exit_time),
        exit_price=round(exit_price, 4),
        exit_reason=exit_reason,
        bars_held=max(1, current_bar_number - position.entry_bar_number + 1),
        pnl=round(pnl, 4),
        return_pct=round(return_pct, 4),
        stop_price=round(position.stop_price, 4),
        take_profit_price=round(position.take_profit_price, 4),
        stop_distance=round(abs(position.entry_price - position.stop_price), 4),
        stop_distance_frac_of_price=round(abs(position.entry_price - position.stop_price) / position.entry_price, 6),
        stop_source=position.stop_source,
        risk_reward_multiple=position.risk_reward_multiple,
        time_stop_triggered=exit_reason == "time_stop",
        macd_failure_triggered=exit_reason == "macd_failure",
    )


def simulate_backtest(
    *,
    symbol: str,
    strategy,
    prepared_df: pd.DataFrame,
    analysis_start_utc: datetime,
    analysis_end_utc: datetime,
    risk_settings: RiskSettings,
    initial_equity: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Run a true trade-by-trade simulation over a prepared DataFrame."""
    analysis_df = prepared_df.loc[
        (prepared_df.index >= analysis_start_utc) & (prepared_df.index < analysis_end_utc)
    ].copy()

    if analysis_df.empty:
        raise ValueError("No prepared strategy rows remain inside the requested backtest window.")

    realized_equity = float(initial_equity)
    position: SimulatedPosition | None = None
    pending_order: PendingOrder | None = None
    trades: list[TradeRecord] = []
    equity_points: list[dict] = []
    skipped_entries = {
        "actual_entry_rejected_count": 0,
        "actual_entry_rejected_max_stop_distance_count": 0,
        "actual_entry_invalid_stop_count": 0,
    }

    for bar_number, (timestamp, row) in enumerate(analysis_df.iterrows(), start=1):
        bar_open = float(row["Open"])
        bar_close = float(row["Close"])

        if pending_order is not None:
            if position is None and pending_order.action in {"BUY", "SELL"}:
                qty = calculate_order_qty(
                    bar_open,
                    realized_equity,
                    risk_settings.risk_fraction_of_buying_power,
                    risk_settings.max_position_qty,
                )
                resulting_side = "long" if pending_order.action == "BUY" else "short"
                exit_levels = build_exit_levels_from_entry_plan(
                    entry_price=bar_open,
                    position_side=resulting_side,
                    stop_loss_pct=risk_settings.stop_loss_pct,
                    take_profit_pct=risk_settings.take_profit_pct,
                    entry_risk_plan=pending_order.entry_risk_plan,
                )
                if exit_levels is None:
                    if pending_order.entry_risk_plan is not None:
                        validation = assess_stop_distance(
                            entry_price=bar_open,
                            stop_price=pending_order.entry_risk_plan.get("stop_price"),
                            position_side=resulting_side,
                            max_stop_distance_frac_of_price=pending_order.entry_risk_plan.get(
                                "max_stop_distance_frac_of_price"
                            ),
                        )
                        skipped_entries["actual_entry_rejected_count"] += 1
                        if validation["rejected_due_to_max_stop_distance"]:
                            skipped_entries["actual_entry_rejected_max_stop_distance_count"] += 1
                        else:
                            skipped_entries["actual_entry_invalid_stop_count"] += 1
                    pending_order = None
                    equity_points.append(
                        {
                            "timestamp": format_timestamp(timestamp),
                            "equity": round(realized_equity, 4),
                            "realized_equity": round(realized_equity, 4),
                            "position_side": "flat",
                            "position_qty": 0,
                            "close": round(bar_close, 4),
                        }
                    )
                    continue

                position = SimulatedPosition(
                    side=resulting_side,
                    qty=qty,
                    entry_time=timestamp,
                    entry_signal_time=pending_order.signal_time,
                    entry_price=bar_open,
                    entry_reason=pending_order.reason,
                    entry_bar_number=bar_number,
                    stop_price=exit_levels["stop_price"],
                    take_profit_price=exit_levels["take_profit_price"],
                    stop_source=exit_levels["stop_source"],
                    risk_reward_multiple=exit_levels["risk_reward_multiple"],
                )
            elif position is not None and pending_order.action == exit_action_for_position_side(position.side):
                closed_trade = close_position(
                    trade_number=len(trades) + 1,
                    position=position,
                    exit_time=timestamp,
                    exit_price=bar_open,
                    exit_reason=pending_order.reason,
                    strategy_name=strategy.name,
                    interval=strategy.interval,
                    symbol=symbol,
                    current_bar_number=bar_number,
                )
                trades.append(closed_trade)
                realized_equity += closed_trade.pnl
                position = None

            pending_order = None

        if position is not None:
            exit_price, exit_reason = intrabar_exit_price(position, row)
            if exit_reason is not None and exit_price is not None:
                closed_trade = close_position(
                    trade_number=len(trades) + 1,
                    position=position,
                    exit_time=timestamp,
                    exit_price=exit_price,
                    exit_reason=exit_reason,
                    strategy_name=strategy.name,
                    interval=strategy.interval,
                    symbol=symbol,
                    current_bar_number=bar_number,
                )
                trades.append(closed_trade)
                realized_equity += closed_trade.pnl
                position = None

        equity_points.append(
            {
                "timestamp": format_timestamp(timestamp),
                "equity": round(realized_equity + unrealized_pnl(position, bar_close), 4),
                "realized_equity": round(realized_equity, 4),
                "position_side": position.side if position is not None else "flat",
                "position_qty": position.qty if position is not None else 0,
                "close": round(bar_close, 4),
            }
        )

        next_bar_exists = bar_number < len(analysis_df)
        if not next_bar_exists:
            continue

        if position is not None:
            position_context = build_backtest_position_context(position, bar_number)
            exit_action = strategy.exit_action(row, position_context)
            if exit_action == exit_action_for_position_side(position.side):
                pending_order = PendingOrder(
                    action=exit_action,
                    reason=strategy.exit_reason(row, position_context),
                    signal_time=timestamp,
                )
        else:
            entry_action = strategy.entry_action(row)
            if entry_action in {"BUY", "SELL"}:
                pending_order = PendingOrder(
                    action=entry_action,
                    reason=strategy.entry_reason(),
                    signal_time=timestamp,
                    entry_risk_plan=strategy.build_entry_risk_plan(
                        row,
                        "long" if entry_action == "BUY" else "short",
                        risk_settings,
                    ),
                )

    if position is not None:
        final_timestamp = analysis_df.index[-1]
        final_close = float(analysis_df.iloc[-1]["Close"])
        closed_trade = close_position(
            trade_number=len(trades) + 1,
            position=position,
            exit_time=final_timestamp,
            exit_price=final_close,
            exit_reason="end_of_backtest",
            strategy_name=strategy.name,
            interval=strategy.interval,
            symbol=symbol,
            current_bar_number=len(analysis_df),
        )
        trades.append(closed_trade)
        realized_equity += closed_trade.pnl
        equity_points[-1]["equity"] = round(realized_equity, 4)
        equity_points[-1]["realized_equity"] = round(realized_equity, 4)
        equity_points[-1]["position_side"] = "flat"
        equity_points[-1]["position_qty"] = 0

    trade_columns = list(TradeRecord.__dataclass_fields__)
    trades_df = pd.DataFrame((asdict(trade) for trade in trades), columns=trade_columns)
    equity_df = pd.DataFrame(equity_points)
    summary = build_backtest_summary(
        symbol=symbol,
        strategy_name=strategy.name,
        strategy_version=strategy.version,
        interval=strategy.interval,
        analysis_start_utc=analysis_start_utc,
        analysis_end_utc=analysis_end_utc,
        initial_equity=initial_equity,
        ending_equity=realized_equity,
        analysis_df=analysis_df,
        trades_df=trades_df,
        equity_df=equity_df,
        strategy_parameters=strategy.build_strategy_parameters(),
        risk_settings=risk_settings,
        run_arguments={},
        skipped_entries=skipped_entries,
    )
    return trades_df, equity_df, summary


def max_drawdown_pct(equity_df: pd.DataFrame) -> float:
    """Return max drawdown percentage from an equity curve."""
    if equity_df.empty:
        return 0.0
    equity_series = equity_df["equity"].astype(float)
    running_max = equity_series.cummax()
    drawdown = (equity_series / running_max) - 1
    return float(drawdown.min() * 100)


def build_backtest_summary(
    *,
    symbol: str,
    strategy_name: str,
    strategy_version: str,
    interval: str,
    analysis_start_utc: datetime,
    analysis_end_utc: datetime,
    initial_equity: float,
    ending_equity: float,
    analysis_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    strategy_parameters: dict,
    risk_settings: RiskSettings,
    run_arguments: dict,
    skipped_entries: dict,
) -> dict:
    """Build a summary dictionary for the offline backtest run."""
    trade_count = int(len(trades_df))
    wins = int((trades_df["pnl"] > 0).sum()) if trade_count else 0
    losses = int((trades_df["pnl"] < 0).sum()) if trade_count else 0
    long_trades = int((trades_df["side"] == "long").sum()) if trade_count else 0
    short_trades = int((trades_df["side"] == "short").sum()) if trade_count else 0
    gross_profit = float(trades_df.loc[trades_df["pnl"] > 0, "pnl"].sum()) if trade_count else 0.0
    gross_loss = float(-trades_df.loc[trades_df["pnl"] < 0, "pnl"].sum()) if trade_count else 0.0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else None
    avg_trade_pnl = float(trades_df["pnl"].mean()) if trade_count else 0.0
    avg_trade_return_pct = float(trades_df["return_pct"].mean()) if trade_count else 0.0
    best_trade_pct = float(trades_df["return_pct"].max()) if trade_count else 0.0
    worst_trade_pct = float(trades_df["return_pct"].min()) if trade_count else 0.0
    exit_reason_counts = (
        trades_df["exit_reason"].value_counts().to_dict()
        if trade_count and "exit_reason" in trades_df.columns
        else {}
    )

    return {
        "symbol": symbol,
        "strategy": strategy_name,
        "strategy_version": strategy_version,
        "interval": interval,
        "analysis_start": analysis_start_utc.isoformat(),
        "analysis_end": analysis_end_utc.isoformat(),
        "analysis_bar_count": int(len(analysis_df)),
        "initial_equity": round(initial_equity, 2),
        "ending_equity": round(ending_equity, 2),
        "total_return_pct": round(((ending_equity / initial_equity) - 1) * 100, 4),
        "max_drawdown_pct": round(max_drawdown_pct(equity_df), 4),
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
        "exit_reason_counts": exit_reason_counts,
        "strategy_exit_counts": {
            "time_stop_count": int(exit_reason_counts.get("time_stop", 0)),
            "macd_failure_count": int(exit_reason_counts.get("macd_failure", 0)),
        },
        "effective_exit_model": {
            "stop_model": strategy_parameters.get("stop_model", "percent_of_entry"),
            "take_profit_model": strategy_parameters.get("take_profit_model", "percent_of_entry"),
            "take_profit_risk_multiple": strategy_parameters.get("take_profit_risk_multiple"),
            "max_stop_distance_frac_of_price": strategy_parameters.get("max_stop_distance_frac_of_price"),
        },
        "entry_filter_stats": {
            "setup_blocked_by_stop_filter_count": int(
                analysis_df.get("entry_blocked_by_stop_filter", pd.Series(dtype=bool)).sum()
            ),
            "setup_blocked_by_max_stop_distance_count": int(
                analysis_df.get("entry_blocked_by_max_stop_distance", pd.Series(dtype=bool)).sum()
            ),
            "preview_entry_invalid_stop_count": int(
                (
                    analysis_df.get("entry_blocked_by_stop_filter", pd.Series(dtype=bool)).fillna(False)
                    & ~analysis_df.get("entry_blocked_by_max_stop_distance", pd.Series(dtype=bool)).fillna(False)
                ).sum()
            ),
            **skipped_entries,
        },
        "strategy_context_counts": {
            "long_trend_bar_count": int(analysis_df.get("long_trend", pd.Series(dtype=bool)).sum()),
            "short_trend_bar_count": int(analysis_df.get("short_trend", pd.Series(dtype=bool)).sum()),
            "sideways_bar_count": int(analysis_df.get("sideways_market", pd.Series(dtype=bool)).sum()),
            "long_pullback_bar_count": int(analysis_df.get("pullback_active_long", pd.Series(dtype=bool)).sum()),
            "short_pullback_bar_count": int(analysis_df.get("pullback_active_short", pd.Series(dtype=bool)).sum()),
            "bullish_reentry_bar_count": int(
                analysis_df.get("bullish_reentry_trigger", pd.Series(dtype=bool)).sum()
            ),
            "bearish_reentry_bar_count": int(
                analysis_df.get("bearish_reentry_trigger", pd.Series(dtype=bool)).sum()
            ),
            "long_entry_signal_count": int(
                analysis_df.get("long_entry_signal", pd.Series(dtype=bool)).sum()
            ),
            "short_entry_signal_count": int(
                analysis_df.get("short_entry_signal", pd.Series(dtype=bool)).sum()
            ),
        },
        "strategy_parameters": strategy_parameters,
        "risk_settings": {
            "max_position_qty": risk_settings.max_position_qty,
            "risk_fraction_of_buying_power": risk_settings.risk_fraction_of_buying_power,
            "stop_loss_pct": risk_settings.stop_loss_pct,
            "take_profit_pct": risk_settings.take_profit_pct,
        },
        "run_arguments": run_arguments,
    }


def print_summary(
    summary: dict,
    *,
    trades_csv_path: Path,
    summary_json_path: Path,
    chart_path: Path | None,
) -> None:
    """Print a concise terminal summary after a backtest run."""
    print(f"Backtest: {summary['strategy']}")
    print(f"Symbol: {summary['symbol']}")
    print(f"Range: {summary['analysis_start']} -> {summary['analysis_end']}")
    print(f"Interval: {summary['interval']}")
    print()
    print(f"Trades: {summary['trade_count']}")
    print(f"Win rate: {summary['win_rate_pct']:.2f}%")
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
    if chart_path is not None:
        print(f"Chart: {chart_path}")
    trade_chart_dir = summary.get("trade_chart_dir")
    trade_chart_count = summary.get("trade_chart_count", 0)
    if trade_chart_dir:
        print(f"Trade charts: {trade_chart_dir} ({trade_chart_count})")


def make_run_slug(symbol: str, strategy_name: str, start_arg: str, end_arg: str) -> str:
    """Build a filesystem-friendly slug for one backtest run."""
    clean_start = start_arg.replace(":", "-").replace(" ", "_")
    clean_end = end_arg.replace(":", "-").replace(" ", "_")
    return f"{symbol.upper()}_{strategy_name}_{clean_start}_to_{clean_end}"


def make_run_timestamp() -> str:
    """Return a sortable timestamp for one backtest run."""
    return datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


def _aligned_timestamp_for_index(raw_timestamp, index: pd.Index) -> pd.Timestamp:
    """Convert a serialized trade timestamp into the same timezone shape as the DataFrame index."""
    timestamp = pd.Timestamp(raw_timestamp)
    if getattr(index, "tz", None) is None:
        return timestamp.tz_localize(None) if timestamp.tzinfo is not None else timestamp
    if timestamp.tzinfo is None:
        return timestamp.tz_localize(index.tz)
    return timestamp.tz_convert(index.tz)


def plot_trade_zoom_charts(
    *,
    prepared_df: pd.DataFrame,
    trades_df: pd.DataFrame,
    output_dir: Path,
    bars_before_entry: int = 24,
    bars_after_exit: int = 16,
) -> list[Path]:
    """Save one zoomed candlestick + MACD chart per completed trade."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle
    except ImportError:
        print("matplotlib is not installed, so the trade zoom charts were skipped.")
        return []

    if trades_df.empty:
        return []

    output_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    def draw_candlesticks(ax, window_df: pd.DataFrame) -> None:
        x_values = mdates.date2num(window_df.index.to_pydatetime())
        if len(x_values) > 1:
            width = float(pd.Series(x_values).diff().dropna().median()) * 0.82
        else:
            width = 1 / (24 * 60)

        for x_value, (_, row) in zip(x_values, window_df.iterrows()):
            open_price = float(row["Open"])
            close_price = float(row["Close"])
            high_price = float(row["High"])
            low_price = float(row["Low"])
            candle_color = "#2ca02c" if close_price >= open_price else "#d62728"
            ax.vlines(x_value, low_price, high_price, color=candle_color, linewidth=1.6, alpha=0.98)
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
                )
            )

    for trade_number, trade in enumerate(trades_df.to_dict("records"), start=1):
        entry_time = _aligned_timestamp_for_index(trade["entry_time"], prepared_df.index)
        exit_time = _aligned_timestamp_for_index(trade["exit_time"], prepared_df.index)
        entry_signal_time = _aligned_timestamp_for_index(trade["entry_signal_time"], prepared_df.index)
        trade_id = str(trade.get("trade_id", f"T{trade_number:03d}"))

        entry_loc = int(prepared_df.index.searchsorted(entry_time, side="left"))
        exit_loc = int(prepared_df.index.searchsorted(exit_time, side="left"))
        exit_loc = max(entry_loc, exit_loc)

        start_loc = max(0, entry_loc - bars_before_entry)
        end_loc = min(len(prepared_df), exit_loc + bars_after_exit + 1)
        window_df = prepared_df.iloc[start_loc:end_loc].copy()
        if window_df.empty:
            continue

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True, height_ratios=[3, 2])
        price_ax, macd_ax = axes
        fig.patch.set_facecolor("#121417")
        for axis in axes:
            axis.set_facecolor("#171b21")
            axis.tick_params(colors="#d8dee9")
            for spine in axis.spines.values():
                spine.set_color("#5f6b7a")

        draw_candlesticks(price_ax, window_df)
        if "EMA12" in window_df.columns:
            price_ax.plot(window_df.index, window_df["EMA12"], label="EMA12", linewidth=1.1, color="#5dade2")
        if "EMA26" in window_df.columns:
            price_ax.plot(window_df.index, window_df["EMA26"], label="EMA26", linewidth=1.1, color="#f5b041")
        if "EMA200_high" in window_df.columns:
            price_ax.plot(window_df.index, window_df["EMA200_high"], label="EMA200 high", linewidth=1.0, color="#ec7063")
        if "EMA200_close" in window_df.columns:
            price_ax.plot(window_df.index, window_df["EMA200_close"], label="EMA200 close", linewidth=1.0, color="#bb8fce")
        if "EMA200_low" in window_df.columns:
            price_ax.plot(window_df.index, window_df["EMA200_low"], label="EMA200 low", linewidth=1.0, color="#48c9b0")

        price_ax.scatter(entry_signal_time, float(trade["entry_price"]), marker="o", s=55, facecolors="none", edgecolors="#f8f9f9", label="Signal")
        price_ax.scatter(entry_time, float(trade["entry_price"]), marker="^" if trade["side"] == "long" else "v", s=70, color="#2ecc71" if trade["side"] == "long" else "#e74c3c", label="Entry")
        price_ax.scatter(exit_time, float(trade["exit_price"]), marker="x", s=70, color="#f8f9f9", label="Exit")
        price_ax.axhline(float(trade["stop_price"]), color="#e74c3c", linestyle="--", linewidth=1.0, alpha=0.8, label="Stop")
        price_ax.axhline(float(trade["take_profit_price"]), color="#2ecc71", linestyle="--", linewidth=1.0, alpha=0.8, label="Target")
        price_ax.set_ylabel("Price", color="#d8dee9")
        price_ax.set_title(
            f"{trade['symbol']} | {trade_id} | {trade['side']} | {trade['exit_reason']} | bars held {trade['bars_held']}"
            ,
            color="#f5f7fa",
        )
        price_ax.grid(alpha=0.25, color="#5f6b7a")
        price_legend = price_ax.legend(loc="upper left", ncol=3)
        price_legend.get_frame().set_facecolor("#171b21")
        price_legend.get_frame().set_edgecolor("#5f6b7a")
        for text in price_legend.get_texts():
            text.set_color("#f5f7fa")

        macd_ax.plot(window_df.index, window_df["MACD"], label="MACD", linewidth=1.0, color="#5dade2")
        macd_ax.plot(window_df.index, window_df["signal_line"], label="Signal", linewidth=1.0, color="#f5b041")
        bar_colors = ["#2ca02c" if value >= 0 else "#d62728" for value in window_df["histogram"].fillna(0)]
        macd_ax.bar(window_df.index, window_df["histogram"], label="Histogram", alpha=0.4, width=0.01, color=bar_colors)
        macd_ax.axhline(0, color="#f8f9f9", linewidth=0.8, alpha=0.6)
        macd_ax.set_ylabel("MACD", color="#d8dee9")
        macd_ax.set_xlabel("Time", color="#d8dee9")
        macd_ax.grid(alpha=0.25, color="#5f6b7a")
        macd_legend = macd_ax.legend(loc="upper left")
        macd_legend.get_frame().set_facecolor("#171b21")
        macd_legend.get_frame().set_edgecolor("#5f6b7a")
        for text in macd_legend.get_texts():
            text.set_color("#f5f7fa")

        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)
        macd_ax.xaxis.set_major_locator(locator)
        macd_ax.xaxis.set_major_formatter(formatter)

        fig.tight_layout()
        signed_return_pct = float(trade["return_pct"])
        filename = (
            f"{trade_id}_{trade['side']}_"
            f"{signed_return_pct:+.2f}pct_{trade['exit_reason']}.png"
        )
        output_path = output_dir / filename
        fig.savefig(output_path, dpi=150)
        plt.close(fig)
        saved_paths.append(output_path)

    return saved_paths


def plot_backtest_chart(
    *,
    prepared_df: pd.DataFrame,
    analysis_start_utc: datetime,
    analysis_end_utc: datetime,
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    summary: dict,
    output_path: Path,
) -> Path | None:
    """Save a simple chart image if matplotlib is available."""
    try:
        import matplotlib.dates as mdates
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed, so the chart image was skipped.")
        return None

    window_df = prepared_df.loc[
        (prepared_df.index >= analysis_start_utc) & (prepared_df.index < analysis_end_utc)
    ].copy()
    if window_df.empty:
        return None

    indicator_panels = 3 if "MACD" in window_df.columns else 2
    fig, axes = plt.subplots(indicator_panels, 1, figsize=(15, 10), sharex=True)

    if indicator_panels == 2:
        price_ax, equity_ax = axes
        indicator_ax = None
    else:
        price_ax, indicator_ax, equity_ax = axes

    price_ax.plot(window_df.index, window_df["Close"], label="Close", linewidth=1.2)
    if "EMA200" in window_df.columns:
        price_ax.plot(window_df.index, window_df["EMA200"], label="EMA200", linewidth=1.0)
    if "EMA200_close" in window_df.columns:
        price_ax.plot(window_df.index, window_df["EMA200_close"], label="EMA200 close", linewidth=1.0)
    if "SMA_FAST" in window_df.columns:
        price_ax.plot(window_df.index, window_df["SMA_FAST"], label="SMA fast", linewidth=1.0)
    if "SMA_SLOW" in window_df.columns:
        price_ax.plot(window_df.index, window_df["SMA_SLOW"], label="SMA slow", linewidth=1.0)

    if not trades_df.empty:
        long_entries = trades_df.loc[trades_df["side"] == "long"]
        short_entries = trades_df.loc[trades_df["side"] == "short"]
        price_ax.scatter(
            pd.to_datetime(long_entries["entry_time"]),
            long_entries["entry_price"],
            marker="^",
            color="green",
            s=50,
            label="Long entry",
        )
        price_ax.scatter(
            pd.to_datetime(short_entries["entry_time"]),
            short_entries["entry_price"],
            marker="v",
            color="red",
            s=50,
            label="Short entry",
        )
        price_ax.scatter(
            pd.to_datetime(trades_df["exit_time"]),
            trades_df["exit_price"],
            marker="x",
            color="black",
            s=35,
            label="Exit",
        )

    price_ax.set_title(
        f"{summary['symbol']} | {summary['strategy']} | Return {summary['total_return_pct']:.2f}% | Trades {summary['trade_count']}"
    )
    price_ax.set_ylabel("Price")
    price_ax.legend(loc="upper left")
    price_ax.grid(alpha=0.3)

    if indicator_ax is not None:
        indicator_ax.plot(window_df.index, window_df["MACD"], label="MACD", linewidth=1.0)
        indicator_ax.plot(window_df.index, window_df["signal_line"], label="Signal", linewidth=1.0)
        indicator_ax.bar(window_df.index, window_df["histogram"], label="Histogram", alpha=0.35, width=0.01)
        indicator_ax.axhline(0, color="black", linewidth=0.8, alpha=0.6)
        indicator_ax.set_ylabel("MACD")
        indicator_ax.legend(loc="upper left")
        indicator_ax.grid(alpha=0.3)

    equity_times = pd.to_datetime(equity_df["timestamp"])
    equity_ax.plot(equity_times, equity_df["equity"], label="Equity", color="tab:blue", linewidth=1.2)
    equity_ax.set_ylabel("Equity")
    equity_ax.set_xlabel("Time")
    equity_ax.grid(alpha=0.3)
    equity_ax.legend(loc="upper left")

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    equity_ax.xaxis.set_major_locator(locator)
    equity_ax.xaxis.set_major_formatter(formatter)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path


def build_output_paths(output_dir: str, slug: str, run_timestamp: str) -> tuple[Path, Path, Path, Path]:
    """Return the standard output paths for one backtest run."""
    root = Path(output_dir) / f"{run_timestamp}_{slug}"
    root.mkdir(parents=True, exist_ok=True)
    filename_prefix = f"{run_timestamp}_{slug}"
    return (
        root,
        root / f"{filename_prefix}_trades.csv",
        root / f"{filename_prefix}_summary.json",
        root / f"{filename_prefix}_chart.png",
    )


def run_backtest(args: argparse.Namespace) -> None:
    """Run one offline backtest from CLI arguments."""
    if args.list_strategies:
        print("\n".join(list_supported_strategies()))
        return

    from trading_bot.data import download_data_range

    raw_config = load_raw_config(args.config)
    config = load_config(args.config)
    strategy_config = resolve_strategy_config(raw_config, args.symbol, args.strategy, args.interval)
    strategy = create_strategy(strategy_config)

    analysis_start_local = parse_window_timestamp(args.start, is_end=False)
    analysis_end_local = parse_window_timestamp(args.end, is_end=True)
    if analysis_end_local <= analysis_start_local:
        raise ValueError("Backtest end must be after start.")

    analysis_start_utc = to_utc(analysis_start_local)
    analysis_end_utc = to_utc(analysis_end_local)
    resolved_output_dir = resolve_output_dir(args)

    raw_df = download_data_range(
        args.symbol.upper(),
        strategy.interval,
        analysis_start_utc,
        analysis_end_utc,
        warmup_bars=strategy.lookback_bars,
    )
    if raw_df.empty:
        raise ValueError("Alpaca returned no bars for the requested historical window.")

    prepared_df = strategy.prepare_dataframe(raw_df)
    trades_df, equity_df, summary = simulate_backtest(
        symbol=args.symbol.upper(),
        strategy=strategy,
        prepared_df=prepared_df,
        analysis_start_utc=analysis_start_utc,
        analysis_end_utc=analysis_end_utc,
        risk_settings=config.risk,
        initial_equity=args.initial_equity,
    )
    summary["run_arguments"] = {
        "symbol": args.symbol.upper(),
        "strategy": strategy.name,
        "strategy_requested": args.strategy,
        "config_path": str(args.config),
        "start": args.start,
        "end": args.end,
        "interval_override": args.interval,
        "resolved_interval": strategy.interval,
        "initial_equity": args.initial_equity,
        "output_dir": resolved_output_dir,
        "shared_run": bool(args.shared),
        "chart_enabled": not args.no_chart,
    }
    summary["resolved_strategy_config"] = strategy_config

    slug = make_run_slug(args.symbol.upper(), strategy.name, args.start, args.end)
    run_timestamp = make_run_timestamp()
    run_root, trades_csv_path, summary_json_path, chart_path = build_output_paths(
        resolved_output_dir,
        slug,
        run_timestamp,
    )
    trades_df.to_csv(trades_csv_path, index=False)

    saved_chart_path = None
    saved_trade_chart_paths: list[Path] = []
    if not args.no_chart:
        saved_chart_path = plot_backtest_chart(
            prepared_df=prepared_df,
            analysis_start_utc=analysis_start_utc,
            analysis_end_utc=analysis_end_utc,
            trades_df=trades_df,
            equity_df=equity_df,
            summary=summary,
            output_path=chart_path,
        )
        saved_trade_chart_paths = plot_trade_zoom_charts(
            prepared_df=prepared_df,
            trades_df=trades_df,
            output_dir=run_root / "trade_charts",
        )

    summary["trade_chart_count"] = len(saved_trade_chart_paths)
    summary["trade_chart_dir"] = None if not saved_trade_chart_paths else str((run_root / "trade_charts"))
    summary_json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_summary(summary, trades_csv_path=trades_csv_path, summary_json_path=summary_json_path, chart_path=saved_chart_path)


def main() -> None:
    """CLI entry point."""
    run_backtest(parse_args())


if __name__ == "__main__":
    main()
