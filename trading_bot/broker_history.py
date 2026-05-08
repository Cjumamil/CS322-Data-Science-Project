"""Reconcile broker-side final order states into the local trade log."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
from alpaca.common.enums import Sort
from alpaca.trading.enums import QueryOrderStatus
from alpaca.trading.requests import GetOrdersRequest

from trading_bot.config import AccountAssignment
from trading_bot.logging_utils import log_trade


TRADE_LOG_PATH = Path("trade_log.csv")
DEFAULT_BROKER_RECONCILIATION_LOOKBACK_DAYS = 30
BROKER_RECONCILIATION_OVERLAP_DAYS = 2


def _load_trade_log(path: Path = TRADE_LOG_PATH) -> pd.DataFrame:
    """Load the local trade log if it exists."""
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, dtype=str).fillna("")


def _latest_account_fill_timestamp(trade_log_df: pd.DataFrame, account_name: str) -> pd.Timestamp | None:
    """Return the latest filled timestamp already logged for one account."""
    if trade_log_df.empty:
        return None
    account_df = trade_log_df[trade_log_df["account"].astype(str) == account_name].copy()
    if account_df.empty:
        return None
    latest_timestamp = account_df["filled_at"].apply(_parse_broker_timestamp).max()
    return None if pd.isna(latest_timestamp) else latest_timestamp


def _account_symbol_strategy_index(account_config: AccountAssignment) -> dict[str, dict]:
    """Index one account's configured symbols so broker rows can be labeled locally."""
    return {
        symbol_assignment.ticker.upper(): {
            "strategy_name": symbol_assignment.strategy.name,
            "strategy_version": symbol_assignment.strategy.version,
        }
        for symbol_assignment in account_config.symbols
    }


def _normalize_broker_timestamp(raw_value) -> str:
    """Convert one Alpaca timestamp into the repo's standard CSV-friendly text."""
    timestamp = pd.Timestamp(raw_value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp.isoformat()


def _normalize_optional_broker_timestamp(raw_value) -> str:
    """Normalize one optional Alpaca timestamp and leave missing values blank."""
    if raw_value in {None, ""}:
        return ""
    return _normalize_broker_timestamp(raw_value)


def _parse_broker_timestamp(raw_value) -> pd.Timestamp | pd.NaT:
    """Parse one persisted broker timestamp while tolerating mixed text formats."""
    if raw_value in {None, ""}:
        return pd.NaT
    raw_text = str(raw_value).strip()
    if not raw_text or raw_text.lower() in {"none", "nan", "nat"}:
        return pd.NaT
    timestamp = pd.Timestamp(raw_text)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def sync_account_trade_log_with_broker(
    *,
    trading_client,
    account_config: AccountAssignment,
    lookback_days: int = DEFAULT_BROKER_RECONCILIATION_LOOKBACK_DAYS,
    start_utc: pd.Timestamp | None = None,
) -> dict:
    """Append missing filled broker-side stop orders to the local trade log."""
    trade_log_df = _load_trade_log()
    seen_order_ids = {
        str(order_id).strip()
        for order_id in trade_log_df.get("order_id", pd.Series(dtype=str)).tolist()
        if str(order_id).strip()
    }
    latest_logged_fill = _latest_account_fill_timestamp(trade_log_df, account_config.name)
    if start_utc is not None:
        request_start = start_utc
    elif latest_logged_fill is not None:
        request_start = latest_logged_fill - pd.Timedelta(days=BROKER_RECONCILIATION_OVERLAP_DAYS)
    else:
        request_start = pd.Timestamp(datetime.now(timezone.utc) - timedelta(days=lookback_days))
    request_end = pd.Timestamp(datetime.now(timezone.utc) + timedelta(days=1))

    order_request = GetOrdersRequest(
        status=QueryOrderStatus.CLOSED,
        limit=500,
        after=request_start.to_pydatetime(),
        until=request_end.to_pydatetime(),
        direction=Sort.ASC,
        nested=False,
    )
    orders = list(trading_client.get_orders(filter=order_request))
    assignment_index = _account_symbol_strategy_index(account_config)
    appended_count = 0

    for order in orders:
        order_id = str(getattr(order, "id", "")).strip()
        if not order_id or order_id in seen_order_ids:
            continue

        status = str(getattr(getattr(order, "status", None), "value", getattr(order, "status", ""))).lower()
        if status != "filled":
            continue

        order_type = str(
            getattr(getattr(order, "order_type", None), "value", getattr(order, "order_type", ""))
        ).lower()
        if "stop" not in order_type:
            continue

        symbol = str(getattr(order, "symbol", "")).upper()
        assignment = assignment_index.get(symbol)
        if assignment is None:
            continue

        intended_action = str(getattr(getattr(order, "side", None), "value", getattr(order, "side", ""))).upper()
        if intended_action == "SELL":
            position_side_before = "long"
        elif intended_action == "BUY":
            position_side_before = "short"
        else:
            continue

        log_trade(
            {
                "decision_id": f"broker_history:{account_config.name}:{order_id}",
                "timestamp": _normalize_broker_timestamp(getattr(order, "filled_at", None)),
                "bot_version": "",
                "account": account_config.name,
                "symbol": symbol,
                "strategy": assignment["strategy_name"],
                "strategy_version": assignment["strategy_version"],
                "intended_action": intended_action,
                "position_side_before": position_side_before,
                "position_side_after_expected": "flat",
                "strategy_reason": "stop_loss",
                "order_id": order_id,
                "client_order_id": str(getattr(order, "client_order_id", "")).strip(),
                "order_type": order_type,
                "requested_qty": str(getattr(order, "qty", "")),
                "filled_qty": str(getattr(order, "filled_qty", "")),
                "final_status": status,
                "decision_price": "",
                "filled_avg_price": str(getattr(order, "filled_avg_price", "")),
                "submitted_at": _normalize_optional_broker_timestamp(getattr(order, "submitted_at", None)),
                "filled_at": _normalize_broker_timestamp(getattr(order, "filled_at", None)),
                "canceled_at": _normalize_optional_broker_timestamp(getattr(order, "canceled_at", None)),
                "failed_at": _normalize_optional_broker_timestamp(getattr(order, "failed_at", None)),
                "expired_at": _normalize_optional_broker_timestamp(getattr(order, "expired_at", None)),
                "note": "broker_history_stop_fill",
            }
        )
        seen_order_ids.add(order_id)
        appended_count += 1

    return {
        "account": account_config.name,
        "request_start": request_start.isoformat(),
        "request_end": request_end.isoformat(),
        "scanned_order_count": len(orders),
        "appended_order_count": appended_count,
    }
