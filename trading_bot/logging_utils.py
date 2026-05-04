"""Writes decision and trade logs for later review of bot behavior.

These logs are observational records for audit and analysis. The bot's
live operational decisions should rely on market data plus Alpaca state,
not on previously written log rows.
"""

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


DECISION_LOG_FIELDS = [
    "decision_id",
    "timestamp",
    "bot_version",
    "account",
    "symbol",
    "strategy",
    "strategy_version",
    "interval",
    "action",
    "reason",
    "latest_close",
    "entry_price",
    "buying_power",
    "in_position",
    "position_side",
    "position_qty",
    "qty",
    "market_is_open",
    "force_test_trade",
    "force_direction",
]

NOTABLE_DECISION_LOG_FIELDS = DECISION_LOG_FIELDS

TRADE_LOG_FIELDS = [
    "decision_id",
    "timestamp",
    "bot_version",
    "account",
    "symbol",
    "strategy",
    "strategy_version",
    "intended_action",
    "position_side_before",
    "position_side_after_expected",
    "strategy_reason",
    "order_id",
    "client_order_id",
    "order_type",
    "requested_qty",
    "filled_qty",
    "final_status",
    "decision_price",
    "filled_avg_price",
    "submitted_at",
    "filled_at",
    "canceled_at",
    "failed_at",
    "expired_at",
    "note",
]

STRATEGY_CONTEXT_LOG_PATH = "strategy_context_log.jsonl"
NOTABLE_DECISION_LOG_PATH = "notable_decision_log.csv"
NOTABLE_STRATEGY_CONTEXT_LOG_PATH = "notable_strategy_context_log.jsonl"

ENABLE_FULL_DECISION_LOG = True
ENABLE_FULL_STRATEGY_CONTEXT_LOG = True
ENABLE_NOTABLE_DECISION_LOG = True
ENABLE_NOTABLE_STRATEGY_CONTEXT_LOG = True


def _utc_timestamp() -> str:
    """Return a UTC timestamp in a CSV-friendly ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _normalize_row(row: dict, fieldnames: list[str]) -> dict:
    """Fill missing fields with blanks so the CSV schema stays stable."""
    return {field: row.get(field, "") for field in fieldnames}


def make_event_id() -> str:
    """Return a compact unique identifier for one logged bot decision."""
    return uuid4().hex


def is_notable_decision(action: str, reason: str) -> bool:
    """Return True when a decision reflects a real action rather than a hold."""
    return action != "HOLD"


def configure_logging(
    *,
    enable_full_decision_log: bool = True,
    enable_full_strategy_context_log: bool = True,
    enable_notable_decision_log: bool = True,
    enable_notable_strategy_context_log: bool = True,
) -> None:
    """Configure which local logs should receive new entries for this runtime."""
    global ENABLE_FULL_DECISION_LOG
    global ENABLE_FULL_STRATEGY_CONTEXT_LOG
    global ENABLE_NOTABLE_DECISION_LOG
    global ENABLE_NOTABLE_STRATEGY_CONTEXT_LOG

    ENABLE_FULL_DECISION_LOG = enable_full_decision_log
    ENABLE_FULL_STRATEGY_CONTEXT_LOG = enable_full_strategy_context_log
    ENABLE_NOTABLE_DECISION_LOG = enable_notable_decision_log
    ENABLE_NOTABLE_STRATEGY_CONTEXT_LOG = enable_notable_strategy_context_log


def _ensure_csv_header(path: Path, fieldnames: list[str]) -> None:
    """Create a CSV file with the expected header if it does not exist yet."""
    if path.exists() and path.stat().st_size > 0:
        return

    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()


def _append_csv_row(file_path: str, fieldnames: list[str], row: dict) -> None:
    """Append one row while preserving a flat, consistent CSV layout."""
    path = Path(file_path)
    _ensure_csv_header(path, fieldnames)
    with path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writerow(_normalize_row(row, fieldnames))


def _append_jsonl_row(file_path: str, row: dict) -> None:
    """Append one JSON object per line for flexible log schemas."""
    path = Path(file_path)
    with path.open("a", encoding="utf-8") as jsonl_file:
        jsonl_file.write(json.dumps(row, separators=(",", ":")) + "\n")


def log_trade(trade_data: dict) -> None:
    """Record the final known state of an Alpaca order in the trade log."""
    row = _normalize_row(trade_data, TRADE_LOG_FIELDS)
    if not row["timestamp"]:
        row["timestamp"] = _utc_timestamp()
    _append_csv_row("trade_log.csv", TRADE_LOG_FIELDS, row)


def log_decision(decision_data: dict) -> None:
    """Record one bot decision, including non-trades and safety skips."""
    row = _normalize_row(decision_data, DECISION_LOG_FIELDS)
    if not row["timestamp"]:
        row["timestamp"] = _utc_timestamp()
    if ENABLE_FULL_DECISION_LOG:
        _append_csv_row("decision_log.csv", DECISION_LOG_FIELDS, row)
    if ENABLE_NOTABLE_DECISION_LOG and is_notable_decision(row["action"], row["reason"]):
        _append_csv_row(NOTABLE_DECISION_LOG_PATH, NOTABLE_DECISION_LOG_FIELDS, row)


def log_strategy_context(context_data: dict) -> None:
    """Record flexible strategy-specific context as JSONL for later analysis."""
    row = dict(context_data)
    if not row.get("timestamp"):
        row["timestamp"] = _utc_timestamp()
    if ENABLE_FULL_STRATEGY_CONTEXT_LOG:
        _append_jsonl_row(STRATEGY_CONTEXT_LOG_PATH, row)


def log_notable_strategy_context(context_data: dict) -> None:
    """Record only non-routine strategy context entries for lighter analysis workflows."""
    row = dict(context_data)
    if not row.get("timestamp"):
        row["timestamp"] = _utc_timestamp()
    if ENABLE_NOTABLE_STRATEGY_CONTEXT_LOG:
        _append_jsonl_row(NOTABLE_STRATEGY_CONTEXT_LOG_PATH, row)
