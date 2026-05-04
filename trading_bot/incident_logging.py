"""Tracks blocker incidents as lifecycle events instead of per-cycle spam."""

import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4


BLOCKER_INCIDENT_LOG_PATH = Path("blocker_incident_log.jsonl")
BLOCKER_INCIDENT_STATE_PATH = Path("blocker_incident_state.json")
BLOCKER_INCIDENT_HEARTBEAT_SECONDS = 30 * 60

POLICY_HOLD_REASONS = {
    "asset_not_tradable",
    "asset_not_shortable",
    "asset_not_easy_to_borrow",
    "portfolio_exposure_cap_reached",
}

BLOCKER_HOLD_REASONS = {
    "pending_order_lookup_failed",
    "broker_open_order_exists",
    "stale_protective_stop_exists",
    "invalid_entry_risk_plan",
}

ERROR_REASONS = {
    "data_download_failed",
    "data_download_value_error",
    "no_market_data",
    "not_enough_data",
    "account_lookup_failed",
    "symbol_cycle_exception",
}


def _utc_timestamp() -> str:
    """Return a UTC timestamp in a compact ISO-8601 format."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _load_incident_state(path: Path = BLOCKER_INCIDENT_STATE_PATH) -> dict:
    """Load the small local incident-state file."""
    if not path.exists():
        return {}

    try:
        with path.open("r", encoding="utf-8") as state_file:
            data = json.load(state_file)
    except (OSError, ValueError) as exc:
        print(f"Could not load blocker incident state: {exc}")
        return {}

    return data if isinstance(data, dict) else {}


def _persist_incident_state(state: dict, path: Path = BLOCKER_INCIDENT_STATE_PATH) -> None:
    """Persist the current open-incident snapshot."""
    try:
        with path.open("w", encoding="utf-8") as state_file:
            json.dump(state, state_file, indent=2, sort_keys=True)
    except OSError as exc:
        print(f"Could not persist blocker incident state: {exc}")


def _append_incident_event(event: dict, path: Path = BLOCKER_INCIDENT_LOG_PATH) -> None:
    """Append one lifecycle event to the blocker incident history log."""
    with path.open("a", encoding="utf-8") as log_file:
        log_file.write(json.dumps(event, separators=(",", ":")) + "\n")


def classify_incident(action: str, reason: str) -> str | None:
    """Classify a cycle outcome as an incident category or return None."""
    if action == "HOLD" and reason in POLICY_HOLD_REASONS:
        return "policy_hold"
    if action == "HOLD" and reason in BLOCKER_HOLD_REASONS:
        return "operational_blocker"
    if action == "ERROR" or reason in ERROR_REASONS:
        return "runtime_error"
    return None


def _incident_key(account_name: str, symbol: str, action: str, reason: str, category: str) -> str:
    """Build a stable key for one currently-open incident."""
    return "|".join((account_name, symbol, action, reason, category))


def sync_incident_state(
    *,
    account_name: str,
    symbol: str,
    action: str,
    reason: str,
    bot_version: str,
    strategy_name: str,
    strategy_version: str,
    interval: str,
    context: dict | None = None,
) -> None:
    """Open, heartbeat, or resolve blocker incidents for one account/symbol."""
    category = classify_incident(action, reason)
    current_key = None if category is None else _incident_key(account_name, symbol, action, reason, category)
    state = _load_incident_state()
    now = _utc_timestamp()
    current_context = dict(context or {})

    symbol_keys = [
        key
        for key, record in state.items()
        if record.get("account") == account_name and record.get("symbol") == symbol
    ]

    for key in symbol_keys:
        if key == current_key:
            continue

        record = state.pop(key)
        _append_incident_event(
            {
                "event_id": uuid4().hex,
                "timestamp": now,
                "event_type": "resolved",
                "category": record["category"],
                "action": record["action"],
                "reason": record["reason"],
                "account": record["account"],
                "symbol": record["symbol"],
                "bot_version": record["bot_version"],
                "strategy": record["strategy"],
                "strategy_version": record["strategy_version"],
                "interval": record["interval"],
                "first_seen_at": record["first_seen_at"],
                "last_seen_at": record["last_seen_at"],
                "resolved_at": now,
                "occurrence_count": record["occurrence_count"],
                "resolved_by_action": action,
                "resolved_by_reason": reason,
                "context": record.get("context", {}),
            }
        )

    if current_key is None:
        _persist_incident_state(state)
        return

    if current_key not in state:
        record = {
            "category": category,
            "action": action,
            "reason": reason,
            "account": account_name,
            "symbol": symbol,
            "bot_version": bot_version,
            "strategy": strategy_name,
            "strategy_version": strategy_version,
            "interval": interval,
            "first_seen_at": now,
            "last_seen_at": now,
            "last_heartbeat_at": now,
            "occurrence_count": 1,
            "context": current_context,
        }
        state[current_key] = record
        _append_incident_event(
            {
                "event_id": uuid4().hex,
                "timestamp": now,
                "event_type": "opened",
                "category": category,
                "action": action,
                "reason": reason,
                "account": account_name,
                "symbol": symbol,
                "bot_version": bot_version,
                "strategy": strategy_name,
                "strategy_version": strategy_version,
                "interval": interval,
                "first_seen_at": now,
                "last_seen_at": now,
                "occurrence_count": 1,
                "context": current_context,
            }
        )
        _persist_incident_state(state)
        return

    record = state[current_key]
    record["last_seen_at"] = now
    record["occurrence_count"] = int(record.get("occurrence_count", 0)) + 1
    record["context"] = current_context

    last_heartbeat_at = record.get("last_heartbeat_at", record["first_seen_at"])
    last_heartbeat_dt = datetime.fromisoformat(last_heartbeat_at.replace("Z", "+00:00"))
    current_dt = datetime.fromisoformat(now.replace("Z", "+00:00"))
    if (current_dt - last_heartbeat_dt).total_seconds() >= BLOCKER_INCIDENT_HEARTBEAT_SECONDS:
        record["last_heartbeat_at"] = now
        _append_incident_event(
            {
                "event_id": uuid4().hex,
                "timestamp": now,
                "event_type": "heartbeat",
                "category": category,
                "action": action,
                "reason": reason,
                "account": account_name,
                "symbol": symbol,
                "bot_version": bot_version,
                "strategy": strategy_name,
                "strategy_version": strategy_version,
                "interval": interval,
                "first_seen_at": record["first_seen_at"],
                "last_seen_at": now,
                "occurrence_count": record["occurrence_count"],
                "context": current_context,
            }
        )

    state[current_key] = record
    _persist_incident_state(state)
