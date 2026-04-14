"""Applies simple position sizing and exit rules for the bot."""

import math
from datetime import datetime
from zoneinfo import ZoneInfo


NEW_YORK_TIMEZONE = ZoneInfo("America/New_York")
REGULAR_MARKET_OPEN_MINUTE = (9 * 60) + 30
REGULAR_MARKET_CLOSE_MINUTE = 16 * 60


def calculate_order_qty(
    price: float,
    buying_power: float,
    risk_fraction_of_buying_power: float,
    max_position_qty: int,
) -> int:
    """Choose a share quantity based on buying power and a risk budget.

    Positive ``max_position_qty`` values apply a hard cap. ``0`` or any lower
    value disables the cap entirely.
    """
    risk_budget = buying_power * risk_fraction_of_buying_power
    qty = math.floor(risk_budget / price)

    # Keep the order size practical, and only apply the share cap when enabled.
    qty = max(1, qty)
    if max_position_qty >= 1:
        qty = min(qty, max_position_qty)

    return qty


def round_price(price: float) -> float:
    """Round a price to two decimals for stock-style order levels."""
    return round(price, 2)


def is_within_flatten_before_close_window(
    timestamp: datetime | None,
    *,
    flatten_before_close: bool,
    flatten_minutes_before_close: int,
) -> bool:
    """Return True when the timestamp falls inside the configured pre-close flatten window."""
    if not flatten_before_close or timestamp is None or flatten_minutes_before_close <= 0:
        return False

    if timestamp.tzinfo is None:
        local_timestamp = timestamp.replace(tzinfo=NEW_YORK_TIMEZONE)
    else:
        local_timestamp = timestamp.astimezone(NEW_YORK_TIMEZONE)

    minutes_of_day = (local_timestamp.hour * 60) + local_timestamp.minute
    return (
        minutes_of_day >= (REGULAR_MARKET_CLOSE_MINUTE - flatten_minutes_before_close)
        and minutes_of_day < REGULAR_MARKET_CLOSE_MINUTE
    )


def assess_stop_distance(
    *,
    entry_price: float | None,
    stop_price: float | None,
    position_side: str,
    max_stop_distance_frac_of_price: float | None = None,
) -> dict:
    """Validate stop placement and measure its distance from entry.

    This helper is shared by strategy preview logic and the concrete trade
    execution/backtest path so both use the same safety checks.
    """
    result = {
        "entry_price": entry_price,
        "stop_price": stop_price,
        "stop_distance": None,
        "stop_distance_frac_of_price": None,
        "max_stop_distance_frac_of_price": max_stop_distance_frac_of_price,
        "rejected_due_to_max_stop_distance": False,
        "is_valid": False,
        "rejection_reason": None,
    }

    if entry_price is None or entry_price <= 0:
        result["rejection_reason"] = "invalid_entry_price"
        return result

    if stop_price is None:
        result["rejection_reason"] = "missing_stop_price"
        return result

    entry_price = float(entry_price)
    stop_price = float(stop_price)
    result["entry_price"] = entry_price
    result["stop_price"] = stop_price

    if position_side == "long" and stop_price >= entry_price:
        result["rejection_reason"] = "stop_not_below_entry"
        return result
    if position_side == "short" and stop_price <= entry_price:
        result["rejection_reason"] = "stop_not_above_entry"
        return result
    if position_side not in {"long", "short"}:
        result["rejection_reason"] = "invalid_position_side"
        return result

    stop_distance = abs(entry_price - stop_price)
    if stop_distance <= 0:
        result["rejection_reason"] = "non_positive_stop_distance"
        return result

    stop_distance_frac = stop_distance / entry_price
    result["stop_distance"] = stop_distance
    result["stop_distance_frac_of_price"] = stop_distance_frac

    if max_stop_distance_frac_of_price is not None and max_stop_distance_frac_of_price > 0:
        result["max_stop_distance_frac_of_price"] = float(max_stop_distance_frac_of_price)
        if stop_distance_frac > float(max_stop_distance_frac_of_price):
            result["rejected_due_to_max_stop_distance"] = True
            result["rejection_reason"] = "max_stop_distance_exceeded"
            return result

    result["is_valid"] = True
    return result


def protective_stop_price(
    entry_price: float,
    position_side: str,
    stop_loss_pct: float,
) -> float | None:
    """Return the stop price that protects the current position side."""
    if position_side == "long":
        return round_price(entry_price * (1 - stop_loss_pct))
    if position_side == "short":
        return round_price(entry_price * (1 + stop_loss_pct))
    return None


def take_profit_price(
    entry_price: float,
    position_side: str,
    take_profit_pct: float,
) -> float | None:
    """Return a simple percent-based take-profit price."""
    if position_side == "long":
        return round_price(entry_price * (1 + take_profit_pct))
    if position_side == "short":
        return round_price(entry_price * (1 - take_profit_pct))
    return None


def take_profit_price_from_risk(
    entry_price: float,
    stop_price: float,
    position_side: str,
    risk_reward_multiple: float,
) -> float | None:
    """Return a take-profit price using an absolute risk distance."""
    if position_side == "long":
        risk_per_share = entry_price - stop_price
        if risk_per_share <= 0:
            return None
        return round_price(entry_price + (risk_per_share * risk_reward_multiple))

    if position_side == "short":
        risk_per_share = stop_price - entry_price
        if risk_per_share <= 0:
            return None
        return round_price(entry_price - (risk_per_share * risk_reward_multiple))

    return None


def build_exit_levels_from_entry_price(
    entry_price: float,
    position_side: str,
    stop_loss_pct: float,
    take_profit_pct: float,
) -> dict | None:
    """Build default percent-based exit levels from an entry price."""
    stop_price = protective_stop_price(entry_price, position_side, stop_loss_pct)
    target_price = take_profit_price(entry_price, position_side, take_profit_pct)
    if stop_price is None or target_price is None:
        return None
    return {
        "stop_price": stop_price,
        "take_profit_price": target_price,
        "stop_source": "percent_of_entry",
        "risk_reward_multiple": None,
        "entry_signal_time": None,
        "stop_distance": abs(entry_price - stop_price),
        "stop_distance_frac_of_price": abs(entry_price - stop_price) / entry_price if entry_price > 0 else None,
        "max_stop_distance_frac_of_price": None,
    }


def build_exit_levels_from_entry_plan(
    *,
    entry_price: float,
    position_side: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    entry_risk_plan: dict | None,
) -> dict | None:
    """Build concrete stop-loss and take-profit prices for a new trade."""
    if entry_risk_plan is None:
        return build_exit_levels_from_entry_price(
            entry_price,
            position_side,
            stop_loss_pct,
            take_profit_pct,
        )

    stop_price = entry_risk_plan.get("stop_price")
    risk_reward_multiple = entry_risk_plan.get("risk_reward_multiple")
    stop_source = entry_risk_plan.get("stop_source", "custom")
    max_stop_distance_frac_of_price = entry_risk_plan.get("max_stop_distance_frac_of_price")
    if stop_price is None:
        return build_exit_levels_from_entry_price(
            entry_price,
            position_side,
            stop_loss_pct,
            take_profit_pct,
        )

    stop_price = round_price(float(stop_price))
    stop_validation = assess_stop_distance(
        entry_price=entry_price,
        stop_price=stop_price,
        position_side=position_side,
        max_stop_distance_frac_of_price=max_stop_distance_frac_of_price,
    )
    if not stop_validation["is_valid"]:
        return None

    if risk_reward_multiple is not None:
        target_price = take_profit_price_from_risk(
            entry_price,
            stop_price,
            position_side,
            float(risk_reward_multiple),
        )
        if target_price is None:
            return None
    else:
        target_price = take_profit_price(entry_price, position_side, take_profit_pct)
        if target_price is None:
            return None

    return {
        "stop_price": stop_price,
        "take_profit_price": target_price,
        "stop_source": stop_source,
        "risk_reward_multiple": risk_reward_multiple,
        "entry_signal_time": None if entry_risk_plan is None else entry_risk_plan.get("entry_signal_time"),
        "stop_distance": stop_validation["stop_distance"],
        "stop_distance_frac_of_price": stop_validation["stop_distance_frac_of_price"],
        "max_stop_distance_frac_of_price": stop_validation["max_stop_distance_frac_of_price"],
    }


def build_exit_levels_for_live_position(
    *,
    entry_price: float,
    position_side: str,
    stop_loss_pct: float,
    take_profit_pct: float,
    protective_stop_order=None,
    active_exit_levels: dict | None = None,
    risk_reward_multiple: float | None = None,
) -> dict | None:
    """Resolve the effective live exit levels for an open trade.

    Preference order:
    1. Session-tracked active exit levels from the original entry
    2. Broker protective stop plus strategy risk-reward multiple
    3. Default percent-based exits
    """
    stop_price = None
    take_profit_target = None
    stop_source = "percent_of_entry"
    stop_distance = None
    stop_distance_frac_of_price = None
    max_stop_distance_frac_of_price = None

    if active_exit_levels is not None:
        raw_stop_price = active_exit_levels.get("stop_price")
        raw_take_profit = active_exit_levels.get("take_profit_price")
        if raw_stop_price is not None:
            stop_price = round_price(float(raw_stop_price))
            stop_source = str(active_exit_levels.get("stop_source", "session_active"))
        if raw_take_profit is not None:
            take_profit_target = round_price(float(raw_take_profit))
        stop_distance = active_exit_levels.get("stop_distance")
        stop_distance_frac_of_price = active_exit_levels.get("stop_distance_frac_of_price")
        max_stop_distance_frac_of_price = active_exit_levels.get("max_stop_distance_frac_of_price")
        if risk_reward_multiple is None:
            raw_multiple = active_exit_levels.get("risk_reward_multiple")
            if raw_multiple is not None:
                risk_reward_multiple = float(raw_multiple)

    if stop_price is None and protective_stop_order is not None:
        broker_stop_price = getattr(protective_stop_order, "stop_price", None)
        if broker_stop_price not in {None, ""}:
            stop_price = round_price(float(broker_stop_price))
            stop_source = "broker_protective_stop"

    if stop_price is None:
        return build_exit_levels_from_entry_price(
            entry_price,
            position_side,
            stop_loss_pct,
            take_profit_pct,
        )

    if stop_distance is None or stop_distance_frac_of_price is None:
        stop_validation = assess_stop_distance(
            entry_price=entry_price,
            stop_price=stop_price,
            position_side=position_side,
            max_stop_distance_frac_of_price=max_stop_distance_frac_of_price,
        )
        if not stop_validation["is_valid"]:
            return None
        stop_distance = stop_validation["stop_distance"]
        stop_distance_frac_of_price = stop_validation["stop_distance_frac_of_price"]
        max_stop_distance_frac_of_price = stop_validation["max_stop_distance_frac_of_price"]

    if take_profit_target is None:
        if risk_reward_multiple is not None:
            take_profit_target = take_profit_price_from_risk(
                entry_price,
                stop_price,
                position_side,
                float(risk_reward_multiple),
            )
        else:
            take_profit_target = take_profit_price(
                entry_price,
                position_side,
                take_profit_pct,
            )

    if take_profit_target is None:
        return None

    return {
        "stop_price": stop_price,
        "take_profit_price": take_profit_target,
        "stop_source": stop_source,
        "risk_reward_multiple": risk_reward_multiple,
        "entry_signal_time": None if active_exit_levels is None else active_exit_levels.get("entry_signal_time"),
        "stop_distance": stop_distance,
        "stop_distance_frac_of_price": stop_distance_frac_of_price,
        "max_stop_distance_frac_of_price": max_stop_distance_frac_of_price,
    }


def exit_action_for_position_side(position_side: str) -> str | None:
    """Return the market action needed to flatten the current position."""
    if position_side == "long":
        return "SELL"
    if position_side == "short":
        return "BUY"
    return None


def should_exit_position(
    current_price: float,
    position_side: str,
    stop_price: float,
    take_profit_price: float,
):
    """Return an exit reason when stop-loss or take-profit is triggered."""
    if position_side == "long":
        if current_price <= stop_price:
            return "stop_loss"
        if current_price >= take_profit_price:
            return "take_profit"
        return None

    if position_side == "short":
        if current_price >= stop_price:
            return "stop_loss"
        if current_price <= take_profit_price:
            return "take_profit"
        return None

    return None
