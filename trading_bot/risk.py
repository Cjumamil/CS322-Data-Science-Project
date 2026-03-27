"""Applies simple position sizing and exit rules for the bot."""

import math


def calculate_order_qty(
    price: float,
    buying_power: float,
    risk_fraction_of_buying_power: float,
    max_position_qty: int,
) -> int:
    """Choose a share quantity based on buying power and a risk budget."""
    risk_budget = buying_power * risk_fraction_of_buying_power
    qty = math.floor(risk_budget / price)

    # Keep the order size in a small, practical range for this project.
    qty = max(1, qty)
    qty = min(qty, max_position_qty)

    return qty


def should_exit_position(
    current_price: float,
    entry_price: float,
    stop_loss_pct: float,
    take_profit_pct: float,
):
    """Return an exit reason when stop-loss or take-profit is triggered."""
    if current_price <= entry_price * (1 - stop_loss_pct):
        return "stop_loss"
    if current_price >= entry_price * (1 + take_profit_pct):
        return "take_profit"
    return None
