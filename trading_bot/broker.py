"""Handles Alpaca connection, account access, order submission, and fill polling."""

from dotenv import load_dotenv
import os
import time

from alpaca.trading.client import TradingClient
from alpaca.trading.enums import OrderSide, QueryOrderStatus, TimeInForce
from alpaca.trading.requests import GetOrderByIdRequest, GetOrdersRequest, MarketOrderRequest, StopOrderRequest

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")
DEFAULT_API_REQUEST_TIMEOUT_SECONDS = 20

FINAL_ORDER_STATUSES = {
    "filled",
    "canceled",
    "expired",
    "rejected",
    "done_for_day",
    "replaced",
    "stopped",
    "suspended",
    "calculated",
}


def _serialize_enum(value) -> str:
    """Convert Alpaca enum-like values into lower-case strings."""
    if hasattr(value, "value"):
        value = value.value
    return str(value).lower()


def _safe_float(value) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _symbol_matches(order, symbol: str) -> bool:
    """Return True when the order belongs to the requested symbol."""
    return str(getattr(order, "symbol", "")).upper() == symbol.upper()


def get_order_side(order) -> str:
    """Return the lower-case side for an Alpaca order."""
    return _serialize_enum(getattr(order, "side", ""))


def get_position_side(position) -> str:
    """Return `flat`, `long`, or `short` from an Alpaca position object."""
    if position is None:
        return "flat"

    side = _serialize_enum(getattr(position, "side", ""))
    if side in {"long", "short"}:
        return side

    qty = _safe_float(getattr(position, "qty", 0))
    if qty < 0:
        return "short"
    if qty > 0:
        return "long"
    return "flat"


def get_position_qty(position) -> int:
    """Return the absolute share quantity for an Alpaca position."""
    return int(abs(_safe_float(getattr(position, "qty", 0))))


def connect_alpaca(*, paper: bool = True) -> TradingClient:
    """Create an Alpaca client from environment variables."""
    if not API_KEY or not SECRET_KEY:
        raise ValueError(
            "Missing Alpaca credentials. Make sure your .env file contains ALPACA_API_KEY and ALPACA_SECRET_KEY."
        )

    trading_client = TradingClient(API_KEY, SECRET_KEY, paper=paper)
    _inject_default_request_timeout(trading_client, DEFAULT_API_REQUEST_TIMEOUT_SECONDS)
    return trading_client


def _inject_default_request_timeout(trading_client: TradingClient, timeout_seconds: int) -> None:
    """Inject a default timeout into the Alpaca client's shared requests session."""
    session = getattr(trading_client, "_session", None)
    if session is None or getattr(session, "_trading_bot_timeout_injected", False):
        return

    original_request = session.request

    def request_with_timeout(method, url, **kwargs):
        kwargs.setdefault("timeout", timeout_seconds)
        return original_request(method, url, **kwargs)

    session.request = request_with_timeout
    session._trading_bot_timeout_injected = True


def get_account(trading_client: TradingClient):
    """Return the Alpaca account object, or None if the request fails."""
    try:
        return trading_client.get_account()
    except Exception as exc:
        print(f"Error getting account: {exc}")
        return None


def get_market_clock(trading_client: TradingClient):
    """Return the Alpaca market clock object, or None if the request fails."""
    try:
        return trading_client.get_clock()
    except Exception as exc:
        print(f"Error checking market clock: {exc}")
        return None


def market_is_open(trading_client: TradingClient) -> bool:
    """Check whether Alpaca reports the market as open right now."""
    clock = get_market_clock(trading_client)
    if clock is None:
        return False
    return bool(clock.is_open)


def get_position(trading_client: TradingClient, symbol: str):
    """Return the current open position for a symbol, if one exists."""
    try:
        return trading_client.get_open_position(symbol)
    except Exception:
        return None


def get_all_positions(trading_client: TradingClient) -> list:
    """Return all currently open Alpaca positions."""
    try:
        return list(trading_client.get_all_positions())
    except Exception as exc:
        print(f"Error getting all positions: {exc}")
        return []


def get_asset(trading_client: TradingClient, symbol: str):
    """Return the Alpaca asset metadata for a symbol, if available."""
    try:
        return trading_client.get_asset(symbol)
    except Exception as exc:
        print(f"Error getting asset {symbol}: {exc}")
        return None


def get_orders(trading_client: TradingClient, status: QueryOrderStatus = QueryOrderStatus.OPEN):
    """Return Alpaca orders for the requested status.

    The bot uses broker orders as operational truth for pending/working
    state instead of inferring that from local memory or prior logs.
    """
    request = GetOrdersRequest(status=status, nested=False)

    try:
        return list(trading_client.get_orders(filter=request))
    except TypeError:
        try:
            return list(trading_client.get_orders(request))
        except Exception as exc:
            print(f"Error getting orders: {exc}")
            return []
    except Exception as exc:
        print(f"Error getting orders: {exc}")
        return []


def get_working_orders(trading_client: TradingClient, symbol: str | None = None, orders: list | None = None) -> list:
    """Return non-final Alpaca orders, optionally filtered to one symbol."""
    if orders is None:
        orders = get_orders(trading_client, status=QueryOrderStatus.OPEN)
    if symbol is None:
        return [order for order in orders if not is_final_order_status(get_order_status(order))]
    return [
        order
        for order in orders
        if not is_final_order_status(get_order_status(order)) and _symbol_matches(order, symbol)
    ]


def get_stop_orders(trading_client: TradingClient, symbol: str, working_orders: list | None = None) -> list:
    """Return all open stop orders for the symbol."""
    return [
        order
        for order in get_working_orders(trading_client, symbol, orders=working_orders)
        if _serialize_enum(getattr(order, "order_type", getattr(order, "type", ""))) == "stop"
    ]


def get_protective_stop_order(
    trading_client: TradingClient,
    symbol: str,
    expected_side: str | None = None,
    stop_orders: list | None = None,
    working_orders: list | None = None,
):
    """Return the first matching open stop order for the symbol, if one exists."""
    for order in get_stop_orders(trading_client, symbol, working_orders=working_orders) if stop_orders is None else stop_orders:
        if expected_side is not None and get_order_side(order) != expected_side:
            continue
        return order
    return None


def build_market_order_request(symbol: str, side: OrderSide, qty: int) -> MarketOrderRequest:
    """Build a market order request.

    Keeping request construction separate makes it easier to add limit,
    stop, or bracket-style requests later without changing the rest of
    the trading flow.
    """
    return MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY,
    )


def build_stop_order_request(symbol: str, side: OrderSide, qty: int, stop_price: float) -> StopOrderRequest:
    """Build a plain stop order request.

    This keeps market and stop orders following the same small pattern,
    which makes future limit or bracket support easier to add.
    """
    return StopOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        stop_price=stop_price,
        time_in_force=TimeInForce.DAY,
    )


def submit_order(trading_client: TradingClient, order_request):
    """Submit a prepared Alpaca order request."""
    return trading_client.submit_order(order_data=order_request)


def submit_market_order(
    trading_client: TradingClient,
    symbol: str,
    side: OrderSide,
    qty: int,
):
    """Submit a simple day market order through Alpaca."""
    return submit_order(trading_client, build_market_order_request(symbol, side, qty))


def submit_stop_order(
    trading_client: TradingClient,
    symbol: str,
    side: OrderSide,
    qty: int,
    stop_price: float,
):
    """Submit a simple stop order through Alpaca."""
    return submit_order(trading_client, build_stop_order_request(symbol, side, qty, stop_price))


def get_order_by_id(trading_client: TradingClient, order_id: str):
    """Fetch the latest order state from Alpaca by order id."""
    try:
        return trading_client.get_order_by_id(order_id)
    except TypeError:
        # Some alpaca-py versions also accept an optional request object.
        try:
            return trading_client.get_order_by_id(order_id, GetOrderByIdRequest(nested=False))
        except Exception as exc:
            print(f"Error getting order {order_id}: {exc}")
            return None
    except Exception as exc:
        print(f"Error getting order {order_id}: {exc}")
        return None


def cancel_order_by_id(trading_client: TradingClient, order_id: str) -> bool:
    """Cancel an existing Alpaca order by id."""
    try:
        trading_client.cancel_order_by_id(order_id)
        return True
    except Exception as exc:
        print(f"Error canceling order {order_id}: {exc}")
        return False


def get_order_status(order) -> str:
    """Return the lower-case order status string."""
    status = getattr(order, "status", "")
    if hasattr(status, "value"):
        status = status.value
    return str(status).lower()


def get_position_market_value(position) -> float:
    """Return the absolute market value for an Alpaca position."""
    market_value = getattr(position, "market_value", None)
    if market_value not in {None, ""}:
        return abs(_safe_float(market_value))

    qty = abs(_safe_float(getattr(position, "qty", 0)))
    entry_price = _safe_float(getattr(position, "avg_entry_price", 0))
    return abs(qty * entry_price)


def get_asset_flags(asset) -> dict:
    """Extract common tradability and shortability flags from an Alpaca asset."""
    if asset is None:
        return {
            "tradable": False,
            "shortable": False,
            "easy_to_borrow": False,
            "marginable": False,
            "fractionable": False,
        }

    return {
        "tradable": bool(getattr(asset, "tradable", False)),
        "shortable": bool(getattr(asset, "shortable", False)),
        "easy_to_borrow": bool(getattr(asset, "easy_to_borrow", False)),
        "marginable": bool(getattr(asset, "marginable", False)),
        "fractionable": bool(getattr(asset, "fractionable", False)),
    }


def is_final_order_status(status: str) -> bool:
    """Return True when Alpaca is unlikely to send more updates for the order."""
    return status in FINAL_ORDER_STATUSES


def wait_for_order_final_state(
    trading_client: TradingClient,
    order_id: str,
    poll_seconds: int,
    timeout_seconds: int,
):
    """Poll Alpaca until the order reaches a final state or the timeout expires.

    This is intentionally simple polling. A streaming approach would be
    faster and more responsive, but polling is easier to understand for
    this project and works fine for paper trading.
    """
    deadline = time.monotonic() + timeout_seconds
    order = get_order_by_id(trading_client, order_id)

    # Partial fills are treated as still in progress, so the bot keeps
    # waiting until the order either fully finishes or the timeout ends.
    while order is not None and not is_final_order_status(get_order_status(order)) and time.monotonic() < deadline:
        time.sleep(poll_seconds)
        order = get_order_by_id(trading_client, order_id)

    return order
