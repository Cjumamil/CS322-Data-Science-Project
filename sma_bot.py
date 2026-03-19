import os
import math
import pandas as pd
import yfinance as yf

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce


API_KEY = os.getenv("ALPACA_API_KEY")
SECRET_KEY = os.getenv("ALPACA_SECRET_KEY")


# ============================
# SETTINGS
# ============================
SYMBOL = "AAPL"
FAST_WINDOW = 20
SLOW_WINDOW = 50
PERIOD = "6mo"
INTERVAL = "1d"

# Risk / position controls
MAX_POSITION_QTY = 5
RISK_FRACTION_OF_BUYING_POWER = 0.02   # 2%
STOP_LOSS_PCT = 0.03                   # 3%
TAKE_PROFIT_PCT = 0.06                 # 6%

# Test mode
FORCE_TEST_TRADE = False
FORCE_DIRECTION = "SELL"


# ============================
# ALPACA HELPERS
# ============================
def connect_alpaca() -> TradingClient:
    if not API_KEY or not SECRET_KEY:
        raise ValueError(
            "Missing Alpaca credentials. Set ALPACA_API_KEY and ALPACA_SECRET_KEY as environment variables."
        )
    return TradingClient(API_KEY, SECRET_KEY, paper=True)


def get_account(trading_client: TradingClient):
    try:
        return trading_client.get_account()
    except Exception as e:
        print(f"Error getting account: {e}")
        return None


def market_is_open(trading_client: TradingClient) -> bool:
    try:
        clock = trading_client.get_clock()
        return bool(clock.is_open)
    except Exception as e:
        print(f"Error checking market clock: {e}")
        return False


def get_position(trading_client: TradingClient, symbol: str):
    try:
        return trading_client.get_open_position(symbol)
    except Exception:
        return None


def submit_market_order(
    trading_client: TradingClient,
    symbol: str,
    side: OrderSide,
    qty: int
):
    order_data = MarketOrderRequest(
        symbol=symbol,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.DAY
    )
    return trading_client.submit_order(order_data=order_data)


# ============================
# LOGGING
# ============================
def log_trade(action: str, symbol: str, qty: int, price: float, note: str = ""):
    row = pd.DataFrame([{
        "action": action,
        "symbol": symbol,
        "qty": qty,
        "price_reference": price,
        "note": note
    }])

    try:
        existing = pd.read_csv("trade_log.csv")
        updated = pd.concat([existing, row], ignore_index=True)
        updated.to_csv("trade_log.csv", index=False)
    except FileNotFoundError:
        row.to_csv("trade_log.csv", index=False)


# ============================
# DATA + STRATEGY
# ============================
def download_data(symbol: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(symbol, period=period, interval=interval, auto_adjust=False)

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.dropna().copy()
    return df


def build_strategy_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["SMA_FAST"] = df["Close"].rolling(FAST_WINDOW).mean()
    df["SMA_SLOW"] = df["Close"].rolling(SLOW_WINDOW).mean()

    df = df.dropna(subset=["SMA_FAST", "SMA_SLOW"]).copy()

    df["signal"] = 0
    df.loc[df["SMA_FAST"] > df["SMA_SLOW"], "signal"] = 1

    df["crossover"] = df["signal"].diff()

    df["daily_return"] = df["Close"].pct_change()
    df["strategy_return"] = df["signal"].shift(1) * df["daily_return"]

    # Simple MAE reference using naive next-day prediction
    df["predicted_close"] = df["Close"].shift(1)
    df["abs_error"] = (df["Close"] - df["predicted_close"]).abs()

    return df


# ============================
# BACKTEST METRICS
# ============================
def calculate_backtest_metrics(df: pd.DataFrame) -> dict:
    work = df.copy()

    if work.empty:
        return {}

    work["daily_return"] = work["daily_return"].fillna(0)
    work["strategy_return"] = work["strategy_return"].fillna(0)

    cumulative_strategy = (1 + work["strategy_return"]).cumprod()
    cumulative_buy_hold = (1 + work["daily_return"]).cumprod()

    strategy_return_pct = (cumulative_strategy.iloc[-1] - 1) * 100
    buy_hold_return_pct = (cumulative_buy_hold.iloc[-1] - 1) * 100

    strategy_std = work["strategy_return"].std()
    if strategy_std and strategy_std != 0:
        sharpe_ratio = (work["strategy_return"].mean() / strategy_std) * (252 ** 0.5)
    else:
        sharpe_ratio = 0.0

    running_max = cumulative_strategy.cummax()
    drawdown = (cumulative_strategy / running_max) - 1
    max_drawdown_pct = drawdown.min() * 100

    buy_signals = int((work["crossover"] == 1).sum())
    sell_signals = int((work["crossover"] == -1).sum())

    mae_close_prediction = work["abs_error"].dropna().mean()

    return {
        "strategy_return_pct": strategy_return_pct,
        "buy_hold_return_pct": buy_hold_return_pct,
        "sharpe_ratio": sharpe_ratio,
        "max_drawdown_pct": max_drawdown_pct,
        "buy_signals": buy_signals,
        "sell_signals": sell_signals,
        "mae_close_prediction": mae_close_prediction
    }


# ============================
# RISK CONTROLS
# ============================
def calculate_order_qty(price: float, buying_power: float) -> int:
    risk_budget = buying_power * RISK_FRACTION_OF_BUYING_POWER
    qty = math.floor(risk_budget / price)

    qty = max(1, qty)
    qty = min(qty, MAX_POSITION_QTY)

    return qty


def should_exit_position(current_price: float, entry_price: float):
    if current_price <= entry_price * (1 - STOP_LOSS_PCT):
        return "stop_loss"
    if current_price >= entry_price * (1 + TAKE_PROFIT_PCT):
        return "take_profit"
    return None


# ============================
# MAIN BOT
# ============================
def run():
    print("\nDownloading market data...")
    raw_df = download_data(SYMBOL, PERIOD, INTERVAL)
    df = build_strategy_dataframe(raw_df)

    if len(df) < 2:
        print("Not enough data to compute strategy.")
        return

    print("\nRecent strategy data:")
    print(df[["Close", "SMA_FAST", "SMA_SLOW", "signal", "crossover"]].tail(8))

    metrics = calculate_backtest_metrics(df)
    print("\nBacktest metrics:")
    for key, value in metrics.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")

    latest_close = float(df["Close"].iloc[-1])
    latest_cross = float(df["crossover"].iloc[-1])

    print(f"\nLatest close: {latest_close:.2f}")
    print(f"Latest crossover: {latest_cross}")

    try:
        trading_client = connect_alpaca()
    except ValueError as e:
        print(e)
        return

    account = get_account(trading_client)
    if account is None:
        print("Could not retrieve account. Exiting.")
        return

    buying_power = float(account.buying_power)

    print(f"Account status: {account.status}")
    print(f"Buying power: {buying_power:.2f}")

    position = get_position(trading_client, SYMBOL)
    in_position = position is not None
    print(f"Currently holding {SYMBOL}: {in_position}")

    # Force-test mode
    if FORCE_TEST_TRADE:
        print("\nFORCE_TEST_TRADE is ON")

        if not market_is_open(trading_client):
            print("Market is closed. Forced order not sent.")
            return

        if FORCE_DIRECTION.upper() == "BUY":
            if in_position:
                print("Already in a position. Force BUY skipped.")
                return

            qty = calculate_order_qty(latest_close, buying_power)
            response = submit_market_order(trading_client, SYMBOL, OrderSide.BUY, qty)
            print("FORCED BUY ORDER SUBMITTED:")
            print(response)
            log_trade("BUY", SYMBOL, qty, latest_close, "forced_test_trade")
            return

        if FORCE_DIRECTION.upper() == "SELL":
            if not in_position:
                print("No open position. Force SELL skipped.")
                return

            qty = int(float(position.qty))
            response = submit_market_order(trading_client, SYMBOL, OrderSide.SELL, qty)
            print("FORCED SELL ORDER SUBMITTED:")
            print(response)
            log_trade("SELL", SYMBOL, qty, latest_close, "forced_test_trade")
            return

        print("FORCE_DIRECTION must be BUY or SELL.")
        return

    # Regular live logic
    if not market_is_open(trading_client):
        print("Market is closed. No live order submitted.")
        return

    # Risk exit first
    if in_position:
        entry_price = float(position.avg_entry_price)
        qty = int(float(position.qty))

        exit_reason = should_exit_position(latest_close, entry_price)
        if exit_reason is not None:
            response = submit_market_order(trading_client, SYMBOL, OrderSide.SELL, qty)
            print(f"SELL ORDER SUBMITTED due to {exit_reason}:")
            print(response)
            log_trade("SELL", SYMBOL, qty, latest_close, exit_reason)
            return

    # SMA crossover logic
    if latest_cross == 1.0 and not in_position:
        qty = calculate_order_qty(latest_close, buying_power)
        response = submit_market_order(trading_client, SYMBOL, OrderSide.BUY, qty)
        print("BUY ORDER SUBMITTED:")
        print(response)
        log_trade("BUY", SYMBOL, qty, latest_close, "sma_crossover")

    elif latest_cross == -1.0 and in_position:
        qty = int(float(position.qty))
        response = submit_market_order(trading_client, SYMBOL, OrderSide.SELL, qty)
        print("SELL ORDER SUBMITTED:")
        print(response)
        log_trade("SELL", SYMBOL, qty, latest_close, "sma_crossover")

    else:
        print("No trade today.")


if __name__ == "__main__":
    run()