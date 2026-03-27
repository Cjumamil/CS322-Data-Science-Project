# Data Science 322: Final Project

## Algorithmic Trading Bot with Alpaca

### Overview

This project implements an automated trading bot using the Alpaca API for paper trading. The bot analyzes intraday market data and executes trades based on a moving average crossover strategy.

The system is designed to simulate real-world trading behavior, including:

* Data retrieval from Alpaca
* Signal generation using technical indicators
* Order execution via Alpaca
* Trade logging and risk management

---

### Features

* Uses **Alpaca API** for both data and execution
* Intraday trading (5-minute intervals)
* Moving Average Crossover strategy (SMA-20 / SMA-50)
* Paper trading mode (no real money used)
* Trade logging with order status tracking
* Basic risk management (stop-loss)

---

### Project Structure

```
trading_bot/
│
├── main.py          # Main execution loop
├── strategy.py      # Trading strategy logic
├── data.py          # Market data retrieval (Alpaca)
├── broker.py        # Order execution and account handling
├── risk.py          # Risk management logic
├── logging_utils.py # Logging and trade tracking
```

---

### Setup Instructions

#### 1. Install dependencies

```bash
pip install alpaca-py python-dotenv pandas
```

#### 2. Create a `.env` file in the project root

```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

#### 3. Run the bot

From the project root directory:

```bash
python trading_bot/main.py
```

---

### Notes

* The bot runs in **paper trading mode** using Alpaca
* No real trades are executed
* Market must be open for trades to occur

---

### Limitations

* Order tracking is polling-based (not real-time streaming)
* Take-profit logic is loop-based, not broker-side
* Strategy is simplified for educational purposes

---

### Disclaimer

This project is for educational purposes only and does not constitute financial advice.
