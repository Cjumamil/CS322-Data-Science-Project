# Data Science 322: Final Project

## Algorithmic Trading Bot with Alpaca

### Overview

This project implements an automated trading bot using the Alpaca API for paper trading. The bot analyzes intraday market data and executes trades based on configurable strategy assignments.

The system is designed to simulate real-world trading behavior, including:

* Data retrieval from Alpaca
* Signal generation using pluggable strategies
* Order execution via Alpaca
* Trade logging and risk management

---

### Features

* Uses **Alpaca API** for both data and execution
* Config-driven setup via `bot_config.toml`
* Multi-symbol sequential evaluation in one run
* Pluggable strategy architecture
* Current built-in strategy: SMA crossover
* Paper trading mode (no real money used)
* Trade logging with linked decision/trade IDs
* Flexible strategy context logging in JSONL
* Broker-state-aware order/position handling
* Basic risk management (broker-side stop-loss and loop-based take-profit)

---

### Project Structure

```text
trading_bot/
|-- main.py               # Main execution loop across configured symbols
|-- config.py             # TOML config loading
|-- data.py               # Market data retrieval (Alpaca)
|-- broker.py             # Order execution and account handling
|-- state.py              # Live broker state snapshots and session helpers
|-- execution.py          # Order lifecycle and execution flow
|-- risk.py               # Risk management logic
|-- logging_utils.py      # CSV / JSONL log writers
|-- decision_logging.py   # Decision and strategy-context log payload builders
`-- strategies/
    |-- base.py           # Shared strategy interface/helpers
    `-- sma_crossover.py  # Current SMA crossover strategy

bot_config.toml           # Bot/runtime/risk settings + per-symbol strategy assignments
```

---

### Setup Instructions

#### 1. Install dependencies

```bash
pip install alpaca-py python-dotenv pandas
```

#### 2. Create a `.env` file in the project root

```text
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
```

#### 3. Review `bot_config.toml`

This file controls:

* global bot settings
* execution settings
* risk settings
* which symbols to trade
* which strategy each symbol uses

The current sample config includes:

* `AAPL` with `sma_crossover`
* `ALB` with `sma_crossover`

#### 4. Run the bot

From the project root directory:

```bash
python -m trading_bot.main
```

---

### Notes

* The bot runs in **paper trading mode** using Alpaca
* No real trades are executed
* Market must be open for trades to occur
* The bot evaluates configured symbols sequentially each polling cycle
* Logs are observational only; Alpaca broker/account state is treated as the source of truth

---

### Logs

The bot writes three main logs in the project root:

* `decision_log.csv` - stable, spreadsheet-friendly record of bot decisions
* `trade_log.csv` - execution/fill log with order lifecycle details
* `strategy_context_log.jsonl` - flexible strategy-specific context for each decision

`decision_id` links the decision log, trade log, and strategy context entries together.

---

### Limitations

* Order tracking is polling-based (not real-time streaming)
* Take-profit logic is loop-based, not broker-side
* Symbols are processed sequentially, not concurrently
* Strategy support is still early-stage and currently includes one built-in strategy

---

### Disclaimer

This project is for educational purposes only and does not constitute financial advice.
