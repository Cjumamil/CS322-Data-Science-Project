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
* Current built-in strategies: SMA crossover and EMA-band MACD pullback
* Paper trading mode (no real money used)
* Trade logging with linked decision/trade IDs
* Flexible strategy context logging in JSONL
* Broker-state-aware order/position handling
* Direction-aware paper trading for flat, long, and short positions
* Basic risk management (broker-side stop-loss and loop-based take-profit)
* Offline backtests with per-run artifacts and per-trade zoom charts

---

### Project Structure

```text
trading_bot/
|-- main.py               # Main execution loop across configured symbols
|-- backtest.py           # Offline historical backtest CLI
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
    |-- sma_crossover.py  # SMA crossover strategy
    `-- macd_pullback.py  # EMA-band + MACD pullback continuation strategy

bot_config.toml           # Bot/runtime/risk settings + per-symbol strategy assignments
shared_backtests/         # Team-shared backtest runs intended for Git commits
```

---

### Setup Instructions

#### 1. Install dependencies

```bash
pip install alpaca-py python-dotenv pandas matplotlib
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
* `ALB` with `macd_pullback` and a wider stop-distance filter
* `MSFT` with `macd_pullback` on `5m` bars

The current `macd_pullback` implementation is built around:

* `EMA12`, `EMA26`, and `EMA200` high/close/low bands
* trend, pullback, and re-entry concept columns
* EMA-band-based stop placement with a `1.5R` target
* optional strategy exits that can be toggled on later for experiments

Recent NVDA backtest sweeps suggest a better exploratory baseline than the original `EMA200` setup: `ema_band_window=72`, `require_pullback_breakout=true`, and `pullback_breakout_lookback=3`. This is currently treated as a research baseline for further chart review and exit tuning rather than a finalized production setting.

#### 4. Run the bot

From the project root directory:

```bash
python -m trading_bot.main
```

#### 5. Run an offline backtest

Use the separate backtest entry point for historical analysis:

```bash
python -m trading_bot.backtest --symbol MSFT --strategy macd_pullback --start 2025-01-01 --end 2025-03-31
```

This saves artifacts under local `backtests/` by default, with each run stored in its own timestamped folder containing a trade CSV, summary JSON, a broad overview chart, and per-trade zoom charts.

You can also override individual strategy settings from the CLI without editing `bot_config.toml`:

```bash
python -m trading_bot.backtest --symbol NVDA --strategy macd_pullback --start 2025-03-31 --end 2026-03-31 --strategy-param ema_band_window=100 --strategy-param require_pullback_breakout=true
```

For one-parameter experiments, use the built-in sweep mode. It runs one batch of backtests with the same fixed settings, varies the chosen parameter across the provided values, saves the normal artifacts for each variant, and writes a comparison CSV plus JSON summary in one sweep folder:

```bash
python -m trading_bot.backtest --symbol NVDA --strategy macd_pullback --start 2025-03-31 --end 2026-03-31 --sweep-param ema_band_window --sweep-values 50,72,100,200
```

One useful follow-up pattern is to lock a promising baseline with `--strategy-param` and then sweep one additional filter:

```bash
python -m trading_bot.backtest --symbol NVDA --strategy macd_pullback --start 2025-03-31 --end 2026-03-31 --strategy-param ema_band_window=72 --strategy-param require_pullback_breakout=true --sweep-param pullback_breakout_lookback --sweep-values 3,5,8,10
```

You can combine fixed overrides with a sweep as long as the swept parameter itself is not also overridden:

```bash
python -m trading_bot.backtest --symbol NVDA --strategy macd_pullback --start 2025-03-31 --end 2026-03-31 --strategy-param require_pullback_breakout=true --sweep-param ema_band_window --sweep-values 50,72,100,200
```

To save a run under the team-shared folder intended for Git commits, add `--shared`:

```bash
python -m trading_bot.backtest --symbol MSFT --strategy macd_pullback --start 2025-01-01 --end 2025-03-31 --shared
```

Example run folder name:

* `backtests/2026-04-02_00-40-22_ALB_macd_pullback_2025-03-31_to_2026-03-31/`
* `shared_backtests/2026-04-02_00-40-22_ALB_macd_pullback_2025-03-31_to_2026-03-31/`
* `backtests/2026-04-06_22-10-00_NVDA_macd_pullback_2025-03-31_to_2026-03-31_sweep_ema_band_window/`

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
* `backtests/<timestamped-run-folder>/` - local offline backtest trade logs, summaries, and charts for one run
* `shared_backtests/<timestamped-run-folder>/` - team-shared offline backtest runs intended for Git commits

Each run can also contain a `trade_charts/` folder with one zoomed candlestick chart per trade, including:

* trade ID and symbol in the chart title
* signed return and exit reason in the filename
* `EMA12`, `EMA26`, and `EMA200` high/close/low overlays
* a MACD panel under the price chart

`decision_id` links the decision log, trade log, and strategy context entries together.

---

### Limitations

* Order tracking is polling-based (not real-time streaming)
* Take-profit logic is loop-based, not broker-side
* Symbols are processed sequentially, not concurrently
* Broker-side stop placement for short positions depends on Alpaca accepting buy stop orders on the symbol in paper trading

---

### Disclaimer

This project is for educational purposes only and does not constitute financial advice.
