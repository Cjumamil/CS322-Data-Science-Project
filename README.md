# Data Science 322: Final Project

## Algorithmic Trading Bot with Alpaca

### Overview

This project implements an automated trading bot using the Alpaca API for paper trading. The bot analyzes intraday market data and executes trades based on configurable account, symbol, and strategy assignments.

The system is designed to simulate real-world trading behavior, including:

* Data retrieval from Alpaca
* Signal generation using pluggable strategies
* Order execution via Alpaca
* Trade logging and risk management

---

### Features

* Uses **Alpaca API** for both data and execution
* Config-driven setup via `bot_config.toml`
* Multi-account, multi-symbol sequential evaluation in one run
* Pluggable strategy architecture
* Current built-in strategies: SMA crossover, EMA-band MACD pullback, VWAP + RSI mean reversion, and an experimental XGBoost-filtered MACD pullback variant
* Paper trading mode (no real money used)
* Startup preflight mode for account / symbol readiness checks
* Account-aware trade logging with linked decision/trade IDs
* Flexible strategy context logging in JSONL
* Broker-state-aware order/position handling
* Direction-aware paper trading for flat, long, and short positions
* Basic risk management (broker-side stop-loss and loop-based take-profit)
* Entry safeguards for shortability, optional easy-to-borrow enforcement, and portfolio exposure caps
* Lightweight persisted live session state for cleaner restarts
* Offline backtests with per-run artifacts and per-trade zoom charts

---

### Project Structure

```text
trading_bot/
|-- main.py               # Main execution loop across configured accounts and symbols
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
    |-- macd_pullback.py  # EMA-band + MACD pullback continuation strategy
    |-- vwap_rsi_mean_reversion.py  # VWAP + RSI intraday mean-reversion strategy
    `-- macd_pullback_xgboost.py  # Experimental XGBoost-gated MACD pullback variant
|-- train_xgboost_filter.py  # Pooled XGBoost training / evaluation workflow
`-- xgboost_filter.py    # Shared XGBoost feature engineering and inference helpers

bot_config.toml           # Bot/runtime/risk settings + per-account strategy assignments
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

# Optional second paper account for live strategy comparison
PAPER2_ALPACA_API_KEY=your_second_api_key_here
PAPER2_ALPACA_SECRET_KEY=your_second_secret_key_here
```

#### 3. Review `bot_config.toml`

This file controls:

* global bot settings
* execution settings
* risk settings
* which Alpaca paper accounts to use
* which symbols each account trades
* which strategy each account/symbol pair uses

Per-account symbol strategy blocks inherit built-in defaults for the selected strategy, so you can keep most symbols minimal and only add overrides when needed. The current `vwap_rsi_paper` blocks intentionally pin a couple of researched overrides rather than relying entirely on live code defaults.

The current active config uses two separate paper-account blocks:

* `macd_pullback_paper` reads `ALPACA_API_KEY` / `ALPACA_SECRET_KEY` and runs `macd_pullback`
* `vwap_rsi_paper` reads `PAPER2_ALPACA_API_KEY` / `PAPER2_ALPACA_SECRET_KEY` and runs `vwap_rsi_mean_reversion`

Both account blocks currently trade:

* `AAPL`
* `AMD`
* `AMZN`
* `META`
* `SPY`
* `QQQ`
* `IWM`
* `ARKK`
* `MSFT`
* `NVDA`

The account name is written into the decision and trade logs so the same ticker can be compared across strategies without mixing results.

Each account can also set `state_namespace` in `bot_config.toml`. The MACD account currently leaves it blank to keep using the legacy bare-symbol session-state keys, while the VWAP account uses a namespace so it can trade the same tickers independently.

The current `vwap_rsi_paper` live-comparison setup explicitly pins:

* `min_distance_from_vwap_frac = 0.003`
* `max_bars_in_trade = 15`

This was chosen to match the best saved first-pass NVDA VWAP/RSI sweep variant by total return, while keeping the rest of the strategy on the current shared defaults.

The current `macd_pullback` implementation is built around:

* `EMA12`, `EMA26`, and `EMA200` high/close/low bands
* trend, pullback, and re-entry concept columns
* EMA-band-based stop placement with a `1.5R` target
* optional strategy exits that can be toggled on later for experiments

The current `macd_pullback` baseline uses the promoted shared defaults around `ema_band_window=72`, `require_pullback_breakout=true`, `pullback_breakout_lookback=3`, and the newer slope-based MACD failure exit settings.

There is also a new `vwap_rsi_mean_reversion` strategy for research and live paper-account comparison work. It uses intraday VWAP as the mean, RSI as the stretch signal, a light stabilization/reversal confirmation before entry, a recent-extreme stop model, and a configurable VWAP-targeted take-profit. As of April 24, 2026, it runs in the separate `vwap_rsi_paper` account while `macd_pullback` remains the baseline in `macd_pullback_paper`.

There is also an experimental `macd_pullback_xgboost_filter` strategy path plus a pooled XGBoost trainer. As of April 21, 2026, the pooled filter had been implemented and backtested, but it was not adopted for the April 22, 2026 paper-trading session because the final out-of-sample holdout still underperformed plain `macd_pullback` on profit factor and average return.

#### 4. Run a preflight check

Before a live paper-trading session, you can set `preflight_only = true` in `bot_config.toml` and run:

```bash
python -m trading_bot.main
```

This prints account state, market clock, symbol tradability / shortability flags, open positions, protective-stop status, and working orders, then exits without trading.

#### 5. Run the bot

From the project root directory:

```bash
python -m trading_bot.main
```

#### 6. Run an offline backtest

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

The same workflow can be used for the mean-reversion strategy. For example:

```bash
python -m trading_bot.backtest --symbol NVDA --strategy vwap_rsi_mean_reversion --start 2025-04-21 --end 2026-04-21 --sweep-param max_bars_in_trade --sweep-values 6,9,12,15,18
```

To save a run under the team-shared folder intended for Git commits, add `--shared`:

```bash
python -m trading_bot.backtest --symbol MSFT --strategy macd_pullback --start 2025-01-01 --end 2025-03-31 --shared
```

For the pooled XGBoost research workflow, use the separate training entry point. It trains one shared artifact across the active symbol set, selects a threshold on a validation window, and writes a report plus per-symbol comparison CSV:

```bash
python -m trading_bot.train_xgboost_filter --train-start 2023-01-03 --train-end 2024-12-31 --validation-start 2025-01-01 --validation-end 2025-09-30 --test-start 2025-10-01 --test-end 2026-04-21
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
* The bot evaluates configured accounts and symbols sequentially each polling cycle
* Logs are observational only; Alpaca broker/account state is treated as the source of truth
* A lightweight `live_session_state.json` file is used to persist active trade context across restarts

---

### Logs

The bot writes five main logs in the project root:

* `decision_log.csv` - full stable decision log, including frequent `HOLD` / `no_trade` rows
* `notable_decision_log.csv` - compact decision log that records everything except `HOLD` with reason `no_trade`
* `trade_log.csv` - execution/fill log with order lifecycle details, including the account label
* `strategy_context_log.jsonl` - full flexible strategy-specific context for each decision, including account metadata
* `notable_strategy_context_log.jsonl` - compact strategy-context log that records everything except `HOLD` with reason `no_trade`

The current `bot_config.toml` also exposes a `[logging]` section so the full logs can stay enabled during development, then be disabled later without code changes if routine polling rows start creating too much noise or file growth.
* `live_session_state.json` - persisted lightweight live trade context used to make restarts safer
* `backtests/<timestamped-run-folder>/` - local offline backtest trade logs, summaries, and charts for one run
* `shared_backtests/<timestamped-run-folder>/` - team-shared offline backtest runs intended for Git commits

Each run can also contain a `trade_charts/` folder with one zoomed candlestick chart per trade, including:

* trade ID and symbol in the chart title
* signed return and exit reason in the filename
* strategy-aware overlays such as `EMA12`, `EMA26`, `EMA200` high/close/low bands, or `VWAP`
* a strategy-aware indicator panel such as MACD or RSI under the price chart when available

`decision_id` links the decision log, trade log, and strategy context entries together.

Historical single-account log rows were backfilled to `macd_pullback_paper` when the repo moved to the separate-account layout, so older records remain analyzable without ambiguous blank account labels.

---

### Limitations

* Order tracking is polling-based (not real-time streaming)
* Take-profit logic is loop-based, not broker-side
* Symbols are processed sequentially, not concurrently
* Broker-side stop placement and short entries still depend on Alpaca accepting the order on the symbol in paper trading

---

### Disclaimer

This project is for educational purposes only and does not constitute financial advice.
