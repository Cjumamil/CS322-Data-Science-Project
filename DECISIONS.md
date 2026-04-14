# Project Decision Log

Purpose: Track major architecture/technical decisions so the team stays aligned and future teams understand *why* choices were made.

Only log decisions that affect architecture, data flow, interfaces, modeling strategy, or assumptions.

Do not worry too much about writing a perfect log entry, staying consistent in writing log entries will be more helpful.
Focus on capturing decision and intent.

---

## Decision Template

## D-###  
**Date:** YYYY-MM-DD  
**Topic:**  
**Decision:**  
**Reasoning:**  
**Approved by:**  

## Example Template

## D-000  
**Date:** 2026-02-26  
**Topic:** Tracking method  
**Decision:** Use IoU-based tracker for baseline.  
**Reasoning:** Lightweight implementation, minimal dependencies, sufficient for prototype. More advanced tracking (e.g., DeepSORT) can be explored later if needed.  
**Approved by:** Tom, Alyssa  

---

## D-001
**Date:** 2026-02-26  
**Topic:** Decision log location  
**Decision:** Maintain decision log in `DECISIONS.md` in repo root.  
**Reasoning:** High visibility + persists for future teams.  
**Approved by:** Team consensus  

## D-002
**Date:** 2026-03-16  
**Topic:** Created Project Roadmap  
**Decision:** Designed a project roadmap chart with various phases.  
**Reasoning:** To have an organized overview of the project.  
**Approved by:** Joshua  

## D-003
**Date:** 2026-03-17  
**Topic:** Role Assignment  
**Decision:** Assigning roles to each member for various parts of the project.  
**Reasoning:** Roles were assigned to improve efficiency, ensure clear responsibility, and support effective teamwork across different parts of the project.  
**Approved by:** Team consensus

## D-004
**Date:** 2026-03-18  
**Topic:** Task Board  
**Decision:** Created a task board in a google document that lists each other's roles and a checklist of tasks to get done  
**Reasoning:** Keep track of relevant project tasks, updates, and info.  
**Approved by:** Joshua  

## D-005
**Date:** 2026-03-24  
**Topic:** Trading Implementation & Test Run  
**Decision:** Implement and test that the code works with the Alpaca API and test it without using our current paper trade balance.  
**Reasoning:** While we haven't completely figured out our trading bot algorithm yet, we want to at least be able to get the code working and connecting with the API so that once we do figure out the algorithm, we will be mostly set up.  
**Approved by:** Joshua, Havanna  

## D-006
**Date:** 2026-03-24  
**Topic:** Initial Trading Philosophy  
**Decision:** Trade less often but enter setups with higher expected value signals in which we will risk more. Combine this strategy with lower expected value signal setups.  
**Reasoning:** This idea of this philosophy is to allow the bot to take advantage of scenarios where there is greater potential for gain, while also combine it with more often lower risk, lower value setups to ensure we have consistent effort by the bot. Our goal is not just for the bot to have high accuracy in its decision-making, but to make the most amount of money possible.  
**Approved by:** Joshua, Havanna  

## D-007  
**Date:** 2026-03-27  
**Topic:** Codebase Refactor (Modular Design)  
**Decision:** Refactored the original `sma_bot.py` into multiple modules (`main.py`, `strategy.py`, `data.py`, `broker.py`, `risk.py`, `logging_utils.py`) to separate responsibilities.  
**Reasoning:** The original implementation had all functionality in a single file, making it harder to read, debug, and extend. Splitting the code into focused modules improves readability, supports modular development, and makes it easier to modify or replace individual components (e.g., strategy or broker logic) without affecting the entire system. This also aligns better with real-world software design practices.  
**Approved by:** Joshua  

## D-008  
**Date:** 2026-03-27  
**Topic:** Data Source and Execution Integration  
**Decision:** Switched from yfinance to Alpaca API for both market data and trade execution.  
**Reasoning:** Using separate sources for data (yfinance) and execution (Alpaca) could introduce inconsistencies in pricing and timing. Moving to Alpaca for both ensures data and execution are aligned, improves realism, and simplifies system design. This also allows the bot to better reflect real-world trading conditions.  
**Approved by:** Joshua  

## D-009  
**Date:** 2026-03-27  
**Topic:** Execution Model (Intraday Loop)  
**Decision:** Transitioned from a daily-style execution model to a continuous intraday loop using fixed intervals (e.g., 5-minute bars).  
**Reasoning:** Day trading requires frequent evaluation of market conditions. A continuous loop allows the bot to process new data and make decisions throughout the trading session. Separating interval (data granularity) from polling frequency also improves flexibility and control over execution timing.  
**Approved by:** Joshua  

## D-010  
**Date:** 2026-03-27  
**Topic:** Order Tracking and Trade Logging  
**Decision:** Updated trade logging to track actual order states (submitted, filled, timestamps, quantities) instead of assuming immediate execution.  
**Reasoning:** Previous logging assumed that order submission equaled execution, which is not realistic. The updated approach improves accuracy by distinguishing between decision, submission, and fill events. This provides better visibility into system behavior and aligns with real trading workflows.  
**Approved by:** Joshua  

## D-011  
**Date:** 2026-03-27  
**Topic:** Risk Management (Stop-Loss Implementation)  
**Decision:** Implemented broker-side stop-loss orders placed after entry fills.  
**Reasoning:** Previously, exit conditions were only evaluated within the bot's polling loop, which could lead to delayed reactions. Adding a broker-side stop-loss ensures downside protection even if the bot is delayed or not actively checking conditions. This improves reliability and reduces risk exposure.  
**Approved by:** Joshua  

## D-012  
**Date:** 2026-03-27  
**Topic:** API Key Security  
**Decision:** Moved Alpaca API credentials out of source code and into environment variables using a `.env` file with `python-dotenv`.  
**Reasoning:** Storing API keys directly in code is insecure, especially in a public repository. Using environment variables prevents accidental exposure and aligns with standard security practices. A `.gitignore` file was also configured to prevent `.env` from being committed.  
**Approved by:** Joshua  

## D-013  
**Date:** 2026-03-27  
**Topic:** Strategy Direction (Context-Aware Trading)  
**Decision:** Expand the trading strategy beyond isolated indicator signals to incorporate broader market context, including sector performance and overall market conditions.  
**Reasoning:** Relying solely on technical indicators (e.g., SMA crossovers) can lead to weak or misleading signals when market conditions are unfavorable. Real traders often consider additional context such as whether the broader market is trending (e.g., S&P 500, Dow Jones) and how a stock performs relative to its sector or related ETFs. Incorporating this context can improve decision quality by filtering trades and aligning positions with stronger market trends. This approach aims to make the bot's behavior more realistic and closer to actual trading practices, rather than purely indicator-driven.  
**Approved by:** Joshua  

## D-014  
**Date:** 2026-03-27  
**Topic:** Broker/API as Source of Truth  
**Decision:** Refactored the bot so that live Alpaca broker/account state is treated as the operational source of truth for positions, open orders, protective stops, and pending order state. Logs remain observational only and are no longer used as operational inputs.  
**Reasoning:** Local session memory and prior logs can become stale after fills, cancellations, delays, or restarts. Using Alpaca as the source of truth makes the bot more reliable and reduces the chance of duplicate orders or incorrect assumptions about live positions and open orders. This also better reflects how real trading systems should treat broker state versus internal memory.  
**Approved by:** Joshua  

## D-015  
**Date:** 2026-03-27  
**Topic:** Logging Architecture Split  
**Decision:** Keep `decision_log.csv` and `trade_log.csv` as stable, spreadsheet-friendly logs, and add `strategy_context_log.jsonl` as a flexible strategy-specific context log. Add a shared `decision_id` to link related log entries across the decision and trade lifecycle.  
**Reasoning:** Core logs should stay readable and stable over time, while strategy-specific data will likely change as new strategies are added. JSONL provides a flexible structure for evolving strategy context without forcing rigid CSV schemas. A shared `decision_id` improves traceability by linking what the bot saw, what decision it made, and what trade or fill followed.  
**Approved by:** Joshua  

## D-016  
**Date:** 2026-03-28  
**Topic:** Decision Log Schema Simplification  
**Decision:** Remove SMA-specific columns from `decision_log.csv` and move strategy-specific signal details into `strategy_context_log.jsonl`.  
**Reasoning:** The decision log should remain generic and durable even as the bot changes strategies. Keeping SMA-only columns in the main decision log would make the schema less reusable for future RSI, multi-factor, or combined strategies. The strategy context log is a better location for indicator-specific values such as SMA fast/slow, signal, and crossover.  
**Approved by:** Joshua  

## D-017  
**Date:** 2026-03-28  
**Topic:** Runtime Refactor for Readability  
**Decision:** Refactored the trading bot into additional focused modules, separating live broker state helpers, execution/order lifecycle handling, and strategy-context logging from the main runtime loop.  
**Reasoning:** `main.py` had grown to include too many responsibilities, making it harder to understand the overall cycle flow. Splitting supporting logic into `state.py`, `execution.py`, and `decision_logging.py` makes the code easier to read, maintain, and extend while keeping the main loop focused on orchestration.  
**Approved by:** Joshua  

## D-018  
**Date:** 2026-03-28  
**Topic:** Strategy Plugin Architecture  
**Decision:** Convert the existing SMA crossover logic into a real pluggable strategy module under `trading_bot/strategies/`, with a small shared strategy interface and `SmaCrossoverStrategy` as the first implementation.  
**Reasoning:** The original design still treated SMA crossover as the implicit built-in strategy. Moving strategy behavior behind a shared interface allows the current SMA implementation to exist as one strategy among many, reducing coupling and making it much easier to add future strategies without rewriting the execution flow.  
**Approved by:** Joshua  

## D-019  
**Date:** 2026-03-28  
**Topic:** Config-Driven Bot Setup  
**Decision:** Introduced a single TOML configuration file (`bot_config.toml`) to define global bot settings, execution settings, risk settings, and per-symbol strategy assignments. The initial config includes both `AAPL` and `ALB` using the SMA crossover strategy.  
**Reasoning:** Hardcoding symbol, strategy, and runtime settings in Python made the bot harder to test and reconfigure. A config file makes the bot easier to use, easier to experiment with, and better prepared for running multiple symbols with different strategy assignments in the future. Starting with one config file keeps the system simple while still supporting growth later if the symbol list becomes large enough to justify splitting config files.  
**Approved by:** Joshua  

## D-020  
**Date:** 2026-03-28  
**Topic:** Bot Version Tracking  
**Decision:** Added a lightweight bot-level version number in `bot_config.toml` and propagated it into runtime output and logs. The current bot version is `0.3.0`.  
**Reasoning:** The bot's architecture, logging model, and configuration structure have changed significantly over time, which makes it useful to distinguish runs produced by different system versions. Recording `bot_version` alongside strategy version helps separate overall system changes from strategy-logic changes and makes archived logs easier to interpret later.  
**Approved by:** Joshua  

## D-021  
**Date:** 2026-03-28  
**Topic:** Legacy Log Archiving Convention  
**Decision:** Archive historical `decision_log.csv` and `trade_log.csv` files into dated subfolders under `archive/`, using a naming convention such as `YYYY-MM-DD-bot-legacy` or `YYYY-MM-DD-bot-X.Y.Z`, and include a short `notes.txt` file when helpful.  
**Reasoning:** As the bot's architecture and logging model evolve, old logs can become harder to interpret if they remain mixed with current runs. Moving prior logs into clearly named archive folders preserves the historical record, creates a clean boundary for new runs, and makes it easier to understand which bot generation produced a given set of files. A short note file adds context when a version boundary or architectural shift is not obvious from the archived CSVs alone.  
**Approved by:** Joshua  

## D-022  
**Date:** 2026-04-02  
**Topic:** MACD Pullback Strategy and Offline Research Workflow  
**Decision:** Add `macd_pullback` as a second pluggable strategy alongside `sma_crossover`, and create a separate offline backtesting workflow that can simulate historical runs independently from the live paper-trading bot.  
**Reasoning:** The project needed a more realistic trend-following strategy than the initial SMA crossover baseline, but adding it should not break the modular strategy architecture. Separating offline backtesting from live trading keeps research and experimentation clean, makes it easier to compare strategies and symbols, and avoids overloading the live runtime with historical-simulation concerns. This also supports saving artifacts such as trade CSVs, summary JSON files, and charts for later analysis.  
**Approved by:** Joshua  

## D-023  
**Date:** 2026-04-02  
**Topic:** Direction-Aware Trading Architecture  
**Decision:** Refactor the bot from a long-only mental model into a direction-aware model that explicitly supports flat, long, and short positions in both live paper trading and backtesting.  
**Reasoning:** The original execution flow assumed `BUY` meant open and `SELL` meant close, which was too limited for a strategy like `macd_pullback` that can generate both long and short signals. Moving to an explicit position-side model makes the architecture cleaner, improves correctness, and better matches real trading behavior. It also keeps Alpaca broker state as the operational source of truth while allowing strategies to express directional intent more naturally.  
**Approved by:** Joshua  

## D-024  
**Date:** 2026-04-02  
**Topic:** Structure-Based Risk Model for MACD Pullback  
**Decision:** Use an EMA200-based stop-loss with a small buffer and a fixed `1.5R` take-profit for `macd_pullback`, and reject entries whose stop distance is too large relative to price using a configurable `max_stop_distance_frac_of_price` filter.  
**Reasoning:** The original shared percent-based stop-loss and take-profit model did not fit the structure-based logic of the MACD pullback setup. Using EMA200 as the stop reference ties risk to the underlying trend structure, while a fixed reward-to-risk target keeps trade management simple and consistent. Adding a maximum stop-distance filter helps avoid low-quality trades where price has already moved too far from the EMA and the required risk becomes too large.  
**Approved by:** Joshua  

## D-025  
**Date:** 2026-04-02  
**Topic:** Strategy-Aware Trade Management for MACD Pullback  
**Decision:** Keep hard risk exits in the shared risk layer, but allow `macd_pullback` to add configurable strategy-aware exits and entry refinements, including time stops, delayed MACD momentum-failure exits, and histogram confirmation at entry.  
**Reasoning:** Some trade-management rules are generic across strategies, while others are specific to how a given strategy expresses momentum and follow-through. Keeping stop-loss and take-profit logic centralized preserves a clean shared risk model, while strategy-specific exits and entry filters allow `macd_pullback` to reduce stagnant trades and weaker crossovers without turning the shared risk layer into strategy-specific code. This hybrid approach improves modularity, readability, and extensibility for future strategies.  
**Approved by:** Joshua  

## D-026  
**Date:** 2026-04-06  
**Topic:** EMA-Band Redesign for MACD Pullback  
**Decision:** Re-center `macd_pullback` around an EMA-band continuation model using `EMA12`, `EMA26`, and `EMA200` high/close/low bands, with named concept columns for trend, pullback, and re-entry logic.  
**Reasoning:** The prior `macd_pullback` implementation had drifted into a mix of entry filters and exit rules that no longer matched the original continuation-style intent. Reframing the strategy around explicit EMA-band concepts makes the logic easier to understand, tune, and debug while keeping the strategy closer to its intended continuation-based behavior.  
**Approved by:** Joshua  

## D-027  
**Date:** 2026-04-06  
**Topic:** Backtest Review Workflow  
**Decision:** Expand the offline backtest workflow to generate per-trade zoom charts and separate local exploratory backtests from team-shared backtest artifacts.  
**Reasoning:** Reviewing one full-run chart was not enough to inspect why the bot entered or exited specific trades. Adding per-trade zoom charts makes strategy review more practical, while separating local backtests from shared backtests keeps the repository cleaner and allows the team to selectively commit only the runs worth sharing.  
**Approved by:** Joshua  

## D-028
**Date:** 2026-04-07  
**Topic:** Position Sizing Cap Behavior  
**Decision:** Treat `risk.max_position_qty` as an optional hard cap. Positive values enforce the cap, while `0` or lower disables it and lets position sizing be determined only by buying power and `risk_fraction_of_buying_power`.  
**Reasoning:** A fixed share cap made backtest results harder to interpret across symbols and price regimes because later high-price trades were artificially constrained in ways that did not reflect the intended risk model. Allowing the cap to be disabled keeps the setting flexible for experiments while preserving the option to re-enable a hard limit later.  
**Approved by:** Joshua  

## D-029
**Date:** 2026-04-07  
**Topic:** Repo-Local Backtest Environment and Chart Scaling  
**Decision:** Add a repo-local dependency bootstrap in `trading_bot/__init__.py` that prefers a bundled `.vendor_bundle` dependency folder and removes user-site packages from project entry points. Also make per-trade zoom-chart context scale with the configured candle interval and include the interval in each trade-chart title.  
**Reasoning:** Fresh chats and shell sessions were repeatedly failing due to inconsistent Python package resolution between Conda, user-site installs, and blocked activation hooks. Moving Alpaca-related dependencies behind a repo-local bootstrap makes the backtest entry points more reproducible. At the same time, interval-aware chart scaling improves backtest review when experimenting with `1m`, `5m`, or slower bars, since a fixed bar window was visually misleading across different intervals.  
**Approved by:** Joshua  

## D-030
**Date:** 2026-04-07  
**Topic:** NVDA MACD Pullback Research Baseline  
**Decision:** Use the current best-tested NVDA exploratory baseline of `ema_band_window = 72`, `require_pullback_breakout = true`, and `pullback_breakout_lookback = 3` for the next round of chart review and exit-focused research. Keep this as a working research baseline rather than a permanent final configuration until additional review is completed.  
**Reasoning:** A one-parameter backtest sweep on `ema_band_window` improved behavior noticeably over the prior `200`-bar EMA band, especially by avoiding the previously identified April 1 short failure and producing a stronger long interpretation instead. A follow-up sweep showed that adding breakout confirmation helped reduce low-quality entries, and a smaller breakout lookback of `3` outperformed stricter values like `5`, `8`, and `10`. This suggests the strategy benefits from light price-structure confirmation without waiting so long that the continuation move is already mostly spent.  
**Approved by:** Joshua  

## D-031
**Date:** 2026-04-08  
**Topic:** Info Dashboard, XGBoost, and Report Writeup  
**Decision:** Splitting tasks between team members, where Havanna works on Info Dashboard, whilst Cooper and Ulysses work on XGBoost and the Report Writeup.  
**Reasoning:** An Info Dashboard is useful for observing data collected from our backtest runs and paper trading. XGBoost is a useful next step, allowing us to optimize our strategies. And we're nearing the time to write the reports. Splitting these tasks between our group members allows me(Joshua) to focus on designing the core functions of the bot.  
**Approved by:** Joshua, Ulysses, Havanna, Cooper  

## D-032
**Date:** 2026-04-14  
**Topic:** MACD Failure Exit Redesign for `macd_pullback`  
**Decision:** Replace the prior MACD-failure exit logic (simple MACD/signal crossover) with a configurable momentum-deterioration model based on MACD slope strength plus optional confirmation filters. Add strategy parameters for `macd_exit_slope_lookback`, `macd_exit_min_slope_frac_of_price`, `macd_exit_requires_histogram_confirmation`, and `macd_exit_requires_ema12_confirmation`, while retaining `min_bars_before_macd_exit` as an arming delay.  
**Reasoning:** Trade-chart review showed many cases where waiting for a full MACD/signal crossover was too late, while overly sensitive momentum checks were too noisy. The slope-based model allows earlier reaction to clear momentum failure without forcing a one-size-fits-all crossover exit. Keeping the new behavior strategy-local (instead of moving it into shared risk utilities) preserves architecture clarity because MACD-specific logic should remain in MACD-based strategies.  
**Approved by:** Joshua  

## D-033
**Date:** 2026-04-14  
**Topic:** NVDA Baseline Finalization and Strategy Default Promotion  
**Decision:** Promote the tuned NVDA `macd_pullback` settings to the working baseline and align `macd_pullback` defaults to match, so new symbols start from the same improved configuration. The active baseline includes: `ema_band_window=72`, `require_pullback_breakout=true`, `pullback_breakout_lookback=3`, `max_stop_distance_frac_of_price=0.04`, `enable_time_stop=true`, `max_bars_in_trade=30`, `enable_macd_failure_exit=true`, `min_bars_before_macd_exit=10`, `macd_exit_slope_lookback=2`, `macd_exit_min_slope_frac_of_price=0.0008`, `macd_exit_requires_histogram_confirmation=true`, and `macd_exit_requires_ema12_confirmation=false`.  
**Reasoning:** Multi-year sweeps showed this setup balanced return, profit factor, and practical exit activity better than stricter or looser alternatives. In particular, very high MACD slope thresholds produced strong headline stats but almost no `macd_failure` exits, while this baseline kept exits meaningfully active without becoming trigger-happy. A cross-check on MSFT with matched settings also preferred `macd_exit_slope_lookback=2`, supporting generalization beyond NVDA. Additional sweeps confirmed that raising `max_stop_distance_frac_of_price` above `0.04` did not improve results.  
**Approved by:** Joshua  

## D-034
**Date:** 2026-04-14  
**Topic:** Paper-Trading Readiness Hardening  
**Decision:** Add a startup preflight report, per-symbol exception isolation, broker-side protective-stop reconciliation for inherited live positions, shortability/ETB entry checks, a portfolio exposure cap, and persisted lightweight live session state.  
**Reasoning:** The bot was close to paper-trading readiness, but the first supervised runs still needed stronger operational safety around restarts, missing broker stops, short eligibility, and multi-symbol failure handling. These additions reduce avoidable runtime surprises while keeping Alpaca as the operational source of truth.  
**Approved by:** Joshua  

## D-035
**Date:** 2026-04-14  
**Topic:** Default-Inherited Symbol Configuration  
**Decision:** Make per-symbol strategy configs inherit built-in strategy defaults, and keep inactive expansion symbols commented in `bot_config.toml` so they can be re-enabled quickly without rewriting full parameter blocks.  
**Reasoning:** The project is expected to scale to a larger symbol universe, and hand-defining every strategy parameter for each symbol would make the config hard to maintain. Default inheritance keeps the active config lightweight while still allowing symbol-specific overrides when needed.  
**Approved by:** Joshua  
