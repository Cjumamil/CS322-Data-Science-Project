# Current Strategy Brief

As of April 15, 2026, the project's working trading strategy is a tuned `macd_pullback` continuation strategy, not the older SMA crossover baseline.

This document is meant to help new team members understand what the bot is currently trying to do, why the team moved in this direction, and where useful strategy-refinement work can happen next.

## 1. One-sentence summary

The current idea is to trade continuation moves in already-trending stocks by waiting for a pullback toward an EMA trend band, then entering only when momentum starts re-expanding in the direction of the trend.

## 2. What the strategy is trying to capture

The strategy is not trying to predict every short-term move.

Instead, it is trying to capture this pattern:

1. A stock is already in a clear trend.
2. Price pulls back toward the trend band instead of chasing far away from it.
3. Momentum turns back in the trend direction.
4. The bot enters on that "pullback then continuation" structure.

In plain language:

- Long idea: strong uptrend -> temporary pullback -> bullish re-entry -> buy.
- Short idea: strong downtrend -> temporary bounce -> bearish re-entry -> short.

This fits the team's broader philosophy from March 24, 2026: fewer but higher-quality setups, with the goal of maximizing return rather than just maximizing prediction accuracy.

## 3. Why the team moved away from the original baseline

The original baseline was an SMA crossover strategy. That was useful as a first working prototype, but it was too simple for the kind of intraday continuation behavior the team wanted to trade.

The current strategy became the focus because it is:

- more aligned with actual trend-following / pullback trading logic,
- easier to reason about visually in charts,
- flexible enough to support both long and short trades,
- better suited to structured backtesting and parameter sweeps.

## 4. The current working baseline

The active default strategy baseline was promoted on April 14, 2026 and is now the default `macd_pullback` configuration used for new symbols.

Core baseline settings:

- Interval: `5m`
- EMA band window: `72`
- Require pullback breakout: `true`
- Pullback breakout lookback: `3`
- Max stop distance fraction of price: `0.04`
- Time stop enabled: `true`
- Max bars in trade: `30`
- MACD failure exit enabled: `true`
- Min bars before MACD exit: `10`
- MACD exit slope lookback: `2`
- MACD exit min slope fraction of price: `0.0008`
- MACD exit requires histogram confirmation: `true`
- MACD exit requires EMA12 confirmation: `false`

Important implementation note:

- In the code, several helper columns are still named like `EMA200_high`, `EMA200_close`, and `EMA200_low`.
- Those names are legacy labels from an older version of the strategy.
- The active baseline is not using a 200-bar EMA band anymore. The active `ema_band_window` is `72`.

## 5. How entries work

An entry only happens if several layers agree.

### Trend filter

The bot first decides whether the stock is in a usable long or short trend.

For longs, price must be persistently above the EMA band and the band slope must be positive enough.
For shorts, price must be persistently below the EMA band and the band slope must be negative enough.

This is meant to stop the bot from taking continuation trades in weak or sideways conditions.

### Pullback filter

The bot then looks for a real pullback into the band, not a completely random bar.

The pullback logic checks things like:

- whether there was a recent trend before the pullback,
- whether price actually touched the band recently,
- whether the pullback stayed shallow enough to still count as a pullback instead of a breakdown / reversal,
- whether there was prior impulse before the pullback.

### Re-entry / momentum filter

After the pullback, the bot wants evidence that momentum is expanding again in the original direction.

The current baseline requires:

- MACD context aligned with direction,
- histogram re-expansion,
- reclaim of EMA12 and EMA26 in the trade direction,
- breakout of the recent pullback structure over the last `3` bars.

That combination is the "do not enter too early" layer.

### Session filter

The bot only allows entries during the regular session and skips the first `6` bars of the day.

At a `5m` interval, that means it avoids the first 30 minutes after the open.

## 6. How exits and risk work

The strategy uses a hybrid exit model:

- Hard risk exits are still handled in a shared risk layer.
- Strategy-specific exits are handled inside `macd_pullback`.

Current trade management:

- Stop-loss is anchored to the EMA band with a small buffer.
- Take-profit is fixed at `1.5R`.
- Entries are rejected if the stop would be too far from the entry. The current cap is `4%` of price.
- A time stop exits stagnant trades after `30` bars.
- A MACD failure exit can close trades earlier if momentum clearly deteriorates.

The MACD failure exit is important because it was redesigned on April 14, 2026.

The current version does not wait for a simple MACD/signal crossover. Instead, it looks at slope-based deterioration in MACD momentum, then optionally confirms it with the histogram and other filters. This was added because the old version was often either too late or too noisy.

## 7. Why this baseline was chosen

The current baseline came from a sequence of backtests and parameter sweeps, especially on NVDA, then a cross-check on MSFT.

Main decisions from the research cycle:

- April 7, 2026: `ema_band_window=72` and `pullback_breakout_lookback=3` emerged as the better exploratory NVDA baseline than the earlier 200-style setup.
- April 14, 2026: the slope-based MACD failure exit replaced the earlier crossover-style failure exit.
- April 14, 2026: the tuned NVDA settings were promoted to the default baseline for `macd_pullback`.

Selected backtest evidence:

- NVDA sweep, March 31, 2021 to March 31, 2026, with the promoted baseline and `max_stop_distance_frac_of_price=0.04`:
  - `669` trades
  - win rate `46.49%`
  - profit factor `1.1162`
  - total return `0.6120%`
  - max drawdown `-0.4688%`
- In the same NVDA sweep, raising the stop-distance cap above `0.04` did not improve the result enough to justify loosening the filter.
- NVDA exit-confirmation sweep:
  - keeping histogram confirmation `true` performed slightly better than turning it off.
  - setting EMA12 confirmation to `false` performed better than keeping it `true`.
- MSFT cross-check on MACD exit slope lookback:
  - `macd_exit_slope_lookback=2` slightly outperformed `1` and `3` on profit factor and total return, which supported using `2` as the baseline instead of treating the NVDA result as purely symbol-specific.

These results should be treated as evidence for the current working baseline, not proof that the strategy is "finished."

## 8. What is active right now

The current `bot_config.toml` uses `macd_pullback` for:

- `AAPL`
- `AMD`
- `AMZN`
- `META`
- `MSFT`
- `NVDA`

Commented expansion symbols:

- `SPY`
- `QQQ`
- `IWM`

There is also a commented legacy `ALB` experiment kept for reference.

## 9. Operational safeguards around the strategy

The project is not only about signal logic anymore. By April 14, 2026, the bot also added paper-trading readiness hardening, including:

- startup preflight reporting,
- per-symbol exception isolation,
- broker-side protective-stop reconciliation,
- shortability / easy-to-borrow checks,
- total portfolio exposure cap,
- lightweight persisted live session state.

This matters because the current strategy should be evaluated as both:

- a signal-generation idea, and
- something that can realistically run in a supervised paper-trading loop.

## 10. The most useful open questions for new team members

If someone wants to help refine strategy without diving deeply into implementation first, these are strong questions to investigate:

1. Are the long and short sides equally healthy, or is one side carrying most of the edge?
2. Are we exiting too early, too late, or at the right time with the current MACD failure logic?
3. Does the current baseline generalize across symbols, or is it still too NVDA-shaped?
4. Should market-context filters be added back in more explicitly, such as ETF or sector confirmation?
5. Which losses are "acceptable pullback failures" versus "signals we should have filtered out"?
6. Is the `1.5R` target still the best shared exit target once the newer momentum exit is active?
7. Does the first-30-minutes no-trade rule remain correct across all symbols in the active watchlist?

## 11. Best way to read the project after this document

If you want to go one layer deeper, the best reading order is:

1. `CURRENT_STRATEGY_BRIEF.md`
2. `DECISIONS.md` entries `D-022` through `D-035`
3. `bot_config.toml`
4. `trading_bot/strategies/macd_pullback.py`
5. recent backtest sweep summaries under `backtests/2026-04-13_*` and `backtests/2026-04-14_*`

## 12. Bottom line

As of April 15, 2026, the project's current strategy is:

- an intraday `5m` EMA-band continuation strategy,
- built around trend -> pullback -> momentum re-entry,
- tuned primarily on NVDA and cross-checked on MSFT,
- using structure-based stops, a `1.5R` target, a time stop, and a slope-based MACD failure exit,
- considered the team's best working baseline, but still open to refinement and broader validation.
