# Research Log

Purpose: Track meaningful strategy-research findings so the team can preserve what was tested, what seemed to matter, and what conclusions were drawn from backtests or live observation.

Use this file for research conclusions, parameter-sweep takeaways, cross-symbol comparisons, and strategy-behavior observations that are useful for future experiments.

Do not worry too much about writing a perfect entry. Consistency is more valuable than polish.

This file is not meant to replace `DECISIONS.md`.

- `DECISIONS.md` is for architecture, workflow, and project-level choices.
- `RESEARCH_LOG.md` is for what the strategy testing is teaching us.

---

## Research Entry Template

## R-###  
**Date:** YYYY-MM-DD  
**Topic:**  
**Question:**  
**What was tested:**  
**Finding:**  
**Conclusion:**  
**Follow-up:**  

## Example Template

## R-000  
**Date:** 2026-02-26  
**Topic:** Baseline tracker comparison  
**Question:** Which lightweight tracker should be used as the first baseline?  
**What was tested:** Compared two candidate tracking approaches on a small validation sample.  
**Finding:** The simpler baseline was easier to debug and was accurate enough for the prototype stage.  
**Conclusion:** Keep the lightweight tracker as the working baseline for now.  
**Follow-up:** Revisit more advanced tracking only if failure cases become a bottleneck.  

---

## R-001
**Date:** 2026-04-16  
**Topic:** Pullback-timing parameters appear to be stock-sensitive  
**Question:** Do pullback-timing parameters in `macd_pullback` behave like universal global improvements, or do they depend on the stock?  
**What was tested:** Reviewed live AMD behavior and ran parameter sweeps on `max_bars_since_pullback_touch` and `pullback_memory_bars` using the tuned `macd_pullback` baseline. AMD was tested over `2023-03-31` to `2026-03-31`. NVDA and MSFT were tested over `2021-03-31` to `2026-03-31` for consistency with the prior baseline research window.  
**Finding:** `max_bars_since_pullback_touch` helped NVDA when loosened from `8` to `10`, but did not generalize cleanly to AMD or MSFT. `pullback_memory_bars` improved AMD when loosened, NVDA preferred the current baseline around `20`, and MSFT changed only slightly with a small edge around `24`. Across these sweeps, the changes mostly produced incremental improvements rather than dramatic performance shifts.  
**Conclusion:** Pullback-timing parameters appear to behave more like stock-fit parameters than universal global upgrades. NVDA seems to reward faster continuation timing, AMD appears to need more setup memory, and MSFT is comparatively stable under these timing changes. The main research takeaway is not that one new global setting has been found, but that symbol behavior is materially shaping how the current strategy expresses its edge.  
**Follow-up:** Keep the current shared baseline unless broader evidence supports a promotion. Treat pullback-timing settings as high-priority candidates for symbol-specific research or future adaptive logic rather than assuming one value should work best for every stock.  

## R-002
**Date:** 2026-04-19  
**Topic:** Current trend-alignment requirement was filtering out valid pullback continuations  
**Question:** Does `macd_pullback` need the recent trend state to remain aligned during the pullback, or is that requirement too strict for the kind of setups the strategy is trying to capture?  
**What was tested:** Added a new strategy parameter, `require_recent_trend_alignment`, and swept `true` versus `false` on the existing `5m` `macd_pullback` baseline. NVDA and MSFT were tested over `2021-03-31` to `2026-03-31`. AMD was tested over `2023-03-31` to `2026-03-31`. This idea came from reviewing live NVDA and AMD sessions where the pullback looked tradable on the chart, but the strategy stayed flat because the trend state no longer looked clean enough by the time price recovered.  
**Finding:** Turning `require_recent_trend_alignment` off improved all three tested symbols. NVDA improved from `0.61%` return to `0.95%` with profit factor holding near `1.12` while trade count rose from `669` to `933`. MSFT improved from `0.15%` return / `1.17` PF to `0.26%` return / `1.21` PF while trades rose from `154` to `211`. AMD remained negative, but improved materially from `-0.96%` return / `0.72` PF to `-0.57%` return / `0.87` PF while trades rose from `381` to `503`.  
**Conclusion:** This appears to be a real conceptual shift in the strategy rather than a minor threshold tweak. The earlier assumption was that valid pullbacks should preserve same-direction trend alignment throughout the setup. The sweeps suggest that this requirement was often filtering out winners by demanding a cleaner trend state than a real pullback naturally allows. A more natural interpretation is that pullback continuations require meaningful recent trend context and a non-sideways market, but not necessarily pristine current trend alignment at the moment of entry.  
**Follow-up:** Promote `require_recent_trend_alignment = false` to the working default and treat this as the new baseline assumption for `macd_pullback`. Continue testing whether this change also improves live paper-trading activity and whether additional filters, including possible XGBoost ranking or filtering later, can help trim weaker setups without restoring the overly rigid trend gate.  

## R-003
**Date:** 2026-04-21  
**Topic:** First pooled XGBoost filter did not beat plain `macd_pullback` on the final holdout  
**Question:** Can one pooled XGBoost entry filter improve trade quality enough across the active symbol set to replace plain `macd_pullback` for April 22, 2026 paper trading?  
**What was tested:** Implemented a pooled XGBoost workflow using the active symbols from `bot_config.toml`: `AAPL`, `AMD`, `AMZN`, `META`, `SPY`, `QQQ`, `IWM`, `ARKK`, `MSFT`, and `NVDA`. Three pooled feature variants were compared: `pooled_symbol_only`, `pooled_context_only`, and `pooled_hybrid`. The train window was January 3, 2023 through December 31, 2024, the validation window was January 1, 2025 through September 30, 2025, and the final out-of-sample test window was October 1, 2025 through April 21, 2026. Threshold selection was done on the validation window, and the test comparison used actual filtered-strategy backtests rather than only trade-label classification metrics.  
**Finding:** The best validation result came from `pooled_symbol_only` with threshold `0.50`. The context-only and hybrid variants both underperformed it on validation. On the final test window, the plain `macd_pullback` baseline remained stronger overall: baseline combined profit factor was `1.1601` versus `1.0828` for the pooled filter, baseline average per-symbol total return was `0.0377%` versus `0.0120%`, and baseline weighted average trade return was `0.0535%` versus `0.0301%`. The pooled filter did reduce drawdown somewhat, with worst-symbol max drawdown improving from `-0.2375%` to `-0.2207%` and average symbol drawdown improving from `-0.0909%` to `-0.0762%`, but the trade-quality improvement was not broad enough. The filter helped `AAPL`, `AMZN`, `META`, `MSFT`, and `QQQ`, but hurt `AMD`, `ARKK`, `IWM`, `NVDA`, and `SPY`. The fact that the winning pooled variant was `symbol_only` and still chose a low threshold of `0.50` suggests the current model is learning symbol priors more strongly than robust cross-symbol setup quality.  
**Conclusion:** The pooled XGBoost infrastructure is now usable, but the current pooled model is still experimental and is not ready to replace plain `macd_pullback` for the April 22, 2026 paper-trading session.  
**Follow-up:** Keep tomorrow on the plain `macd_pullback` baseline. If ML filtering is revisited, the next research step should focus on stronger walk-forward validation, possible per-symbol gating or calibration, and better pooled labels/features that capture trade quality without collapsing into mostly symbol-driven behavior.  

## R-004
**Date:** 2026-04-21  
**Topic:** First baseline comparison for `vwap_rsi_mean_reversion`  
**Question:** Does the new VWAP + RSI mean-reversion strategy behave differently enough from `macd_pullback` to justify keeping it as an active research path, and is it close to paper-trading readiness yet?  
**What was tested:** Ran one-year `5m` backtests from April 21, 2025 through April 21, 2026 on both `MSFT` and `NVDA`, comparing `vwap_rsi_mean_reversion` against plain `macd_pullback`.  
**Finding:** The new strategy clearly behaved like a distinct system rather than a noisy copy of `macd_pullback`. On `MSFT`, it underperformed the pullback baseline: `vwap_rsi_mean_reversion` returned `-0.06%` with `77` trades, `41.56%` win rate, and `0.75` profit factor, while `macd_pullback` returned `+0.03%` with `22` trades, `50.00%` win rate, and `1.30` profit factor. On `NVDA`, the new strategy looked stronger: `vwap_rsi_mean_reversion` returned `+0.22%` with `127` trades, `51.18%` win rate, and `1.50` profit factor, while `macd_pullback` returned `+0.06%` with `90` trades, `47.78%` win rate, and `1.11` profit factor. The mean-reversion strategy also relied heavily on `stop_loss` and `time_stop` exits, which suggests the first version is viable but still rough around trade management.  
**Conclusion:** `vwap_rsi_mean_reversion` is worth keeping as a serious research path because it is genuinely different from the trend-following baseline and already shows some promise on a volatile name like `NVDA`. It is not ready to replace the current paper-trading baseline across the active symbol set yet because the first-pass results were mixed and symbol-sensitive.  
**Follow-up:** Keep the strategy in research mode, then tune the highest-leverage mean-reversion parameters before any live promotion. The first follow-up sweep should focus on VWAP distance, RSI thresholds, max holding bars, and how aggressively the take-profit aims for full VWAP reversion.  

## R-005
**Date:** 2026-04-21  
**Topic:** NVDA sweep takeaways for `vwap_rsi_mean_reversion`  
**Question:** Which first-pass parameter changes look most promising on `NVDA` for the new mean-reversion strategy?  
**What was tested:** Ran one-year `5m` sweeps on `NVDA` over April 21, 2025 through April 21, 2026 for `min_distance_from_vwap_frac`, `max_bars_in_trade`, `take_profit_vwap_fraction`, and `rsi_oversold_threshold` while holding `rsi_overbought_threshold=65` during the oversold-threshold sweep.  
**Finding:** Extending the time stop helped the most cleanly: `max_bars_in_trade=15` was the best run in this batch at `+0.28%`, `50.39%` win rate, and `1.59` profit factor. A slightly looser VWAP-distance filter also helped modestly: `min_distance_from_vwap_frac=0.002` edged out the default `0.003` with `+0.23%` versus `+0.22%`, while tighter distances reduced both trade count and return. Changing `take_profit_vwap_fraction` mostly changed trade style rather than raw return: `0.5` kept return near baseline at `+0.21%` but increased win rate to `60.00%`, while `1.0` kept the fuller reversion target with a lower win rate and similar total return. The RSI oversold sweep was less convincing: `35` produced the best headline return at `+0.26%`, but trade count jumped to `297` and profit factor fell to `1.23`, which looks more like aggressive overtrading than a clean upgrade.  
**Conclusion:** The strongest current candidates are `max_bars_in_trade=15` and, secondarily, `min_distance_from_vwap_frac=0.002`. The best take-profit setting depends on whether the team prefers a higher-win-rate profile (`0.5`) or a more complete reversion target (`1.0`). The RSI sweep did not produce a clear, robust improvement.  
**Follow-up:** Run a combined confirmation backtest on `NVDA` that stacks the cleaner improvements together, starting with `min_distance_from_vwap_frac=0.002` and `max_bars_in_trade=15`, then compare `take_profit_vwap_fraction=0.5` versus `1.0` before considering a broader cross-symbol check.  

## R-006
**Date:** 2026-04-23  
**Topic:** Paper2 live-comparison parameter choice for `vwap_rsi_mean_reversion`  
**Question:** Which VWAP/RSI settings should be pinned in the separate Paper2 live comparison account for the first supervised side-by-side run against `macd_pullback`?  
**What was tested:** Reviewed the saved April 21, 2026 `NVDA` VWAP/RSI baseline and sweep artifacts, especially the `max_bars_in_trade`, `min_distance_from_vwap_frac`, `take_profit_vwap_fraction`, and RSI-threshold runs over April 21, 2025 through April 21, 2026. The choice needed to balance “best saved run” reproducibility against avoiding a more aggressive multi-parameter jump that had not yet been rechecked across symbols.  
**Finding:** The best saved first-pass VWAP/RSI run by total return was the `max_bars_in_trade=15` variant, which returned `+0.2751%` with `50.39%` win rate, `1.5873` profit factor, `127` trades, and `-0.0936%` max drawdown. The earlier baseline used `min_distance_from_vwap_frac=0.003`, and that distance remained part of the best saved `max_bars_in_trade=15` run. Although the separate distance sweep suggested `0.002` was a modest secondary improvement, the strongest clean result was still the saved `0.003` / `15` combination.  
**Conclusion:** The initial Paper2 live-comparison config should pin `min_distance_from_vwap_frac=0.003` and `max_bars_in_trade=15` for all `vwap_rsi_mean_reversion` symbols, while leaving the rest of the strategy on the shared current defaults. This keeps the live comparison tied to an actual saved best-run configuration instead of drifting with default changes.  
**Follow-up:** Once the separate-account live logs accumulate enough evidence, revisit whether `0.002` should replace `0.003` globally for VWAP/RSI, or whether the best distance setting is likely symbol-specific rather than a universal promotion candidate.  
