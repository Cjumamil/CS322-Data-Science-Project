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
