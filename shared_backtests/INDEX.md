# Shared Backtests Index

This folder is the team's published backtest set.

Use `backtests/` as local scratch space.
Use `shared_backtests/` as the small curated set that teammates, report writers, dashboards, and AI tools should rely on.

## Read These First

- `shared_backtests/manifest.csv`: AI-friendly manifest with audience, purpose, and headline metrics for each published artifact.
- `RESEARCH_LOG.md`: best source for the strategy-testing narrative and conclusions.
- `DECISIONS.md`: best source for why the workflow, strategy defaults, and team split changed over time.
- `shared_backtests/pooled_macd_pullback_xgb_report.json`: best source for the pooled XGBoost train/validation/test story.
- `shared_backtests/pooled_macd_pullback_xgb_test_comparison.csv`: best source for symbol-by-symbol XGBoost baseline vs filtered deltas.
- `shared_backtests/*_sweep_summary.json`: compact JSON summaries for the most important parameter sweeps.
- `shared_backtests/*_sweep_comparison.csv`: spreadsheet-friendly companion files for those sweep summaries.

## Dashboard Canonical Files

Use these first for dashboard design and schema decisions:

- `shared_backtests/2026-04-21_01-41-30_NVDA_macd_pullback_2025-04-21_to_2026-04-21`
  Current `macd_pullback` one-year comparison run for `NVDA`.
- `shared_backtests/2026-04-21_01-45-07_NVDA_vwap_rsi_mean_reversion_2025-04-21_to_2026-04-21_max_bars_in_trade_15`
  Current published `vwap_rsi_mean_reversion` run that matches the promoted Paper2-style config from `R-006`.

These two runs are the easiest pair to use for a side-by-side strategy dashboard because they share the same symbol and one-year window.

## Report-Worthy Artifacts

- `nvda_trend_alignment_sweep_summary.json`
  Strong conceptual turning point. This supports `R-002` and shows why `require_recent_trend_alignment=false` became the working default on `NVDA`.
- `msft_trend_alignment_sweep_summary.json`
  Cross-symbol confirmation for the same idea on `MSFT`.
- `amd_trend_alignment_sweep_summary.json`
  Important because it shows improvement even on a harder symbol, but still not enough to make `AMD` clearly good.
- `pooled_macd_pullback_xgb_report.json`
  Best single source for the pooled XGBoost go/no-go story in `R-003`.
- `pooled_macd_pullback_xgb_test_comparison.csv`
  Best per-symbol evidence for where the pooled filter helped and hurt.
- `2026-04-21_01-41-23_NVDA_vwap_rsi_mean_reversion_2025-04-21_to_2026-04-21`
  Baseline one-year `NVDA` VWAP/RSI run from `R-004`.
- `2026-04-21_01-41-30_NVDA_macd_pullback_2025-04-21_to_2026-04-21`
  Matching one-year `NVDA` MACD run for the same comparison window in `R-004`.
- `2026-04-21_01-41-10_MSFT_vwap_rsi_mean_reversion_2025-04-21_to_2026-04-21`
  Shows that VWAP/RSI did not generalize cleanly to `MSFT` in the first pass.
- `2026-04-21_01-41-16_MSFT_macd_pullback_2025-04-21_to_2026-04-21`
  Matching `MSFT` MACD comparison run for the same window.
- `nvda_vwap_max_bars_sweep_summary.json`
  Best first-pass tuning artifact for `R-005`.
- `2026-04-21_01-45-07_NVDA_vwap_rsi_mean_reversion_2025-04-21_to_2026-04-21_max_bars_in_trade_15`
  Best saved first-pass VWAP/RSI run and the cleanest current candidate for dashboard use.

## Suggested Report Story

If you want a tight report narrative, the cleanest sequence is:

1. Start from the `macd_pullback` baseline and the move to more realistic offline backtesting.
2. Show the trend-alignment finding as a meaningful strategy interpretation change, not just a tiny threshold tweak.
3. Show the pooled XGBoost result as a serious experiment that was informative but not adopted.
4. Show VWAP/RSI as a distinct second strategy with mixed first-pass cross-symbol behavior.
5. End with the current published live-comparison style setup: baseline `macd_pullback` vs tuned `vwap_rsi_mean_reversion`.

## Optional Extra Sources

- `decision_log.csv`
  Useful if someone wants live/paper decision fields for dashboard schema ideas.
- `trade_log.csv`
  Useful if someone wants execution-log fields or examples of order lifecycle data.
- `strategy_context_log.jsonl`
  Useful if someone wants richer model/strategy context, but it is large and less teammate-friendly than the files above.

## Notes

- Every published backtest run here is copied from the ignored local `backtests/` workspace.
- The point of this folder is not completeness. The point is to preserve the evidence behind the decisions that mattered.
