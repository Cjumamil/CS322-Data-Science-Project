"""Train and save an XGBoost entry filter for the MACD pullback strategy."""

from __future__ import annotations

import argparse
import json
import pickle
import site
import sys
from datetime import datetime, timezone
from pathlib import Path

USER_SITE_PACKAGES = site.getusersitepackages()
WORKSPACE_XGBOOST_PACKAGES = Path(__file__).resolve().parents[1] / "workspace_pkgs" / "xgboost_local"
LEGACY_WORKSPACE_XGBOOST_PACKAGES = Path(__file__).resolve().parents[1] / ".vendor_pkgs" / "xgboost_local"
if WORKSPACE_XGBOOST_PACKAGES.exists() and str(WORKSPACE_XGBOOST_PACKAGES) not in sys.path:
    sys.path.insert(0, str(WORKSPACE_XGBOOST_PACKAGES))
elif LEGACY_WORKSPACE_XGBOOST_PACKAGES.exists() and str(LEGACY_WORKSPACE_XGBOOST_PACKAGES) not in sys.path:
    sys.path.insert(0, str(LEGACY_WORKSPACE_XGBOOST_PACKAGES))
if USER_SITE_PACKAGES and USER_SITE_PACKAGES not in sys.path:
    site.addsitedir(USER_SITE_PACKAGES)

from xgboost import XGBClassifier

from trading_bot.backtest import (
    parse_strategy_param_overrides,
    parse_window_timestamp,
    resolve_strategy_config,
    simulate_backtest,
    to_utc,
)
from trading_bot.config import load_config, load_raw_config
from trading_bot.data import download_data_range
from trading_bot.strategies import create_strategy
from trading_bot.xgboost_filter import build_training_examples


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for training one MACD pullback XGBoost filter."""
    parser = argparse.ArgumentParser(description="Train an XGBoost filter for MACD pullback entries.")
    parser.add_argument("--symbol", required=True, help="Ticker symbol to train on, for example NVDA.")
    parser.add_argument("--start", required=True, help="Inclusive training start date or datetime.")
    parser.add_argument("--end", required=True, help="Inclusive training end date or exclusive end datetime.")
    parser.add_argument("--config", default="bot_config.toml", help="Path to the bot TOML config.")
    parser.add_argument(
        "--output-path",
        help="Path for the saved pickle artifact. Defaults to models/<SYMBOL>_macd_pullback_xgb.pkl.",
    )
    parser.add_argument(
        "--strategy-param",
        action="append",
        default=[],
        help="Override one MACD pullback config field with KEY=VALUE. Repeat to set multiple values.",
    )
    parser.add_argument("--initial-equity", type=float, default=100_000.0)
    parser.add_argument("--n-estimators", type=int, default=250)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.9)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=42)
    return parser.parse_args()


def default_output_path(symbol: str) -> Path:
    """Return the default artifact path for one symbol-specific filter."""
    return Path("models") / f"{symbol.upper()}_macd_pullback_xgb.pkl"


def main() -> None:
    """Train a model artifact using historical MACD pullback trade outcomes."""
    args = parse_args()
    raw_config = load_raw_config(args.config)
    config = load_config(args.config)

    base_strategy_config = resolve_strategy_config(raw_config, args.symbol, "macd_pullback", None)
    overrides = parse_strategy_param_overrides(args.strategy_param, base_strategy_config)
    base_strategy_config.update(overrides)
    strategy = create_strategy(base_strategy_config)

    training_start_local = parse_window_timestamp(args.start, is_end=False)
    training_end_local = parse_window_timestamp(args.end, is_end=True)
    if training_end_local <= training_start_local:
        raise ValueError("Training end must be after start.")

    training_start_utc = to_utc(training_start_local)
    training_end_utc = to_utc(training_end_local)
    raw_df = download_data_range(
        args.symbol.upper(),
        strategy.interval,
        training_start_utc,
        training_end_utc,
        warmup_bars=strategy.lookback_bars,
    )
    if raw_df.empty:
        raise ValueError("Alpaca returned no bars for the requested training window.")

    prepared_df = strategy.prepare_dataframe(raw_df)
    trades_df, _, summary = simulate_backtest(
        symbol=args.symbol.upper(),
        strategy=strategy,
        prepared_df=prepared_df,
        analysis_start_utc=training_start_utc,
        analysis_end_utc=training_end_utc,
        risk_settings=config.risk,
        initial_equity=args.initial_equity,
    )
    features, labels = build_training_examples(prepared_df.loc[
        (prepared_df.index >= training_start_utc) & (prepared_df.index < training_end_utc)
    ], trades_df)
    if features.empty or labels.empty:
        raise ValueError("No training examples were available from the MACD pullback trades in the requested window.")
    if labels.nunique() < 2:
        raise ValueError("Training labels contain only one class. Expand the training window before fitting XGBoost.")

    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
    )
    model.fit(features, labels)

    output_path = Path(args.output_path) if args.output_path else default_output_path(args.symbol)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    artifact = {
        "model": model,
        "feature_columns": list(features.columns),
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "symbol": args.symbol.upper(),
        "strategy_name": strategy.name,
        "strategy_version": strategy.version,
        "strategy_config": base_strategy_config,
        "training_window": {
            "start": training_start_utc.isoformat(),
            "end": training_end_utc.isoformat(),
        },
        "training_summary": {
            "training_trade_count": int(len(trades_df)),
            "training_example_count": int(len(features)),
            "positive_label_count": int(labels.sum()),
            "negative_label_count": int(len(labels) - int(labels.sum())),
            "positive_label_rate": round(float(labels.mean()), 4),
            "base_backtest_total_return_pct": summary["total_return_pct"],
            "base_backtest_profit_factor": summary["profit_factor"],
            "base_backtest_win_rate_pct": summary["win_rate_pct"],
        },
        "model_params": {
            "n_estimators": args.n_estimators,
            "max_depth": args.max_depth,
            "learning_rate": args.learning_rate,
            "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree,
            "random_state": args.random_state,
        },
    }

    with output_path.open("wb") as artifact_file:
        pickle.dump(artifact, artifact_file)

    print(f"Saved XGBoost filter artifact: {output_path}")
    print(json.dumps(artifact["training_summary"], indent=2))


if __name__ == "__main__":
    main()
