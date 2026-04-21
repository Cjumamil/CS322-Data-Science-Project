"""Train and evaluate pooled XGBoost entry filters for MACD pullback."""

from __future__ import annotations

import argparse
import json
import pickle
import site
import statistics
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

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
from trading_bot.config import RiskSettings, load_config, load_raw_config
from trading_bot.data import download_data_range
from trading_bot.strategies import create_strategy
from trading_bot.xgboost_filter import (
    apply_xgboost_entry_filter,
    build_labeled_trade_examples,
    select_feature_columns,
)


@dataclass(frozen=True)
class WindowSpec:
    """One train, validation, or test window."""

    name: str
    start_utc: datetime
    end_utc: datetime


@dataclass(frozen=True)
class SymbolDataset:
    """Prepared historical data plus base strategy config for one symbol."""

    symbol: str
    strategy_config: dict
    base_strategy: object
    raw_df: pd.DataFrame
    prepared_df: pd.DataFrame


@dataclass(frozen=True)
class VariantSpec:
    """Feature-set variant for pooled XGBoost experiments."""

    name: str
    include_symbol_features: bool
    include_context_features: bool
    description: str


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for pooled MACD pullback XGBoost training."""
    parser = argparse.ArgumentParser(description="Train a pooled XGBoost filter for MACD pullback entries.")
    parser.add_argument(
        "--symbols",
        help="Optional comma-separated ticker list. Defaults to the active symbols from bot_config.toml.",
    )
    parser.add_argument("--train-start", required=True, help="Inclusive training start date or datetime.")
    parser.add_argument("--train-end", required=True, help="Inclusive training end date or exclusive end datetime.")
    parser.add_argument(
        "--validation-start",
        required=True,
        help="Inclusive validation start date or datetime used for threshold selection.",
    )
    parser.add_argument(
        "--validation-end",
        required=True,
        help="Inclusive validation end date or exclusive end datetime used for threshold selection.",
    )
    parser.add_argument(
        "--test-start",
        required=True,
        help="Inclusive test start date or datetime used for final out-of-sample evaluation.",
    )
    parser.add_argument(
        "--test-end",
        required=True,
        help="Inclusive test end date or exclusive end datetime used for final out-of-sample evaluation.",
    )
    parser.add_argument("--config", default="bot_config.toml", help="Path to the bot TOML config.")
    parser.add_argument(
        "--output-path",
        help="Path for the final saved pooled artifact. Defaults to models/pooled_macd_pullback_xgb.pkl.",
    )
    parser.add_argument(
        "--strategy-param",
        action="append",
        default=[],
        help="Override one MACD pullback config field with KEY=VALUE. Repeat to set multiple values.",
    )
    parser.add_argument("--initial-equity", type=float, default=100_000.0)
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--subsample", type=float, default=0.8)
    parser.add_argument("--colsample-bytree", type=float, default=0.8)
    parser.add_argument("--min-child-weight", type=float, default=8.0)
    parser.add_argument("--reg-lambda", type=float, default=1.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--threshold-grid",
        default="0.50,0.55,0.60,0.65,0.70,0.75",
        help="Comma-separated probability thresholds to evaluate on the validation window.",
    )
    return parser.parse_args()


def parse_symbol_list(raw_symbols: str | None, raw_config: dict) -> list[str]:
    """Return the explicit symbol list or the active symbols from config."""
    if raw_symbols:
        symbols = [part.strip().upper() for part in raw_symbols.split(",") if part.strip()]
    else:
        symbols = [str(entry["ticker"]).upper() for entry in raw_config.get("symbols", [])]
    if not symbols:
        raise ValueError("No symbols were provided and no active symbols were found in the config.")
    return list(dict.fromkeys(symbols))


def parse_threshold_grid(raw_thresholds: str) -> list[float]:
    """Parse one comma-separated threshold list."""
    thresholds = sorted({float(part.strip()) for part in raw_thresholds.split(",") if part.strip()})
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    return thresholds


def build_windows(args: argparse.Namespace) -> dict[str, WindowSpec]:
    """Build ordered UTC windows for training, validation, and test."""
    train_start_utc = to_utc(parse_window_timestamp(args.train_start, is_end=False))
    train_end_utc = to_utc(parse_window_timestamp(args.train_end, is_end=True))
    validation_start_utc = to_utc(parse_window_timestamp(args.validation_start, is_end=False))
    validation_end_utc = to_utc(parse_window_timestamp(args.validation_end, is_end=True))
    test_start_utc = to_utc(parse_window_timestamp(args.test_start, is_end=False))
    test_end_utc = to_utc(parse_window_timestamp(args.test_end, is_end=True))

    if not (train_start_utc < train_end_utc <= validation_start_utc < validation_end_utc <= test_start_utc < test_end_utc):
        raise ValueError(
            "Expected ordered non-overlapping windows: "
            "train_start < train_end <= validation_start < validation_end <= test_start < test_end."
        )

    return {
        "train": WindowSpec("train", train_start_utc, train_end_utc),
        "validation": WindowSpec("validation", validation_start_utc, validation_end_utc),
        "test": WindowSpec("test", test_start_utc, test_end_utc),
    }


def default_output_path() -> Path:
    """Return the default pooled model path."""
    return Path("models") / "pooled_macd_pullback_xgb.pkl"


def variant_temp_path(variant: VariantSpec) -> Path:
    """Return a temporary ignored artifact path for one validation-time variant."""
    temp_dir = Path("backtests") / "_tmp_xgb_filter"
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir / f"{variant.name}_candidate.pkl"


def download_symbol_dataset(
    *,
    symbol: str,
    raw_config: dict,
    window_start_utc: datetime,
    window_end_utc: datetime,
    strategy_param_overrides: list[str],
) -> SymbolDataset:
    """Download and prepare one symbol's full history for the requested windows."""
    strategy_config = resolve_strategy_config(raw_config, symbol, "macd_pullback", None)
    overrides = parse_strategy_param_overrides(strategy_param_overrides, strategy_config)
    strategy_config.update(overrides)
    strategy_config["name"] = "macd_pullback"
    strategy_config["symbol"] = symbol.upper()
    base_strategy = create_strategy(strategy_config)

    raw_df = download_data_range(
        symbol.upper(),
        base_strategy.interval,
        window_start_utc,
        window_end_utc,
        warmup_bars=base_strategy.lookback_bars,
    )
    if raw_df.empty:
        raise ValueError(f"Alpaca returned no bars for {symbol} in the requested pooled-model window.")

    prepared_df = base_strategy.prepare_dataframe(raw_df)
    prepared_df["symbol"] = symbol.upper()
    return SymbolDataset(
        symbol=symbol.upper(),
        strategy_config=strategy_config,
        base_strategy=base_strategy,
        raw_df=raw_df,
        prepared_df=prepared_df,
    )


def run_backtest_for_dataset(
    *,
    dataset: SymbolDataset,
    strategy,
    prepared_df: pd.DataFrame,
    window: WindowSpec,
    risk_settings: RiskSettings,
    initial_equity: float,
) -> tuple[pd.DataFrame, dict]:
    """Run one simulated backtest for one symbol and return trades plus summary."""
    trades_df, _, summary = simulate_backtest(
        symbol=dataset.symbol,
        strategy=strategy,
        prepared_df=prepared_df,
        analysis_start_utc=window.start_utc,
        analysis_end_utc=window.end_utc,
        risk_settings=risk_settings,
        initial_equity=initial_equity,
    )
    return trades_df, summary


def weighted_average(values: list[tuple[float, int]]) -> float:
    """Return one trade-count-weighted average."""
    total_weight = sum(weight for _, weight in values)
    if total_weight <= 0:
        return 0.0
    return sum(value * weight for value, weight in values) / total_weight


def aggregate_summaries(per_symbol_summaries: list[dict]) -> dict:
    """Aggregate per-symbol summaries into one pooled comparison block."""
    gross_profit = sum(float(summary["gross_profit"]) for summary in per_symbol_summaries)
    gross_loss = sum(float(summary["gross_loss"]) for summary in per_symbol_summaries)
    trade_count = sum(int(summary["trade_count"]) for summary in per_symbol_summaries)
    profits = [float(summary["total_return_pct"]) for summary in per_symbol_summaries]
    drawdowns = [float(summary["max_drawdown_pct"]) for summary in per_symbol_summaries]
    trade_weighted_avg_return = weighted_average(
        [(float(summary["average_trade_return_pct"]), int(summary["trade_count"])) for summary in per_symbol_summaries]
    )
    combined_profit_factor = gross_profit / gross_loss if gross_loss > 0 else None

    return {
        "symbol_count": len(per_symbol_summaries),
        "trade_count": trade_count,
        "gross_profit": round(gross_profit, 4),
        "gross_loss": round(gross_loss, 4),
        "combined_profit_factor": None if combined_profit_factor is None else round(combined_profit_factor, 4),
        "average_total_return_pct": round(statistics.mean(profits), 4) if profits else 0.0,
        "median_total_return_pct": round(statistics.median(profits), 4) if profits else 0.0,
        "worst_symbol_max_drawdown_pct": round(min(drawdowns), 4) if drawdowns else 0.0,
        "average_symbol_max_drawdown_pct": round(statistics.mean(drawdowns), 4) if drawdowns else 0.0,
        "weighted_average_trade_return_pct": round(trade_weighted_avg_return, 4),
        "symbols_profitable_count": sum(1 for summary in per_symbol_summaries if float(summary["total_return_pct"]) > 0),
        "symbols_with_pf_above_one_count": sum(
            1 for summary in per_symbol_summaries if summary["profit_factor"] is not None and float(summary["profit_factor"]) > 1
        ),
    }


def build_examples_for_window(
    *,
    datasets: list[SymbolDataset],
    window: WindowSpec,
    risk_settings: RiskSettings,
    initial_equity: float,
) -> tuple[pd.DataFrame, list[dict]]:
    """Backtest the base strategy and return pooled labeled examples plus summaries."""
    example_frames: list[pd.DataFrame] = []
    per_symbol_summaries: list[dict] = []

    for dataset in datasets:
        trades_df, summary = run_backtest_for_dataset(
            dataset=dataset,
            strategy=dataset.base_strategy,
            prepared_df=dataset.prepared_df,
            window=window,
            risk_settings=risk_settings,
            initial_equity=initial_equity,
        )
        per_symbol_summaries.append(summary)
        examples = build_labeled_trade_examples(dataset.prepared_df, trades_df, symbol=dataset.symbol)
        if not examples.empty:
            example_frames.append(examples)

    pooled_examples = pd.concat(example_frames, ignore_index=True) if example_frames else pd.DataFrame()
    return pooled_examples, per_symbol_summaries


def fit_model(
    *,
    examples: pd.DataFrame,
    feature_columns: list[str],
    args: argparse.Namespace,
) -> tuple[XGBClassifier, dict]:
    """Fit one conservative pooled XGBoost classifier."""
    labels = examples["label"].astype("int64")
    positives = int(labels.sum())
    negatives = int(len(labels) - positives)
    if labels.nunique() < 2:
        raise ValueError("The pooled training labels contain only one class. Expand the training window before fitting.")

    scale_pos_weight = (negatives / positives) if positives > 0 else 1.0
    model = XGBClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        subsample=args.subsample,
        colsample_bytree=args.colsample_bytree,
        min_child_weight=args.min_child_weight,
        reg_lambda=args.reg_lambda,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=args.random_state,
    )
    model.fit(examples[feature_columns], labels)
    return model, {
        "example_count": int(len(examples)),
        "positive_label_count": positives,
        "negative_label_count": negatives,
        "positive_label_rate": round(float(labels.mean()), 4),
        "scale_pos_weight": round(float(scale_pos_weight), 4),
    }


def save_artifact(
    *,
    output_path: Path,
    model: XGBClassifier,
    feature_columns: list[str],
    symbols: list[str],
    variant: VariantSpec,
    threshold: float | None,
    windows: dict[str, WindowSpec],
    training_summary: dict,
    model_params: dict,
    extra_metadata: dict | None = None,
) -> None:
    """Save one filter artifact to disk."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    artifact = {
        "model": model,
        "feature_columns": feature_columns,
        "trained_at": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "symbols": symbols,
        "strategy_name": "macd_pullback_xgboost_filter",
        "base_strategy_name": "macd_pullback",
        "variant": {
            "name": variant.name,
            "include_symbol_features": variant.include_symbol_features,
            "include_context_features": variant.include_context_features,
            "description": variant.description,
        },
        "recommended_threshold": threshold,
        "windows": {
            name: {
                "start": window.start_utc.isoformat(),
                "end": window.end_utc.isoformat(),
            }
            for name, window in windows.items()
        },
        "training_summary": training_summary,
        "model_params": model_params,
    }
    if extra_metadata:
        artifact.update(extra_metadata)

    with output_path.open("wb") as artifact_file:
        pickle.dump(artifact, artifact_file)


def evaluate_filtered_strategy(
    *,
    datasets: list[SymbolDataset],
    window: WindowSpec,
    artifact_path: Path,
    threshold: float,
    risk_settings: RiskSettings,
    initial_equity: float,
) -> tuple[list[dict], dict]:
    """Run actual filtered backtests for one threshold across all pooled symbols."""
    per_symbol_summaries: list[dict] = []
    for dataset in datasets:
        filtered_strategy_config = {
            **dataset.strategy_config,
            "name": "macd_pullback_xgboost_filter",
            "symbol": dataset.symbol,
            "xgb_model_path": str(artifact_path.resolve()),
            "xgb_probability_threshold": float(threshold),
        }
        filtered_strategy = create_strategy(filtered_strategy_config)
        filtered_prepared_df = apply_xgboost_entry_filter(dataset.prepared_df, str(artifact_path.resolve()), threshold)
        _, summary = run_backtest_for_dataset(
            dataset=dataset,
            strategy=filtered_strategy,
            prepared_df=filtered_prepared_df,
            window=window,
            risk_settings=risk_settings,
            initial_equity=initial_equity,
        )
        per_symbol_summaries.append(summary)
    return per_symbol_summaries, aggregate_summaries(per_symbol_summaries)


def build_validation_score(filtered_aggregate: dict, baseline_aggregate: dict) -> tuple:
    """Return a lexicographic score tuple used to pick the validation threshold."""
    baseline_pf = baseline_aggregate["combined_profit_factor"] or 0.0
    filtered_pf = filtered_aggregate["combined_profit_factor"] or 0.0
    baseline_trade_count = int(baseline_aggregate["trade_count"])
    filtered_trade_count = int(filtered_aggregate["trade_count"])
    trade_retention = (filtered_trade_count / baseline_trade_count) if baseline_trade_count > 0 else 0.0
    minimum_required_trades = max(5, int(round(baseline_trade_count * 0.25)))
    is_valid = filtered_trade_count >= minimum_required_trades

    return (
        1 if is_valid else 0,
        round(filtered_pf - baseline_pf, 6),
        round(
            filtered_aggregate["worst_symbol_max_drawdown_pct"] - baseline_aggregate["worst_symbol_max_drawdown_pct"],
            6,
        ),
        round(
            filtered_aggregate["weighted_average_trade_return_pct"]
            - baseline_aggregate["weighted_average_trade_return_pct"],
            6,
        ),
        round(filtered_aggregate["average_total_return_pct"] - baseline_aggregate["average_total_return_pct"], 6),
        round(trade_retention, 6),
    )


def build_symbol_comparison_rows(baseline_summaries: list[dict], filtered_summaries: list[dict]) -> list[dict]:
    """Create per-symbol baseline-vs-filter comparison rows."""
    baseline_by_symbol = {summary["symbol"]: summary for summary in baseline_summaries}
    filtered_by_symbol = {summary["symbol"]: summary for summary in filtered_summaries}
    rows: list[dict] = []

    for symbol in sorted(baseline_by_symbol):
        baseline = baseline_by_symbol[symbol]
        filtered = filtered_by_symbol[symbol]
        baseline_pf = baseline["profit_factor"]
        filtered_pf = filtered["profit_factor"]
        rows.append(
            {
                "symbol": symbol,
                "baseline_trade_count": int(baseline["trade_count"]),
                "filtered_trade_count": int(filtered["trade_count"]),
                "baseline_total_return_pct": float(baseline["total_return_pct"]),
                "filtered_total_return_pct": float(filtered["total_return_pct"]),
                "total_return_delta_pct": round(
                    float(filtered["total_return_pct"]) - float(baseline["total_return_pct"]),
                    4,
                ),
                "baseline_profit_factor": None if baseline_pf is None else float(baseline_pf),
                "filtered_profit_factor": None if filtered_pf is None else float(filtered_pf),
                "profit_factor_delta": None
                if baseline_pf is None or filtered_pf is None
                else round(float(filtered_pf) - float(baseline_pf), 4),
                "baseline_max_drawdown_pct": float(baseline["max_drawdown_pct"]),
                "filtered_max_drawdown_pct": float(filtered["max_drawdown_pct"]),
                "max_drawdown_delta_pct": round(
                    float(filtered["max_drawdown_pct"]) - float(baseline["max_drawdown_pct"]),
                    4,
                ),
                "baseline_average_trade_return_pct": float(baseline["average_trade_return_pct"]),
                "filtered_average_trade_return_pct": float(filtered["average_trade_return_pct"]),
                "average_trade_return_delta_pct": round(
                    float(filtered["average_trade_return_pct"]) - float(baseline["average_trade_return_pct"]),
                    4,
                ),
                "xgb_blocked_setup_count": int(filtered["entry_filter_stats"]["setup_blocked_by_xgb_filter_count"]),
            }
        )
    return rows


def variant_specs() -> list[VariantSpec]:
    """Return the pooled feature variants to compare."""
    return [
        VariantSpec(
            name="pooled_symbol_only",
            include_symbol_features=True,
            include_context_features=False,
            description="Base MACD pullback state plus symbol identity one-hots.",
        ),
        VariantSpec(
            name="pooled_context_only",
            include_symbol_features=False,
            include_context_features=True,
            description="Base MACD pullback state plus relative volatility and candle context features.",
        ),
        VariantSpec(
            name="pooled_hybrid",
            include_symbol_features=True,
            include_context_features=True,
            description="Base MACD pullback state plus both symbol identity and context features.",
        ),
    ]


def ensure_symbol_columns(examples: pd.DataFrame, symbols: list[str]) -> pd.DataFrame:
    """Ensure every active symbol has a stable one-hot feature column."""
    if examples.empty:
        return examples
    work = examples.copy()
    symbol_series = work["symbol"].astype(str).str.upper()
    for symbol in symbols:
        column = f"symbol_{symbol.upper()}"
        work[column] = (symbol_series == symbol.upper()).astype("int8")
    return work


def main() -> None:
    """Train pooled candidates, select a threshold on validation, and test the winner."""
    args = parse_args()
    raw_config = load_raw_config(args.config)
    config = load_config(args.config)
    windows = build_windows(args)
    symbols = parse_symbol_list(args.symbols, raw_config)
    thresholds = parse_threshold_grid(args.threshold_grid)
    output_path = Path(args.output_path) if args.output_path else default_output_path()

    datasets = [
        download_symbol_dataset(
            symbol=symbol,
            raw_config=raw_config,
            window_start_utc=windows["train"].start_utc,
            window_end_utc=windows["test"].end_utc,
            strategy_param_overrides=args.strategy_param,
        )
        for symbol in symbols
    ]

    train_examples, train_base_summaries = build_examples_for_window(
        datasets=datasets,
        window=windows["train"],
        risk_settings=config.risk,
        initial_equity=args.initial_equity,
    )
    validation_examples, validation_base_summaries = build_examples_for_window(
        datasets=datasets,
        window=windows["validation"],
        risk_settings=config.risk,
        initial_equity=args.initial_equity,
    )
    _, test_base_summaries = build_examples_for_window(
        datasets=datasets,
        window=windows["test"],
        risk_settings=config.risk,
        initial_equity=args.initial_equity,
    )

    if train_examples.empty:
        raise ValueError("No pooled training examples were available from the training window.")
    if validation_examples.empty:
        raise ValueError("No validation examples were available. Expand the validation window.")

    train_examples = ensure_symbol_columns(train_examples, symbols)
    validation_examples = ensure_symbol_columns(validation_examples, symbols)

    baseline_validation_aggregate = aggregate_summaries(validation_base_summaries)
    baseline_test_aggregate = aggregate_summaries(test_base_summaries)
    variant_results: list[dict] = []
    best_variant_result: dict | None = None

    model_params = {
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "learning_rate": args.learning_rate,
        "subsample": args.subsample,
        "colsample_bytree": args.colsample_bytree,
        "min_child_weight": args.min_child_weight,
        "reg_lambda": args.reg_lambda,
        "random_state": args.random_state,
    }

    for variant in variant_specs():
        feature_columns = select_feature_columns(
            train_examples,
            include_symbol_features=variant.include_symbol_features,
            include_context_features=variant.include_context_features,
        )
        model, training_summary = fit_model(examples=train_examples, feature_columns=feature_columns, args=args)
        temp_artifact_path = variant_temp_path(variant)
        save_artifact(
            output_path=temp_artifact_path,
            model=model,
            feature_columns=feature_columns,
            symbols=symbols,
            variant=variant,
            threshold=None,
            windows=windows,
            training_summary=training_summary,
            model_params=model_params,
        )

        threshold_results: list[dict] = []
        best_threshold_result: dict | None = None
        for threshold in thresholds:
            validation_filtered_summaries, validation_filtered_aggregate = evaluate_filtered_strategy(
                datasets=datasets,
                window=windows["validation"],
                artifact_path=temp_artifact_path,
                threshold=threshold,
                risk_settings=config.risk,
                initial_equity=args.initial_equity,
            )
            score = build_validation_score(validation_filtered_aggregate, baseline_validation_aggregate)
            threshold_result = {
                "threshold": threshold,
                "score": score,
                "aggregate": validation_filtered_aggregate,
                "per_symbol": validation_filtered_summaries,
            }
            threshold_results.append(threshold_result)
            if best_threshold_result is None or threshold_result["score"] > best_threshold_result["score"]:
                best_threshold_result = threshold_result

        if best_threshold_result is None:
            raise ValueError(f"No validation thresholds were evaluated for variant {variant.name}.")

        variant_result = {
            "variant": {
                "name": variant.name,
                "include_symbol_features": variant.include_symbol_features,
                "include_context_features": variant.include_context_features,
                "description": variant.description,
            },
            "feature_column_count": len(feature_columns),
            "training_summary": training_summary,
            "best_threshold": best_threshold_result["threshold"],
            "best_validation_score": best_threshold_result["score"],
            "best_validation_aggregate": best_threshold_result["aggregate"],
            "threshold_results": [
                {
                    "threshold": result["threshold"],
                    "score": result["score"],
                    "aggregate": result["aggregate"],
                }
                for result in threshold_results
            ],
        }
        variant_results.append(variant_result)

        if best_variant_result is None or best_threshold_result["score"] > best_variant_result["best_validation_score"]:
            best_variant_result = {
                **variant_result,
                "feature_columns": feature_columns,
                "variant_spec": variant,
            }

    if best_variant_result is None:
        raise ValueError("No pooled variant was successfully evaluated.")

    train_validation_examples = pd.concat([train_examples, validation_examples], ignore_index=True)
    final_model, final_training_summary = fit_model(
        examples=train_validation_examples,
        feature_columns=best_variant_result["feature_columns"],
        args=args,
    )
    chosen_variant = best_variant_result["variant_spec"]
    chosen_threshold = float(best_variant_result["best_threshold"])

    save_artifact(
        output_path=output_path,
        model=final_model,
        feature_columns=best_variant_result["feature_columns"],
        symbols=symbols,
        variant=chosen_variant,
        threshold=chosen_threshold,
        windows=windows,
        training_summary={
            **final_training_summary,
            "refit_on_train_and_validation": True,
            "train_example_count": int(len(train_examples)),
            "validation_example_count": int(len(validation_examples)),
        },
        model_params=model_params,
        extra_metadata={
            "selection_summary": {
                "baseline_validation_aggregate": baseline_validation_aggregate,
                "chosen_validation_aggregate": best_variant_result["best_validation_aggregate"],
            }
        },
    )

    test_filtered_summaries, test_filtered_aggregate = evaluate_filtered_strategy(
        datasets=datasets,
        window=windows["test"],
        artifact_path=output_path,
        threshold=chosen_threshold,
        risk_settings=config.risk,
        initial_equity=args.initial_equity,
    )

    comparison_rows = build_symbol_comparison_rows(test_base_summaries, test_filtered_summaries)
    comparison_df = pd.DataFrame(comparison_rows)
    report = {
        "symbols": symbols,
        "artifact_path": str(output_path.resolve()),
        "chosen_variant": best_variant_result["variant"],
        "chosen_threshold": chosen_threshold,
        "windows": {
            name: {
                "start": window.start_utc.isoformat(),
                "end": window.end_utc.isoformat(),
            }
            for name, window in windows.items()
        },
        "baseline_validation_aggregate": baseline_validation_aggregate,
        "baseline_test_aggregate": baseline_test_aggregate,
        "filtered_test_aggregate": test_filtered_aggregate,
        "variant_results": variant_results,
        "test_comparison_rows": comparison_rows,
        "model_params": model_params,
    }

    report_path = output_path.with_name(f"{output_path.stem}_report.json")
    comparison_csv_path = output_path.with_name(f"{output_path.stem}_test_comparison.csv")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    comparison_df.to_csv(comparison_csv_path, index=False)

    print(f"Saved pooled XGBoost artifact: {output_path}")
    print(f"Saved pooled training report: {report_path}")
    print(f"Saved per-symbol test comparison: {comparison_csv_path}")
    print()
    print(f"Chosen variant: {best_variant_result['variant']['name']}")
    print(f"Chosen threshold: {chosen_threshold:.2f}")
    print(
        "Validation aggregate: "
        f"PF={best_variant_result['best_validation_aggregate']['combined_profit_factor']} | "
        f"avg_return={best_variant_result['best_validation_aggregate']['average_total_return_pct']:.2f}% | "
        f"worst_dd={best_variant_result['best_validation_aggregate']['worst_symbol_max_drawdown_pct']:.2f}% | "
        f"trades={best_variant_result['best_validation_aggregate']['trade_count']}"
    )
    print(
        "Test aggregate: "
        f"PF={test_filtered_aggregate['combined_profit_factor']} | "
        f"avg_return={test_filtered_aggregate['average_total_return_pct']:.2f}% | "
        f"worst_dd={test_filtered_aggregate['worst_symbol_max_drawdown_pct']:.2f}% | "
        f"trades={test_filtered_aggregate['trade_count']}"
    )


if __name__ == "__main__":
    main()
