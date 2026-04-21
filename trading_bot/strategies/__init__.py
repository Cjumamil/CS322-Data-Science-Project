"""Strategy implementations and shared strategy interfaces."""

from copy import deepcopy

from trading_bot.strategies.macd_pullback import MacdPullbackStrategy
from trading_bot.strategies.macd_pullback_xgboost import MacdPullbackXgboostFilterStrategy
from trading_bot.strategies.sma_crossover import SmaCrossoverStrategy
from trading_bot.strategies.vwap_rsi_mean_reversion import VwapRsiMeanReversionStrategy


DEFAULT_STRATEGY_CONFIGS = {
    "sma_crossover": {
        "name": "sma_crossover",
        "version": "v1",
        "interval": "1m",
        "lookback_bars": 1000,
        "fast_window": 8,
        "slow_window": 20,
    },
    "macd_pullback": {
        "name": "macd_pullback",
        "version": "v2",
        "interval": "5m",
        "lookback_bars": 700,
        "ema_fast_window": 12,
        "ema_slow_window": 26,
        "macd_fast_window": 12,
        "macd_slow_window": 26,
        "macd_signal_window": 9,
        "ema_band_window": 72,
        "opening_no_trade_bars": 6,
        "trend_persistence_lookback": 12,
        "min_bars_above_band_for_long": 8,
        "min_bars_below_band_for_short": 8,
        "trend_slope_lookback": 12,
        "min_ema_band_slope_frac": 0.001,
        "sideways_lookback": 12,
        "max_inside_band_bars": 9,
        "require_recent_trend_alignment": False,
        "require_prior_impulse": True,
        "prior_impulse_lookback": 20,
        "min_prior_impulse_frac_of_price": 0.01,
        "pullback_memory_bars": 20,
        "max_bars_since_pullback_touch": 8,
        "max_pullback_band_overshoot_frac": 0.002,
        "require_macd_above_zero_for_long": True,
        "require_macd_below_zero_for_short": True,
        "require_histogram_reexpansion": True,
        "require_ema12_reclaim": True,
        "require_ema26_reclaim": True,
        "require_pullback_breakout": True,
        "pullback_breakout_lookback": 3,
        "ema_stop_buffer_pct": 0.001,
        "take_profit_risk_multiple": 1.5,
        "max_stop_distance_frac_of_price": 0.04,
        "enable_time_stop": True,
        "max_bars_in_trade": 30,
        "enable_macd_failure_exit": True,
        "min_bars_before_macd_exit": 10,
        "macd_exit_slope_lookback": 2,
        "macd_exit_min_slope_frac_of_price": 0.0008,
        "macd_exit_requires_histogram_confirmation": True,
        "macd_exit_requires_ema12_confirmation": False,
    },
    "macd_pullback_xgboost_filter": {
        "name": "macd_pullback_xgboost_filter",
        "version": "v1",
        "interval": "5m",
        "lookback_bars": 700,
        "ema_fast_window": 12,
        "ema_slow_window": 26,
        "macd_fast_window": 12,
        "macd_slow_window": 26,
        "macd_signal_window": 9,
        "ema_band_window": 72,
        "opening_no_trade_bars": 6,
        "trend_persistence_lookback": 12,
        "min_bars_above_band_for_long": 8,
        "min_bars_below_band_for_short": 8,
        "trend_slope_lookback": 12,
        "min_ema_band_slope_frac": 0.001,
        "sideways_lookback": 12,
        "max_inside_band_bars": 9,
        "require_recent_trend_alignment": False,
        "require_prior_impulse": True,
        "prior_impulse_lookback": 20,
        "min_prior_impulse_frac_of_price": 0.01,
        "pullback_memory_bars": 20,
        "max_bars_since_pullback_touch": 8,
        "max_pullback_band_overshoot_frac": 0.002,
        "require_macd_above_zero_for_long": True,
        "require_macd_below_zero_for_short": True,
        "require_histogram_reexpansion": True,
        "require_ema12_reclaim": True,
        "require_ema26_reclaim": True,
        "require_pullback_breakout": True,
        "pullback_breakout_lookback": 3,
        "ema_stop_buffer_pct": 0.001,
        "take_profit_risk_multiple": 1.5,
        "max_stop_distance_frac_of_price": 0.04,
        "enable_time_stop": True,
        "max_bars_in_trade": 30,
        "enable_macd_failure_exit": True,
        "min_bars_before_macd_exit": 10,
        "macd_exit_slope_lookback": 2,
        "macd_exit_min_slope_frac_of_price": 0.0008,
        "macd_exit_requires_histogram_confirmation": True,
        "macd_exit_requires_ema12_confirmation": False,
        "xgb_model_path": "",
        "xgb_probability_threshold": 0.55,
    },
    "vwap_rsi_mean_reversion": {
        "name": "vwap_rsi_mean_reversion",
        "version": "v1",
        "interval": "5m",
        "lookback_bars": 500,
        "rsi_window": 14,
        "rsi_oversold_threshold": 30.0,
        "rsi_overbought_threshold": 70.0,
        "min_distance_from_vwap_frac": 0.003,
        "confirmation_lookback_bars": 3,
        "min_reversal_move_frac_of_price": 0.0015,
        "require_reversal_bar": True,
        "require_candle_color_confirmation": False,
        "stop_reference_lookback_bars": 4,
        "stop_buffer_pct": 0.001,
        "take_profit_vwap_fraction": 1.0,
        "max_stop_distance_frac_of_price": 0.02,
        "entry_start_minutes_after_open": 30,
        "entry_end_minutes_before_close": 30,
        "enable_time_stop": True,
        "max_bars_in_trade": 12,
        "enable_extreme_trend_filter": True,
        "trend_lookback_bars": 6,
        "max_trend_move_frac_of_price": 0.02,
    },
}


def list_supported_strategies() -> list[str]:
    """Return the supported built-in strategy names."""
    return sorted(DEFAULT_STRATEGY_CONFIGS)


def default_strategy_config(name: str) -> dict:
    """Return a copy of the default configuration for one strategy."""
    if name not in DEFAULT_STRATEGY_CONFIGS:
        raise ValueError(f"Unsupported strategy in config: {name}")
    return deepcopy(DEFAULT_STRATEGY_CONFIGS[name])


def create_strategy(strategy_config: dict):
    """Build one strategy instance from config data."""
    name = strategy_config["name"]

    if name == "sma_crossover":
        return SmaCrossoverStrategy(
            fast_window=int(strategy_config["fast_window"]),
            slow_window=int(strategy_config["slow_window"]),
            interval=str(strategy_config["interval"]),
            lookback_bars=int(strategy_config["lookback_bars"]),
            version=str(strategy_config.get("version", "v1")),
        )

    if name == "macd_pullback":
        return MacdPullbackStrategy(
            interval=str(strategy_config["interval"]),
            lookback_bars=int(strategy_config["lookback_bars"]),
            ema_fast_window=int(strategy_config.get("ema_fast_window", 12)),
            ema_slow_window=int(strategy_config.get("ema_slow_window", 26)),
            macd_fast_window=int(strategy_config["macd_fast_window"]),
            macd_slow_window=int(strategy_config["macd_slow_window"]),
            macd_signal_window=int(strategy_config["macd_signal_window"]),
            ema_band_window=int(strategy_config.get("ema_band_window", 200)),
            opening_no_trade_bars=int(strategy_config["opening_no_trade_bars"]),
            trend_persistence_lookback=int(strategy_config.get("trend_persistence_lookback", 12)),
            min_bars_above_band_for_long=int(strategy_config.get("min_bars_above_band_for_long", 8)),
            min_bars_below_band_for_short=int(strategy_config.get("min_bars_below_band_for_short", 8)),
            trend_slope_lookback=int(strategy_config.get("trend_slope_lookback", 12)),
            min_ema_band_slope_frac=float(strategy_config.get("min_ema_band_slope_frac", 0.001)),
            sideways_lookback=int(strategy_config.get("sideways_lookback", 12)),
            max_inside_band_bars=int(strategy_config.get("max_inside_band_bars", 9)),
            require_recent_trend_alignment=bool(strategy_config.get("require_recent_trend_alignment", False)),
            require_prior_impulse=bool(strategy_config.get("require_prior_impulse", True)),
            prior_impulse_lookback=int(strategy_config.get("prior_impulse_lookback", 20)),
            min_prior_impulse_frac_of_price=float(strategy_config.get("min_prior_impulse_frac_of_price", 0.01)),
            pullback_memory_bars=int(strategy_config.get("pullback_memory_bars", 20)),
            max_bars_since_pullback_touch=int(strategy_config.get("max_bars_since_pullback_touch", 8)),
            max_pullback_band_overshoot_frac=float(strategy_config.get("max_pullback_band_overshoot_frac", 0.002)),
            require_macd_above_zero_for_long=bool(strategy_config.get("require_macd_above_zero_for_long", True)),
            require_macd_below_zero_for_short=bool(strategy_config.get("require_macd_below_zero_for_short", True)),
            require_histogram_reexpansion=bool(strategy_config.get("require_histogram_reexpansion", True)),
            require_ema12_reclaim=bool(strategy_config.get("require_ema12_reclaim", True)),
            require_ema26_reclaim=bool(strategy_config.get("require_ema26_reclaim", True)),
            require_pullback_breakout=bool(strategy_config.get("require_pullback_breakout", False)),
            pullback_breakout_lookback=int(strategy_config.get("pullback_breakout_lookback", 5)),
            ema_stop_buffer_pct=float(strategy_config["ema_stop_buffer_pct"]),
            take_profit_risk_multiple=float(strategy_config["take_profit_risk_multiple"]),
            max_stop_distance_frac_of_price=(
                None
                if strategy_config.get("max_stop_distance_frac_of_price") in {None, ""}
                else float(strategy_config["max_stop_distance_frac_of_price"])
            ),
            enable_time_stop=bool(strategy_config.get("enable_time_stop", False)),
            max_bars_in_trade=int(strategy_config.get("max_bars_in_trade", 18)),
            enable_macd_failure_exit=bool(strategy_config.get("enable_macd_failure_exit", False)),
            min_bars_before_macd_exit=int(strategy_config.get("min_bars_before_macd_exit", 4)),
            macd_exit_slope_lookback=int(strategy_config.get("macd_exit_slope_lookback", 1)),
            macd_exit_min_slope_frac_of_price=float(
                strategy_config.get("macd_exit_min_slope_frac_of_price", 0.0005)
            ),
            macd_exit_requires_histogram_confirmation=bool(
                strategy_config.get("macd_exit_requires_histogram_confirmation", True)
            ),
            macd_exit_requires_ema12_confirmation=bool(
                strategy_config.get("macd_exit_requires_ema12_confirmation", True)
            ),
            version=str(strategy_config.get("version", "v2")),
        )

    if name == "macd_pullback_xgboost_filter":
        base_strategy = MacdPullbackStrategy(
            interval=str(strategy_config["interval"]),
            lookback_bars=int(strategy_config["lookback_bars"]),
            ema_fast_window=int(strategy_config.get("ema_fast_window", 12)),
            ema_slow_window=int(strategy_config.get("ema_slow_window", 26)),
            macd_fast_window=int(strategy_config["macd_fast_window"]),
            macd_slow_window=int(strategy_config["macd_slow_window"]),
            macd_signal_window=int(strategy_config["macd_signal_window"]),
            ema_band_window=int(strategy_config.get("ema_band_window", 200)),
            opening_no_trade_bars=int(strategy_config["opening_no_trade_bars"]),
            trend_persistence_lookback=int(strategy_config.get("trend_persistence_lookback", 12)),
            min_bars_above_band_for_long=int(strategy_config.get("min_bars_above_band_for_long", 8)),
            min_bars_below_band_for_short=int(strategy_config.get("min_bars_below_band_for_short", 8)),
            trend_slope_lookback=int(strategy_config.get("trend_slope_lookback", 12)),
            min_ema_band_slope_frac=float(strategy_config.get("min_ema_band_slope_frac", 0.001)),
            sideways_lookback=int(strategy_config.get("sideways_lookback", 12)),
            max_inside_band_bars=int(strategy_config.get("max_inside_band_bars", 9)),
            require_recent_trend_alignment=bool(strategy_config.get("require_recent_trend_alignment", False)),
            require_prior_impulse=bool(strategy_config.get("require_prior_impulse", True)),
            prior_impulse_lookback=int(strategy_config.get("prior_impulse_lookback", 20)),
            min_prior_impulse_frac_of_price=float(strategy_config.get("min_prior_impulse_frac_of_price", 0.01)),
            pullback_memory_bars=int(strategy_config.get("pullback_memory_bars", 20)),
            max_bars_since_pullback_touch=int(strategy_config.get("max_bars_since_pullback_touch", 8)),
            max_pullback_band_overshoot_frac=float(strategy_config.get("max_pullback_band_overshoot_frac", 0.002)),
            require_macd_above_zero_for_long=bool(strategy_config.get("require_macd_above_zero_for_long", True)),
            require_macd_below_zero_for_short=bool(strategy_config.get("require_macd_below_zero_for_short", True)),
            require_histogram_reexpansion=bool(strategy_config.get("require_histogram_reexpansion", True)),
            require_ema12_reclaim=bool(strategy_config.get("require_ema12_reclaim", True)),
            require_ema26_reclaim=bool(strategy_config.get("require_ema26_reclaim", True)),
            require_pullback_breakout=bool(strategy_config.get("require_pullback_breakout", False)),
            pullback_breakout_lookback=int(strategy_config.get("pullback_breakout_lookback", 5)),
            ema_stop_buffer_pct=float(strategy_config["ema_stop_buffer_pct"]),
            take_profit_risk_multiple=float(strategy_config["take_profit_risk_multiple"]),
            max_stop_distance_frac_of_price=(
                None
                if strategy_config.get("max_stop_distance_frac_of_price") in {None, ""}
                else float(strategy_config["max_stop_distance_frac_of_price"])
            ),
            enable_time_stop=bool(strategy_config.get("enable_time_stop", False)),
            max_bars_in_trade=int(strategy_config.get("max_bars_in_trade", 18)),
            enable_macd_failure_exit=bool(strategy_config.get("enable_macd_failure_exit", False)),
            min_bars_before_macd_exit=int(strategy_config.get("min_bars_before_macd_exit", 4)),
            macd_exit_slope_lookback=int(strategy_config.get("macd_exit_slope_lookback", 1)),
            macd_exit_min_slope_frac_of_price=float(
                strategy_config.get("macd_exit_min_slope_frac_of_price", 0.0005)
            ),
            macd_exit_requires_histogram_confirmation=bool(
                strategy_config.get("macd_exit_requires_histogram_confirmation", True)
            ),
            macd_exit_requires_ema12_confirmation=bool(
                strategy_config.get("macd_exit_requires_ema12_confirmation", True)
            ),
            version=str(strategy_config.get("base_strategy_version", "v2")),
        )
        return MacdPullbackXgboostFilterStrategy(
            base_strategy=base_strategy,
            xgb_model_path=str(strategy_config.get("xgb_model_path", "")),
            xgb_probability_threshold=float(strategy_config.get("xgb_probability_threshold", 0.55)),
            symbol=None if strategy_config.get("symbol") in {None, ""} else str(strategy_config.get("symbol")).upper(),
            version=str(strategy_config.get("version", "v1")),
        )

    if name == "vwap_rsi_mean_reversion":
        return VwapRsiMeanReversionStrategy(
            interval=str(strategy_config["interval"]),
            lookback_bars=int(strategy_config["lookback_bars"]),
            rsi_window=int(strategy_config.get("rsi_window", 14)),
            rsi_oversold_threshold=float(strategy_config.get("rsi_oversold_threshold", 30.0)),
            rsi_overbought_threshold=float(strategy_config.get("rsi_overbought_threshold", 70.0)),
            min_distance_from_vwap_frac=float(strategy_config.get("min_distance_from_vwap_frac", 0.003)),
            confirmation_lookback_bars=int(strategy_config.get("confirmation_lookback_bars", 3)),
            min_reversal_move_frac_of_price=float(
                strategy_config.get("min_reversal_move_frac_of_price", 0.0015)
            ),
            require_reversal_bar=bool(strategy_config.get("require_reversal_bar", True)),
            require_candle_color_confirmation=bool(
                strategy_config.get("require_candle_color_confirmation", False)
            ),
            stop_reference_lookback_bars=int(strategy_config.get("stop_reference_lookback_bars", 4)),
            stop_buffer_pct=float(strategy_config.get("stop_buffer_pct", 0.001)),
            take_profit_vwap_fraction=float(strategy_config.get("take_profit_vwap_fraction", 1.0)),
            max_stop_distance_frac_of_price=(
                None
                if strategy_config.get("max_stop_distance_frac_of_price") in {None, ""}
                else float(strategy_config["max_stop_distance_frac_of_price"])
            ),
            entry_start_minutes_after_open=int(strategy_config.get("entry_start_minutes_after_open", 30)),
            entry_end_minutes_before_close=int(strategy_config.get("entry_end_minutes_before_close", 30)),
            enable_time_stop=bool(strategy_config.get("enable_time_stop", True)),
            max_bars_in_trade=int(strategy_config.get("max_bars_in_trade", 12)),
            enable_extreme_trend_filter=bool(strategy_config.get("enable_extreme_trend_filter", True)),
            trend_lookback_bars=int(strategy_config.get("trend_lookback_bars", 6)),
            max_trend_move_frac_of_price=float(strategy_config.get("max_trend_move_frac_of_price", 0.02)),
            version=str(strategy_config.get("version", "v1")),
        )

    raise ValueError(f"Unsupported strategy in config: {name}")


__all__ = [
    "MacdPullbackStrategy",
    "MacdPullbackXgboostFilterStrategy",
    "SmaCrossoverStrategy",
    "VwapRsiMeanReversionStrategy",
    "create_strategy",
    "default_strategy_config",
    "list_supported_strategies",
]
