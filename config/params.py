from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

IndicatorConfig = Dict[str, List[Optional[Union[int, bool]]]]

DEFAULT_INDICATOR_CONFIG: IndicatorConfig = {
    "ema":    [False, None, None],
    "rsi":    [False, None],
    "volume": [False],
}


@dataclass
class StrategyParams:
    # --- core ---
    sl: float = 3.0
    tp: float = 4.0
    delay_open: int = 0
    holding_minutes: int = 60

    # --- indicators ---
    indicator_config: IndicatorConfig = field(
        default_factory=lambda: DEFAULT_INDICATOR_CONFIG.copy()
    )

    # --- execution ---
    commission: float = 0.002
    slippage: float = 0.0002
    bar_minutes: int = 1

    def __post_init__(self):
        for k, v in DEFAULT_INDICATOR_CONFIG.items():
            if k not in self.indicator_config:
                self.indicator_config[k] = v.copy()


def build_optuna_params(trial, args) -> StrategyParams:
    indicator_config = {}

    # ===== EMA =====
    ema_enabled = trial.suggest_categorical("ema_enabled", [True, False])
    if ema_enabled:
        ema_fast = trial.suggest_int("ema_fast", 10, 30, step=5)
        ema_slow = trial.suggest_int("ema_slow", 40, 80, step=5)
        ema_sign = trial.suggest_categorical("ema_sign", ["above", "below"])
        indicator_config["ema"] = [True, ema_sign, ema_fast, ema_slow]
    else:
        indicator_config["ema"] = [False, None, None, None]

    # ===== RSI =====
    rsi_enabled = trial.suggest_categorical("rsi_enabled", [True, False])
    if rsi_enabled:
        rsi_period = trial.suggest_int("rsi_period", 12, 21, step=3)
        rsi_level = trial.suggest_int("rsi_level", 20, 80, step=10)
        rsi_sign = trial.suggest_categorical("rsi_sign", ["above", "below"])
        indicator_config["rsi"] = [True, rsi_sign, rsi_level, rsi_period]
    else:
        indicator_config["rsi"] = [False, None, None, None]

    # ===== VOLUME =====
    # volume_enabled = trial.suggest_categorical("volume_enabled", [True, False])
    # indicator_config["volume"] = [volume_enabled]

    return StrategyParams(
        sl=trial.suggest_float("sl", args.sl_min, args.sl_max, step=args.sl_step),
        tp=trial.suggest_float("tp", args.tp_min, args.tp_max, step=args.tp_step),
        delay_open=trial.suggest_int(
            "delay_open",
            args.delay_open_min,
            args.delay_open_max,
            step=args.delay_open_step,
        ),
        holding_minutes=trial.suggest_int(
            "holding_minutes",
            args.holding_minutes_min,
            args.holding_minutes_max,
            step=args.holding_minutes_step,
        ),
        indicator_config=indicator_config,
        commission=getattr(args, "commission", 0.002),
        slippage=getattr(args, "slippage", 0.0002),
        bar_minutes=getattr(args, "bar_minutes", 1),
    )

def build_single_params(args) -> StrategyParams:
    indicator_config = {}
    # ===== EMA =====
    ema_enabled = args.ema_enabled
    if ema_enabled:
        ema_fast = args.ema_fast
        ema_slow = args.ema_slow
        ema_sign = args.ema_sign
        indicator_config["ema"] = [True, ema_sign, ema_fast, ema_slow]
    else:
        indicator_config["ema"] = [False, None, None]

    # ===== RSI =====
    rsi_enabled = args.rsi_enabled
    if rsi_enabled:
        rsi_period = args.rsi_period
        rsi_level = args.rsi_level
        rsi_sign = args.rsi_sign
        indicator_config["rsi"] = [True, rsi_sign, rsi_level, rsi_period]
    else:
        indicator_config["rsi"] = [False, None, None, None]
    return StrategyParams(
        sl=args.sl,
        tp=args.tp,
        delay_open=args.delay_open,
        holding_minutes=args.holding_minutes,
        indicator_config=indicator_config,
        commission=getattr(args, "commission", 0.002),
        slippage=getattr(args, "slippage", 0.0002),
        bar_minutes=getattr(args, "bar_minutes", 1),
    )