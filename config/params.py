from dataclasses import dataclass, field
from typing import Dict, List, Union, Any

# indicator_config хранит списки фиксированной длины (для совместимости с текущим кодом)
# ema: [enabled, sign, fast, slow]
# rsi: [enabled, sign, level, period]
# volume (зарезервировано): [enabled]
IndicatorValue = Union[int, float, bool, str, None]
IndicatorConfig = Dict[str, List[IndicatorValue]]

DEFAULT_INDICATOR_CONFIG: IndicatorConfig = {
    "ema":    [False, None, None, None],
    "rsi":    [False, None, None, None],
    "volume": [False],
}

def _copy_default_indicator_config() -> IndicatorConfig:
    return {k: list(v) for k, v in DEFAULT_INDICATOR_CONFIG.items()}

@dataclass
class StrategyParams:
    # --- core ---
    sl: float = 3.0
    tp: float = 4.0
    delay_open: int = 0
    holding_minutes: int = 600

    # --- PSAR trailing stop ---
    psar_enabled: bool = False
    psar_max: float = 0.1
    psar_step: float = 0.005

    # --- Trailing Stop ---
    ts_enabled: bool = False
    ts_dist: float = 2.0
    ts_step: float = 0.5

    # --- market ---
    bar_minutes: int = 15

    # --- execution costs ---
    commission: float = 0.02
    # Слиппедж: доля цены (0.0004 = 4 bps)
    slippage: float = 0.0004

    # --- filters/indicators ---
    indicator_config: IndicatorConfig = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_INDICATOR_CONFIG.items()}
    )


def _bool(x, default=False) -> bool:
    if x is None:
        return bool(default)
    return bool(x)

def build_single_params(args: Any) -> StrategyParams:
    """Сбор параметров из argparse (run_single.py)."""
    indicator_config = _copy_default_indicator_config()

    ema_use = _bool(getattr(args, "ema_use", False))
    rsi_use = _bool(getattr(args, "rsi_use", False))
    psar_use = _bool(getattr(args, "psar_use", False))
    ts_use = _bool(getattr(args, "ts_use", False))

    if ema_use:
        indicator_config["ema"] = [
            True,
            getattr(args, "ema_sign", None),
            getattr(args, "ema_fast", None),
            getattr(args, "ema_slow", None),
        ]

    if rsi_use:
        indicator_config["rsi"] = [
            True,
            getattr(args, "rsi_sign", None),
            getattr(args, "rsi_level", None),
            getattr(args, "rsi_period", None),
        ]

    return StrategyParams(
        sl=float(getattr(args, "sl", 3.0)),
        tp=float(getattr(args, "tp", 4.0)),
        delay_open=int(getattr(args, "delay_open", 0)),
        holding_minutes=int(getattr(args, "holding_minutes", 600)),

        psar_enabled=psar_use,
        psar_max=float(getattr(args, "psar_max", 0.1)),
        psar_step=float(getattr(args, "psar_step", 0.005)),

        ts_enabled=ts_use,
        ts_dist=float(getattr(args, "ts_dist", 2.0)),
        ts_step=float(getattr(args, "ts_step", 0.5)),

        indicator_config=indicator_config,
        commission=float(getattr(args, "commission", 0.02)),
        slippage=float(getattr(args, "slippage", 0.0004)),
        bar_minutes=int(getattr(args, "bar_minutes", 15)),
    )

def build_optuna_params(trial, args: Any) -> StrategyParams:
    indicator_config = _copy_default_indicator_config()

    sl = trial.suggest_float("sl", args.sl_min, args.sl_max, step=args.sl_step)
    tp = trial.suggest_float("tp", args.tp_min, args.tp_max, step=args.tp_step)

    delay_open = trial.suggest_int("delay_open", args.delay_open_min, args.delay_open_max, step=args.delay_open_step)
    holding_minutes = trial.suggest_int("holding_minutes", args.holding_minutes_min, args.holding_minutes_max, step=args.holding_minutes_step)
    
    # --- PSAR gate/use ---
    psar_use = _bool(getattr(args, "psar_use", False))
    if psar_use:
        psar_enabled = trial.suggest_categorical("psar_enabled", [False, True])
        psar_max = trial.suggest_float("psar_max", 0.05, 0.5, step=0.05)
        psar_step = trial.suggest_float("psar_step", 0.001, 0.01, step=0.001)
    else:
        psar_enabled = False
        trial.suggest_categorical("psar_enabled", [False])
        psar_max = float(getattr(args, "psar_max", 0.1))
        psar_step = float(getattr(args, "psar_step", 0.005))

    # --- TS gate/use ---
    ts_use = _bool(getattr(args, "ts_use", False))
    if ts_use:
        ts_enabled = trial.suggest_categorical("ts_enabled", [False, True])
        ts_step = float(getattr(args, "ts_step", 0.5))
        ts_dist = trial.suggest_float("ts_dist", 0.5, 5.0, step=ts_step)
    else:
        ts_enabled = False
        trial.suggest_categorical("ts_enabled", [False])
        ts_step = float(getattr(args, "ts_step", 0.5))
        ts_dist = float(getattr(args, "ts_dist", 2.0))

    delay_open = trial.suggest_int("delay_open", args.delay_open_min, args.delay_open_max, step=args.delay_open_step)
    holding_minutes = trial.suggest_int("holding_minutes", args.holding_minutes_min, args.holding_minutes_max, step=args.holding_minutes_step)

    # --- EMA gate/use ---
    ema_use = _bool(getattr(args, "ema_use", False))
    if ema_use:
        ema_enabled = trial.suggest_categorical("ema_enabled", [False, True])
        if ema_enabled:
            ema_sign = trial.suggest_categorical("ema_sign", ["above", "below"])
            ema_fast = trial.suggest_int("ema_fast", 10, 30, step=5)
            ema_slow = trial.suggest_int("ema_slow", 40, 120, step=5)
            if ema_fast >= ema_slow:
                ema_fast = max(5, min(int(ema_fast), int(ema_slow) - 1))
            indicator_config["ema"] = [True, ema_sign, int(ema_fast), int(ema_slow)]
    else:
        trial.suggest_categorical("ema_enabled", [False])


    # --- RSI gate/use ---
    rsi_use = _bool(getattr(args, "rsi_use", False))
    if rsi_use:
        rsi_enabled = trial.suggest_categorical("rsi_enabled", [False, True])
        if rsi_enabled:
            rsi_sign = trial.suggest_categorical("rsi_sign", ["above", "below"])
            rsi_period = trial.suggest_int("rsi_period", 12, 21, step=3)
            rsi_level = trial.suggest_int("rsi_level", 20, 80, step=10)
            indicator_config["rsi"] = [True, rsi_sign, int(rsi_level), int(rsi_period)]
    else:
        trial.suggest_categorical("rsi_enabled", [False])

    return StrategyParams(
        sl=sl,
        tp=tp,
        delay_open=delay_open,
        holding_minutes=holding_minutes,

        psar_enabled=psar_enabled,
        psar_step=psar_step,
        psar_max=psar_max,

        ts_enabled=ts_enabled,
        ts_dist=ts_dist,
        ts_step=ts_step,

        indicator_config=indicator_config,
        commission=float(getattr(args, "commission", 0.02)),
        slippage=float(getattr(args, "slippage", 0.0004)),
        bar_minutes=int(getattr(args, "bar_minutes", 15)),
    )
