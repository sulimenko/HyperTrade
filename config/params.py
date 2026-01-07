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


@dataclass
class StrategyParams:
    # --- core ---
    sl: float = 3.0
    tp: float = 4.0
    delay_open: int = 0
    holding_minutes: int = 600

    # --- market ---
    bar_minutes: int = 15

    # --- execution costs ---
    # Комиссия: 2 цента на 1 акцию за сторону (entry и exit) → в симуляторе умножаем *2
    commission: float = 0.02
    # Слиппедж: доля цены (0.0004 = 4 bps)
    slippage: float = 0.0004

    # --- filters/indicators ---
    indicator_config: IndicatorConfig = field(
        default_factory=lambda: {k: list(v) for k, v in DEFAULT_INDICATOR_CONFIG.items()}
    )

def _copy_default_indicator_config() -> IndicatorConfig:
    return {k: list(v) for k, v in DEFAULT_INDICATOR_CONFIG.items()}


def build_single_params(args: Any) -> StrategyParams:
    """Сбор параметров из argparse (run_single.py)."""
    indicator_config = _copy_default_indicator_config()

    # EMA filter
    if getattr(args, "ema_enabled", False):
        fast = getattr(args, "ema_fast", None)
        slow = getattr(args, "ema_slow", None)
        sign = getattr(args, "ema_sign", None)  # above/below
        indicator_config["ema"] = [True, sign, fast, slow]
    else:
        indicator_config["ema"] = [False, None, None, None]

    # RSI filter
    if getattr(args, "rsi_enabled", False):
        period = getattr(args, "rsi_period", None)
        level = getattr(args, "rsi_level", None)
        sign = getattr(args, "rsi_sign", None)  # above/below
        indicator_config["rsi"] = [True, sign, level, period]
    else:
        indicator_config["rsi"] = [False, None, None, None]

    return StrategyParams(
        sl=float(getattr(args, "sl", 3.0)),
        tp=float(getattr(args, "tp", 3.0)),
        delay_open=int(getattr(args, "delay_open", 0)),
        holding_minutes=int(getattr(args, "holding_minutes", 60)),
        indicator_config=indicator_config,
        commission=float(getattr(args, "commission", 0.02)),
        slippage=float(getattr(args, "slippage", 0.0004)),
        bar_minutes=int(getattr(args, "bar_minutes", 15)),
    )

def build_optuna_params(trial, args: Any) -> StrategyParams:
    """Сбор параметров для Optuna."""
    indicator_config = _copy_default_indicator_config()

    sl = trial.suggest_float("sl", args.sl_min, args.sl_max, step=args.sl_step)
    tp = trial.suggest_float("tp", args.tp_min, args.tp_max, step=args.tp_step)
    delay_open = trial.suggest_int("delay_open", args.delay_open_min, args.delay_open_max, step=args.delay_open_step)
    holding_minutes = trial.suggest_int("holding_minutes", args.holding_minutes_min, args.holding_minutes_max, step=args.holding_minutes_step)

    # --- EMA ---
    use_ema = False
    # use_ema = trial.suggest_categorical("ema_enabled", [False, True])
    if use_ema:
        ema_sign = trial.suggest_categorical("ema_sign", ["above", "below"])
        ema_fast = trial.suggest_int("ema_fast", 10, 30, step=5)
        ema_slow = trial.suggest_int("ema_slow", 40, 120, step=5)
        if ema_fast >= ema_slow:
            ema_fast = max(5, min(ema_fast, ema_slow - 1))
        indicator_config["ema"] = [True, ema_sign, ema_fast, ema_slow]
    else:
        indicator_config["ema"] = [False, None, None, None]

    # --- RSI ---
    use_rsi = False
    use_rsi = trial.suggest_categorical("rsi_enabled", [False, True])
    if use_rsi:
        rsi_sign = trial.suggest_categorical("rsi_sign", ["above", "below"])
        rsi_period = trial.suggest_int("rsi_period", 12, 21, step=3)
        rsi_level = trial.suggest_int("rsi_level", 20, 80, step=10)
        indicator_config["rsi"] = [True, rsi_sign, rsi_level, rsi_period]
    else:
        indicator_config["rsi"] = [False, None, None, None]

    return StrategyParams(
        sl=sl,
        tp=tp,
        delay_open=delay_open,
        holding_minutes=holding_minutes,
        indicator_config=indicator_config,
        commission=float(getattr(args, "commission", 0.02)),
        slippage=float(getattr(args, "slippage", 0.0004)),
        bar_minutes=int(getattr(args, "bar_minutes", 15)),
    )
