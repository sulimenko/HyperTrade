from __future__ import annotations

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
    holding_minutes: int = 60

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
    if getattr(args, "use_ema", False):
        fast = getattr(args, "ema_fast", None)
        slow = getattr(args, "ema_slow", None)
        sign = getattr(args, "ema_sign", None)  # above/below
        indicator_config["ema"] = [True, sign, fast, slow]
    else:
        indicator_config["ema"] = [False, None, None, None]

    # RSI filter
    if getattr(args, "use_rsi", False):
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

    sl = trial.suggest_float("sl", 1.0, 5.0)
    tp = trial.suggest_float("tp", 1.0, 6.0)
    delay_open = trial.suggest_int("delay_open", 0, 60, step=5)
    holding_minutes = trial.suggest_int("holding_minutes", 15, 240, step=15)

    # --- EMA ---
    use_ema = trial.suggest_categorical("use_ema", [False, True])
    if use_ema:
        ema_sign = trial.suggest_categorical("ema_sign", ["above", "below"])
        ema_fast = trial.suggest_int("ema_fast", 5, 30, step=1)
        ema_slow = trial.suggest_int("ema_slow", 20, 120, step=1)
        if ema_fast >= ema_slow:
            ema_fast = max(5, min(ema_fast, ema_slow - 1))
        indicator_config["ema"] = [True, ema_sign, ema_fast, ema_slow]
    else:
        indicator_config["ema"] = [False, None, None, None]

    # --- RSI ---
    use_rsi = trial.suggest_categorical("use_rsi", [False, True])
    if use_rsi:
        rsi_sign = trial.suggest_categorical("rsi_sign", ["above", "below"])
        rsi_period = trial.suggest_int("rsi_period", 7, 28, step=1)
        rsi_level = trial.suggest_int("rsi_level", 20, 80, step=5)
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
