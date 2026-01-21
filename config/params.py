from dataclasses import dataclass, field
from typing import Dict, List, Union, Any

# indicator_config хранит списки фиксированной длины (для совместимости с текущим кодом)
# ema: [enabled, sign, fast, slow]
# rsi: [enabled, sign, level, period]
# volume: [enabled, sign, period, k]
# adx: [enabled, sign, period, min]
# macd: [enabled, sign, fast, slow, signal]
# bb: [enabled, sign, std, period]
# vwap: [enabled, sign, k]
IndicatorValue = Union[int, float, bool, str, None]
IndicatorConfig = Dict[str, List[IndicatorValue]]

DEFAULT_INDICATOR_CONFIG: IndicatorConfig = {
    "ema": [False, None, None, None],
    "rsi": [False, None, None, None],
    "volume": [False, None, None, None],
    "adx": [False, None, None, None],
    "atr": [False, None, None, None],
    "macd": [False, None, None, None, None],
    "bb": [False, None, None, None],
    "donchian": [False, None],
    "vwap": [False, None, None],
}

def _copy_default_indicator_config() -> IndicatorConfig:
    return {k: list(v) for k, v in DEFAULT_INDICATOR_CONFIG.items()}

@dataclass
class StrategyParams:
    # --- core ---
    sl: float | None = None
    tp: float | None = None
    delay_open: int = 0
    holding_minutes: int = 600

    # --- NEW: ATR SLTP mode ---
    atr_use: bool = False
    atr_period: int = 14
    atr_sl: float = 1.0
    atr_tp: float = 1.5

    # --- PSAR trailing stop ---
    psar_enabled: bool = False
    psar_max: float | None = None
    psar_step: float | None = None

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

    atr_use = bool(getattr(args, "atr_use", False))
    atr_period = int(getattr(args, "atr_period", 14))
    atr_sl = float(getattr(args, "atr_sl", 0.5))
    atr_tp = float(getattr(args, "atr_tp", 0.5))
    if atr_use:
        indicator_config["atr"] = [True, atr_period]

    return StrategyParams(
        sl=float(getattr(args, "sl", 3.0)),
        tp=float(getattr(args, "tp", 4.0)),
        delay_open=int(getattr(args, "delay_open", 0)),
        holding_minutes=int(getattr(args, "holding_minutes", 600)),

        atr_use=atr_use,
        atr_period=atr_period,
        atr_sl=atr_sl,
        atr_tp=atr_tp,

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

    atr_use = _bool(getattr(args, "atr_use", False))
    trial.suggest_categorical("atr_use", [atr_use])
    sl = tp = atr_period = atr_sl = atr_tp = None
    if not atr_use:
        sl = trial.suggest_float("sl", round(args.sl_min, 4), round(args.sl_max, 4), step=args.sl_step)
        tp = trial.suggest_float("tp", round(args.tp_min, 4), round(args.tp_max, 4), step=args.tp_step)
    else:
        atr_sl = trial.suggest_float("atr_sl", round(float(getattr(args, "atr_sl_min", 0.5)), 4), round(float(getattr(args, "atr_sl_max", 2.0)), 4), step=round(float(getattr(args, "atr_sl_step", 0.25)), 4))
        atr_tp = trial.suggest_float("atr_tp", round(float(getattr(args, "atr_tp_min", 0.25)), 4), round(float(getattr(args, "atr_tp_max", 1.5)), 4), step=round(float(getattr(args, "atr_tp_step", 0.25)), 4))
        atr_period = trial.suggest_int("atr_period", int(getattr(args, "atr_period_min", 10)), int(getattr(args, "atr_period_max", 28)), step=int(getattr(args, "atr_period_step", 2)))
        indicator_config["atr"] = [True, int(atr_period)]

    # --- Donchian gate/use ---
    donchian_use = _bool(getattr(args, "donchian_use", False))
    if donchian_use:
        donchian_enabled = trial.suggest_categorical("donchian_enabled", [False, True])
        if donchian_enabled:
            donchian_period = trial.suggest_categorical("donchian_period", [2000, 5000])
            indicator_config["donchian"] = [True, int(donchian_period)]
    else:
        donchian_enabled = trial.suggest_categorical("donchian_enabled", [False])

    # --- PSAR gate/use ---
    psar_use = _bool(getattr(args, "psar_use", False))
    psar_max = psar_step = None
    if psar_use:
        psar_enabled = trial.suggest_categorical("psar_enabled", [False, True])
        if psar_enabled:
            psar_max = trial.suggest_float("psar_max", 0.05, 0.5, step=0.05)
            psar_step = trial.suggest_float("psar_step", 0.001, 0.01, step=0.001) 
    else:
        psar_enabled = trial.suggest_categorical("psar_enabled", [False])

    # --- TS gate/use ---
    ts_use = _bool(getattr(args, "ts_use", False))
    ts_dist = ts_step = None
    if ts_use:
        ts_enabled = trial.suggest_categorical("ts_enabled", [False, True])
        if ts_enabled:
            ts_step = round(float(getattr(args, "ts_step", 0.5)), 4)
            ts_dist = trial.suggest_float("ts_dist", 0.5, 5.0, step=ts_step)
    else:
        ts_enabled = trial.suggest_categorical("ts_enabled", [False])

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
        ema_enabled = trial.suggest_categorical("ema_enabled", [False])

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
        rsi_enabled = trial.suggest_categorical("rsi_enabled", [False])

    # --- ADX ---
    adx_use = _bool(getattr(args, "adx_use", False))
    if adx_use:
        adx_enabled = trial.suggest_categorical("adx_enabled", [False, True])
        if adx_enabled:
            adx_sign = trial.suggest_categorical("adx_sign", ["trend", "range"])
            adx_min = trial.suggest_float("adx_min", 10.0, 30.0, step=2.0)
            adx_period = trial.suggest_int("adx_period", 10, 28, step=2)
            indicator_config["adx"] = [True, adx_sign, float(adx_min), int(adx_period)]
    else:
        adx_enabled = trial.suggest_categorical("adx_enabled", [False])

    # --- MACD ---
    macd_use = _bool(getattr(args, "macd_use", False))
    if macd_use:
        macd_enabled = trial.suggest_categorical("macd_enabled", [False, True])
        if macd_enabled:
            macd_sign = trial.suggest_categorical("macd_sign", ["above", "below"])
            macd_fast = trial.suggest_int("macd_fast", 8, 20, step=2)
            macd_slow = trial.suggest_int("macd_slow", 18, 40, step=2)
            macd_signal = trial.suggest_int("macd_signal", 5, 15, step=1)
            if macd_fast >= macd_slow:
                macd_fast = max(2, min(int(macd_fast), int(macd_slow) - 1))
            indicator_config["macd"] = [True, macd_sign, int(macd_fast), int(macd_slow), int(macd_signal)]
    else:
        macd_enabled = trial.suggest_categorical("macd_enabled", [False])

    # --- Bollinger Bands ---
    bb_use = _bool(getattr(args, "bb_use", False))
    if bb_use:
        bb_enabled = trial.suggest_categorical("bb_enabled", [False, True])
        if bb_enabled:
            bb_sign = trial.suggest_categorical("bb_sign", ["above", "below"])
            bb_std = trial.suggest_float("bb_std", 1.5, 3.0, step=0.5)
            bb_period = trial.suggest_int("bb_period", 10, 40, step=5)
            indicator_config["bb"] = [True, bb_sign, float(bb_std), int(bb_period)]
    else:
        bb_enabled = trial.suggest_categorical("bb_enabled", [False])

    # --- VWAP ---
    vwap_use = _bool(getattr(args, "vwap_use", False))
    if vwap_use:
        vwap_enabled = trial.suggest_categorical("vwap_enabled", [False, True])
        if vwap_enabled:
            vwap_sign = trial.suggest_categorical("vwap_sign", ["above", "below"])
            vwap_k = trial.suggest_float("vwap_k", 0.8, 1.20, step=0.05)
            indicator_config["vwap"] = [True, vwap_sign, round(float(vwap_k), 4)]
    else:
        vwap_enabled = trial.suggest_categorical("vwap_enabled", [False])

    # --- Volume filters ---
    volume_use = _bool(getattr(args, "volume_use", False))
    if volume_use:
        volume_enabled = trial.suggest_categorical("volume_enabled", [False, True])
        if volume_enabled:
            vol_sign = trial.suggest_categorical("volume_sign", ["above", "below"])
            vol_ma = trial.suggest_int("vol_ma_period", 10, 60, step=10)
            vol_k = trial.suggest_float("vol_k", 0.5, 2.0, step=0.25)
            indicator_config["volume"] = [True, vol_sign, int(vol_ma), round(float(vol_k), 4)]
    else:
        volume_enabled = trial.suggest_categorical("volume_enabled", [False])

    return StrategyParams(
        sl=sl,
        tp=tp,
        delay_open=delay_open,
        holding_minutes=holding_minutes,

        atr_use=atr_use,
        atr_period=atr_period,
        atr_sl=atr_sl,
        atr_tp=atr_tp,

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
