from __future__ import annotations

from hypertrade.config.strategy import StrategyParams, default_indicator_config
from hypertrade.config.schemas import FilterSearchSpace


def build_strategy_params(trial, search: FilterSearchSpace) -> StrategyParams:
    indicator_config = default_indicator_config()

    atr_use = bool(search.atr_use)
    sl = tp = atr_period = atr_sl = atr_tp = None
    if not atr_use:
        sl = trial.suggest_float("sl", search.sl_min, search.sl_max, step=search.sl_step)
        tp = trial.suggest_float("tp", search.tp_min, search.tp_max, step=search.tp_step)
    else:
        atr_period = trial.suggest_int(
            "atr_period",
            search.atr_period_min,
            search.atr_period_max,
            step=search.atr_period_step,
        )
        atr_sl = trial.suggest_float("atr_sl", search.atr_sl_min, search.atr_sl_max, step=search.atr_sl_step)
        atr_tp = trial.suggest_float("atr_tp", search.atr_tp_min, search.atr_tp_max, step=search.atr_tp_step)
        indicator_config["atr"] = {"enabled": True, "period": int(atr_period)}

    donchian_enabled = False
    if search.donchian_use:
        donchian_enabled = trial.suggest_categorical("donchian_enabled", [False, True])
        if donchian_enabled:
            donchian_period = trial.suggest_categorical("donchian_period", [2000, 5000])
            indicator_config["donchian"] = {"enabled": True, "period": int(donchian_period)}

    psar_enabled = False
    psar_step = psar_max = None
    if search.psar_use:
        psar_enabled = trial.suggest_categorical("psar_enabled", [False, True])
        if psar_enabled:
            psar_step = trial.suggest_float("psar_step", 0.001, 0.01, step=0.001)
            psar_max = trial.suggest_float("psar_max", 0.05, 0.5, step=0.05)

    ts_enabled = False
    ts_dist = ts_step = None
    if search.ts_use:
        ts_enabled = trial.suggest_categorical("ts_enabled", [False, True])
        if ts_enabled:
            ts_step = 0.5
            ts_dist = trial.suggest_float("ts_dist", 0.5, 5.0, step=ts_step)

    delay_open = trial.suggest_int(
        "delay_open",
        search.delay_open_min,
        search.delay_open_max,
        step=search.delay_open_step,
    )
    holding_minutes = trial.suggest_int(
        "holding_minutes",
        search.holding_minutes_min,
        search.holding_minutes_max,
        step=search.holding_minutes_step,
    )
    confirm_bars = trial.suggest_int(
        "confirm_bars",
        search.confirm_bars_min,
        search.confirm_bars_max,
        step=search.confirm_bars_step,
    )
    ranking_mode = trial.suggest_categorical("ranking_mode", search.ranking_mode_choices)
    accept_start_minute = accept_end_minute = None
    if search.time_gate_use:
        accept_start_minute = trial.suggest_int("accept_start_minute", search.accept_start_min, search.accept_start_max, step=search.accept_start_step)
        accept_end_minute = trial.suggest_int("accept_end_minute", search.accept_end_min, search.accept_end_max, step=search.accept_end_step)
        if accept_end_minute <= accept_start_minute:
            accept_end_minute = accept_start_minute + max(1, search.accept_end_step)
    cooldown_minutes = 0
    if search.cooldown_use:
        cooldown_minutes = trial.suggest_int("cooldown_minutes", search.cooldown_min, search.cooldown_max, step=search.cooldown_step)
    max_signals_per_side = trial.suggest_int("max_signals_per_side", search.max_signals_per_side_min, search.max_signals_per_side_max)

    if search.ema_use:
        ema_enabled = trial.suggest_categorical("ema_enabled", [False, True])
        if ema_enabled:
            ema_sign = trial.suggest_categorical("ema_sign", ["above", "below"])
            ema_fast = trial.suggest_int("ema_fast", 10, 30, step=5)
            ema_slow = trial.suggest_int("ema_slow", 40, 120, step=5)
            if ema_fast >= ema_slow:
                ema_fast = max(5, ema_slow - 1)
            indicator_config["ema"] = {"enabled": True, "sign": ema_sign, "fast": int(ema_fast), "slow": int(ema_slow)}

    if search.rsi_use:
        rsi_enabled = trial.suggest_categorical("rsi_enabled", [False, True])
        if rsi_enabled:
            rsi_sign = trial.suggest_categorical("rsi_sign", ["above", "below"])
            rsi_period = trial.suggest_int("rsi_period", 12, 21, step=3)
            rsi_level = trial.suggest_int("rsi_level", 20, 80, step=10)
            indicator_config["rsi"] = {"enabled": True, "sign": rsi_sign, "level": int(rsi_level), "period": int(rsi_period)}

    if search.adx_use:
        adx_enabled = trial.suggest_categorical("adx_enabled", [False, True])
        if adx_enabled:
            adx_sign = trial.suggest_categorical("adx_sign", ["trend", "range"])
            adx_min = trial.suggest_float("adx_min", 10.0, 30.0, step=2.0)
            adx_period = trial.suggest_int("adx_period", 10, 28, step=2)
            indicator_config["adx"] = {"enabled": True, "mode": adx_sign, "minimum": float(adx_min), "period": int(adx_period)}

    if search.macd_use:
        macd_enabled = trial.suggest_categorical("macd_enabled", [False, True])
        if macd_enabled:
            macd_sign = trial.suggest_categorical("macd_sign", ["above", "below"])
            macd_fast = trial.suggest_int("macd_fast", 8, 20, step=2)
            macd_slow = trial.suggest_int("macd_slow", 18, 40, step=2)
            macd_signal = trial.suggest_int("macd_signal", 5, 15, step=1)
            if macd_fast >= macd_slow:
                macd_fast = max(2, macd_slow - 1)
            indicator_config["macd"] = {
                "enabled": True,
                "sign": macd_sign,
                "fast": int(macd_fast),
                "slow": int(macd_slow),
                "signal": int(macd_signal),
            }

    if search.bb_use:
        bb_enabled = trial.suggest_categorical("bb_enabled", [False, True])
        if bb_enabled:
            bb_sign = trial.suggest_categorical("bb_sign", ["above", "below"])
            bb_period = trial.suggest_int("bb_period", 10, 40, step=5)
            bb_std = trial.suggest_float("bb_std", 1.5, 3.0, step=0.5)
            indicator_config["bb"] = {"enabled": True, "sign": bb_sign, "period": int(bb_period), "std": float(bb_std)}

    if search.vwap_use:
        vwap_enabled = trial.suggest_categorical("vwap_enabled", [False, True])
        if vwap_enabled:
            vwap_sign = trial.suggest_categorical("vwap_sign", ["above", "below"])
            vwap_k = trial.suggest_float("vwap_k", 0.8, 1.20, step=0.05)
            indicator_config["vwap"] = {"enabled": True, "sign": vwap_sign, "threshold": round(float(vwap_k), 4)}

    if search.volume_use:
        volume_enabled = trial.suggest_categorical("volume_enabled", [False, True])
        if volume_enabled:
            volume_sign = trial.suggest_categorical("volume_sign", ["above", "below"])
            vol_ma = trial.suggest_int("vol_ma_period", 10, 60, step=10)
            vol_k = trial.suggest_float("vol_k", 0.5, 2.0, step=0.25)
            indicator_config["volume"] = {
                "enabled": True,
                "sign": volume_sign,
                "ma_period": int(vol_ma),
                "threshold": round(float(vol_k), 4),
            }

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
        confirm_bars=confirm_bars,
        accept_start_minute=accept_start_minute,
        accept_end_minute=accept_end_minute,
        cooldown_minutes=cooldown_minutes,
        max_signals_per_side=max_signals_per_side,
        ranking_mode=ranking_mode,
        indicator_config=indicator_config,
        commission=search.commission,
        slippage=search.slippage,
        bar_minutes=search.bar_minutes,
    )
