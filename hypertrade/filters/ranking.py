from __future__ import annotations
import pandas as pd
from hypertrade.config.strategy import indicator_enabled, indicator_settings
LONG = 1

def rank_score(ohlc: pd.DataFrame, entry_idx: int, params, direction: int) -> float:
    ranking_mode = getattr(params, "ranking_mode", "none")
    if ranking_mode == "none":
        return 0.0

    score = 0.0
    row = ohlc.iloc[entry_idx]
    cfg = params.indicator_config

    ema_cfg = indicator_settings(cfg, "ema")
    if indicator_enabled(cfg, "ema"):
        fast = ema_cfg.get("fast")
        slow = ema_cfg.get("slow")
        fast_col = f"ema_{int(fast)}"
        slow_col = f"ema_{int(slow)}"
        if fast_col in ohlc.columns and slow_col in ohlc.columns:
            diff = float(row[fast_col] - row[slow_col])
            score += diff if direction == LONG else -diff

    rsi_cfg = indicator_settings(cfg, "rsi")
    if indicator_enabled(cfg, "rsi"):
        level = rsi_cfg.get("level")
        period = rsi_cfg.get("period")
        col = f"rsi_{int(period)}"
        if col in ohlc.columns:
            score += abs(float(row[col]) - float(level)) * 0.1

    adx_cfg = indicator_settings(cfg, "adx")
    if indicator_enabled(cfg, "adx"):
        period = adx_cfg.get("period")
        col = f"adx_{int(period)}"
        if col in ohlc.columns:
            score += float(row[col]) * 0.05

    vol_cfg = indicator_settings(cfg, "volume")
    if indicator_enabled(cfg, "volume"):
        ma_period = vol_cfg.get("ma_period")
        col = f"vol_ratio_{int(ma_period)}"
        if col in ohlc.columns and pd.notna(row[col]):
            score += float(row[col]) * 0.2

    return score


