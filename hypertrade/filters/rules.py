from __future__ import annotations
import numpy as np
import pandas as pd
from hypertrade.config.strategy import indicator_enabled, indicator_settings
LONG = 1
SHORT = -1

def _get_confirm_slice(ohlc: pd.DataFrame, entry_idx: int, confirm_bars: int) -> pd.DataFrame:
    if confirm_bars is None or int(confirm_bars) <= 1:
        return ohlc.iloc[entry_idx : entry_idx + 1]
    cb = int(confirm_bars)
    start = max(0, entry_idx - cb + 1)
    return ohlc.iloc[start : entry_idx + 1]


def filters(ohlc: pd.DataFrame, entry_idx: int, params, direction: int) -> bool:
    cfg = params.indicator_config
    win = _get_confirm_slice(ohlc, entry_idx, int(getattr(params, "confirm_bars", 1)))

    ema_cfg = indicator_settings(cfg, "ema")
    if indicator_enabled(cfg, "ema"):
        sign = ema_cfg.get("sign")
        fast = ema_cfg.get("fast")
        slow = ema_cfg.get("slow")
        col_fast = f"ema_{int(fast)}"
        col_slow = f"ema_{int(slow)}"
        if col_fast not in win.columns or col_slow not in win.columns:
            return False
        if win[col_fast].isna().any() or win[col_slow].isna().any():
            return False
        ok = (win[col_fast] > win[col_slow]).all() if sign == "above" else (win[col_fast] < win[col_slow]).all()
        if not ok:
            return False

    rsi_cfg = indicator_settings(cfg, "rsi")
    if indicator_enabled(cfg, "rsi"):
        sign = rsi_cfg.get("sign")
        level = rsi_cfg.get("level")
        period = rsi_cfg.get("period")
        col = f"rsi_{int(period)}"
        if col not in win.columns or win[col].isna().any():
            return False
        if sign == "above" and not (win[col] > float(level)).all():
            return False
        if sign != "above" and not (win[col] < float(level)).all():
            return False

    adx_cfg = indicator_settings(cfg, "adx")
    if indicator_enabled(cfg, "adx"):
        adx_sign = adx_cfg.get("mode")
        adx_min = adx_cfg.get("minimum")
        period = int(adx_cfg.get("period"))
        cols = [f"adx_{period}", f"di_plus_{period}", f"di_minus_{period}"]
        for col in cols:
            if col not in win.columns or win[col].isna().any():
                return False
        adx_min = float(adx_min) if adx_min is not None else 0.0
        if adx_sign == "trend":
            if not (win[f"adx_{period}"] >= adx_min).all():
                return False
        elif adx_sign == "range":
            if not (win[f"adx_{period}"] <= adx_min).all():
                return False
        else:
            return False
        if direction == LONG and not (win[f"di_plus_{period}"] > win[f"di_minus_{period}"]).all():
            return False
        if direction == SHORT and not (win[f"di_minus_{period}"] > win[f"di_plus_{period}"]).all():
            return False

    macd_cfg = indicator_settings(cfg, "macd")
    if indicator_enabled(cfg, "macd"):
        sign = macd_cfg.get("sign")
        fast = macd_cfg.get("fast")
        slow = macd_cfg.get("slow")
        sig = macd_cfg.get("signal")
        col_m = f"macd_{int(fast)}_{int(slow)}_{int(sig)}"
        col_s = f"macds_{int(fast)}_{int(slow)}_{int(sig)}"
        if col_m not in win.columns or col_s not in win.columns:
            return False
        if win[col_m].isna().any() or win[col_s].isna().any():
            return False
        if direction == LONG:
            ok = (win[col_m] > win[col_s]).all() if sign == "above" else (win[col_m] < win[col_s]).all()
        else:
            ok = (win[col_m] < win[col_s]).all() if sign == "above" else (win[col_m] > win[col_s]).all()
        if not ok:
            return False

    bb_cfg = indicator_settings(cfg, "bb")
    if indicator_enabled(cfg, "bb"):
        period = bb_cfg.get("period")
        n_std = bb_cfg.get("std")
        col_p = f"bb_p_{int(period)}_{float(n_std):g}"
        if col_p not in win.columns or win[col_p].isna().any():
            return False
        lower, upper = float(bb_cfg['lower']), float(bb_cfg['upper'])
        values = win[col_p]
        mode = bb_cfg['mode']
        if mode == 'avoid_extremes':
            ok = ((values >= lower) & (values <= upper)).all()
        elif mode == 'mean_reversion':
            ok = (values <= lower).all() if direction == LONG else (values >= upper).all()
        elif mode == 'trend_continuation':
            ok = (values >= upper).all() if direction == LONG else (values <= lower).all()
        else:
            raise ValueError(f'Unknown Bollinger mode: {mode}')
        if not ok:
            return False

    vwap_cfg = indicator_settings(cfg, "vwap")
    if indicator_enabled(cfg, "vwap"):
        vwap_sign = vwap_cfg.get("sign")
        vwap_k = vwap_cfg.get("threshold")
        if "vwap" not in win.columns or win["vwap"].isna().any():
            return False
        ratio = (win["close"].astype(float) / win["vwap"].astype(float).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        if ratio.isna().any():
            return False
        vwap_k = float(vwap_k) if vwap_k is not None else 1.0
        if direction == LONG:
            if vwap_sign == "above" and not (ratio >= vwap_k).all():
                return False
            if vwap_sign != "above" and not (ratio <= vwap_k).all():
                return False
        else:
            if vwap_sign == "below" and not (ratio <= vwap_k).all():
                return False
            if vwap_sign != "below" and not (ratio >= vwap_k).all():
                return False

    vol_cfg = indicator_settings(cfg, "volume")
    if indicator_enabled(cfg, "volume"):
        vol_sign = vol_cfg.get("sign")
        ma_period = vol_cfg.get("ma_period")
        vol_k = vol_cfg.get("threshold")
        col = f"vol_ratio_{int(ma_period)}"
        if col not in win.columns or win[col].isna().any():
            return False
        if vol_sign == "above" and not (win[col] >= float(vol_k)).all():
            return False
        if vol_sign != "above" and not (win[col] <= float(vol_k)).all():
            return False

    atr_cfg = indicator_settings(cfg, "atr")
    if indicator_enabled(cfg, "atr"):
        period = atr_cfg.get("period")
        col = f"atr_{int(period)}"
        if col not in win.columns or win[col].isna().any():
            return False
        atr_pct = (win[col] / win["close"]).astype(float)
        if not ((atr_pct >= float(getattr(params, "atr_min_pct", 0.0))) & (atr_pct <= float(getattr(params, "atr_max_pct", 1e9)))).all():
            return False

    don_cfg = indicator_settings(cfg, "donchian")
    if indicator_enabled(cfg, "donchian"):
        period = don_cfg.get("period")
        hcol = f"don_h_{int(period)}"
        lcol = f"don_l_{int(period)}"
        if hcol not in win.columns or lcol not in win.columns:
            return False
        if win[hcol].isna().any() or win[lcol].isna().any():
            return False
        pad = 0.002
        last = win.iloc[-1]
        if direction == LONG and last["close"] >= last[hcol] * (1.0 - pad):
            return False
        if direction == SHORT and last["close"] <= last[lcol] * (1.0 + pad):
            return False

    return True


