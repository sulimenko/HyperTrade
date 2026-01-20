import pandas as pd

LONG = 1
SHORT = -1

def _get_confirm_slice(ohlc: pd.DataFrame, entry_idx: int, confirm_bars: int) -> pd.DataFrame:
    if confirm_bars is None or int(confirm_bars) <= 1:
        return ohlc.iloc[entry_idx:entry_idx+1]
    cb = int(confirm_bars)
    a = entry_idx - cb + 1
    if a < 0:
        a = 0
    return ohlc.iloc[a:entry_idx+1]

def filters(ohlc: pd.DataFrame, entry_idx: int, params, direction: int) -> bool:
    cfg = params.indicator_config
    confirm_bars = int(getattr(params, "confirm_bars", 1))
    win = _get_confirm_slice(ohlc, entry_idx, confirm_bars)

    # ---------- EMA cross-like ----------
    ema_cfg = cfg.get("ema")
    if ema_cfg and ema_cfg[0]:
        _, sign, fast, slow = ema_cfg
        fast = int(fast)
        slow = int(slow)
        col_fast = f"ema_{fast}"
        col_slow = f"ema_{slow}"

        if col_fast not in win.columns or col_slow not in win.columns:
            return False
        if win[col_fast].isna().any() or win[col_slow].isna().any():
            return False

        if sign == "above":
            ok = (win[col_fast] > win[col_slow]).all()
        else:
            ok = (win[col_fast] < win[col_slow]).all()
        if not ok:
            return False

    # ---------- RSI threshold ----------
    rsi_cfg = cfg.get("rsi")
    if rsi_cfg and len(rsi_cfg) >= 4 and rsi_cfg[0]:
        _, sign, level, period = rsi_cfg
        period = int(period)
        col = f"rsi_{period}"
        if col not in win.columns or win[col].isna().any():
            return False

        if sign == "above":
            if not (win[col] > float(level)).all():
                return False
        else:
            if not (win[col] < float(level)).all():
                return False

    # # ---------- ADX + DI direction ----------
    # adx_cfg = cfg.get("adx")
    # if adx_cfg and len(adx_cfg) >= 4 and adx_cfg[0]:
    #     # adx: [enabled, sign, min_adx, period]
    #     _, adx_sign, adx_min, period = adx_cfg[:4]
    #     period = int(period)
    #     col_adx = f"adx_{period}"
    #     col_p = f"di_plus_{period}"
    #     col_m = f"di_minus_{period}"

    #     for c in (col_adx, col_p, col_m):
    #         if c not in win.columns or win[c].isna().any():
    #             return False

    #     adx_min = float(adx_min) if adx_min is not None else 0.0

    #     # sign управляет режимом фильтра по "силe тренда"
    #     # trend => ADX >= min; range => ADX <= min
    #     if adx_sign == "trend":
    #         if not (win[col_adx] >= adx_min).all():
    #             return False
    #     elif adx_sign == "range":
    #         if not (win[col_adx] <= adx_min).all():
    #             return False
    #     else:
    #         # неизвестный режим => лучше отклонить, чтобы не пропускать мусор
    #         return False
        
    #     # направление через DI всегда (если хочешь, можно сделать опциональным флагом)
    #     if direction == LONG:
    #         if not (win[col_p] > win[col_m]).all():
    #             return False
    #     else:
    #         if not (win[col_m] > win[col_p]).all():
    #             return False
            
    # # ---------- MACD confirmation ----------
    # macd_cfg = cfg.get("macd")
    # if macd_cfg and len(macd_cfg) >= 4 and macd_cfg[0]:
    #     _, fast, slow, sig = macd_cfg[:4]
    #     fast = int(fast)
    #     slow = int(slow)
    #     sig = int(sig)

    #     col_m = f"macd_{fast}_{slow}_{sig}"
    #     col_s = f"macds_{fast}_{slow}_{sig}"

    #     if col_m not in win.columns or col_s not in win.columns:
    #         return False
    #     if win[col_m].isna().any() or win[col_s].isna().any():
    #         return False

    #     if direction == LONG:
    #         ok = (win[col_m] > win[col_s]).all()
    #     else:
    #         ok = (win[col_m] < win[col_s]).all()
    #     if not ok:
    #         return False

    # # ---------- Bollinger: не входить у края против тренда ----------
    # bb_cfg = cfg.get("bb")
    # if bb_cfg and len(bb_cfg) >= 3 and bb_cfg[0]:
    #     _, period, n_std = bb_cfg[:3]
    #     period = int(period)
    #     n_std = float(n_std)
    #     col_p = f"bb_p_{period}_{n_std:g}"
    #     if col_p not in win.columns or win[col_p].isna().any():
    #         return False

    #     # пример: long — не покупать когда уже слишком высоко в диапазоне
    #     bb_max_p = float(getattr(params, "bb_max_p", 0.95))
    #     bb_min_p = float(getattr(params, "bb_min_p", 0.05))
    #     if direction == LONG:
    #         if not (win[col_p] < bb_max_p).all():
    #             return False
    #     else:
    #         if not (win[col_p] > bb_min_p).all():
    #             return False

    # # ---------- VWAP filter ----------
    # vwap_cfg = cfg.get("vwap")
    # if vwap_cfg and len(vwap_cfg) >= 1 and vwap_cfg[0]:
    #     if "vwap" not in win.columns or win["vwap"].isna().any():
    #         return False
    #     if direction == LONG:
    #         if not (win["close"] >= win["vwap"]).all():
    #             return False
    #     else:
    #         if not (win["close"] <= win["vwap"]).all():
    #             return False

    # ---------- Volume filter ----------
    vol_cfg = cfg.get("volume")
    if vol_cfg and len(vol_cfg) >= 4 and vol_cfg[0]:
        _, vol_sign, ma_period, vol_k = vol_cfg[:4]
        ma_period = int(ma_period)
        col = f"vol_ratio_{ma_period}"
        if col not in win.columns or win[col].isna().any():
            return False
        
        vol_k = float(vol_k)
        if vol_sign == "above":
            if not (win[col] >= vol_k).all():
                return False
        else:
            if not (win[col] <= vol_k).all():
                return False

    # ---------- ATR volatility gate ----------
    atr_cfg = cfg.get("atr")
    if atr_cfg and len(atr_cfg) >= 2 and atr_cfg[0]:
        _, period = atr_cfg[:2]
        period = int(period)
        col = f"atr_{period}"
        if col not in win.columns or win[col].isna().any():
            return False

        # нормируем на цену: ATR% = atr/close
        atr_pct = (win[col] / win["close"]).astype(float)
        atr_min = float(getattr(params, "atr_min_pct", 0.0))
        atr_max = float(getattr(params, "atr_max_pct", 1e9))
        if not ((atr_pct >= atr_min) & (atr_pct <= atr_max)).all():
            return False

    # ---------- Donchian: не входить прямо в сопротивление/поддержку ----------
    don_cfg = cfg.get("donchian")
    if don_cfg and len(don_cfg) >= 2 and don_cfg[0]:
        _, period = don_cfg[:2]
        period = int(period)
        hcol = f"don_h_{period}"
        lcol = f"don_l_{period}"
        if hcol not in win.columns or lcol not in win.columns:
            return False
        if win[hcol].isna().any() or win[lcol].isna().any():
            return False

        pad = float(getattr(params, "don_pad_pct", 0.002))  # 0.2% запас
        last = win.iloc[-1]
        if direction == LONG:
            # если цена слишком близко к верхнему каналу — риск отскока/SL
            if last["close"] >= last[hcol] * (1.0 - pad):
                return False
        else:
            if last["close"] <= last[lcol] * (1.0 + pad):
                return False

    return True
