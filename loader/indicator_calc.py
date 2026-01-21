import pandas as pd
import numpy as np
import ta


def calculate_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    indicators = pd.DataFrame({"datetime": df["datetime"]})

    if not isinstance(indicator_config, dict):
        return indicators

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # ===== EMA =====
    ema_periods = set()

    ema_cfg = indicator_config.get("ema")
    if ema_cfg and len(ema_cfg) >= 4 and bool(ema_cfg[0]):
        _, _, fast, slow = ema_cfg[:4]
        if fast is not None:
            ema_periods.add(int(fast))
        if slow is not None:
            ema_periods.add(int(slow))

    macd_cfg = indicator_config.get("macd")
    if macd_cfg and len(macd_cfg) >= 5 and bool(macd_cfg[0]):
        _, _, fast, slow, _ = macd_cfg[:5]
        if fast is None or slow is None:
            raise ValueError("MACD включен, но fast/slow не заданы")
        ema_periods.add(int(fast))
        ema_periods.add(int(slow))

    for p in sorted(ema_periods):
        indicators[f"ema_{p}"] = close.ewm(span=int(p), adjust=False).mean()

    # ===== RSI =====
    rsi_cfg = indicator_config.get("rsi")
    if rsi_cfg and len(rsi_cfg) >= 4 and bool(rsi_cfg[0]):
        _, _, _, period = rsi_cfg[:4]
        if period is None:
            raise ValueError("RSI включен, но period не задан")
        indicators[f"rsi_{int(period)}"] = ta.momentum.RSIIndicator(close, int(period)).rsi()

    # ===== ATR =====
    atr_cfg = indicator_config.get("atr")
    if atr_cfg and len(atr_cfg) >= 2 and bool(atr_cfg[0]):
        _, period = atr_cfg[:2]
        if period is None:
            raise ValueError("ATR включен, но period не задан")

        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=int(period)).average_true_range()
        indicators[f"atr_{int(period)}"] = atr.shift(1)

    # ===== Donchian =====
    don_cfg = indicator_config.get("donchian")
    if don_cfg and len(don_cfg) >= 2 and bool(don_cfg[0]):
        _, period = don_cfg[:2]
        if period is None:
            raise ValueError("Donchian включен, но period не задан")

        indicators[f"don_h_{int(period)}"] = high.rolling(int(period)).max().shift(1)
        indicators[f"don_l_{int(period)}"] = low.rolling(int(period)).min().shift(1)

    # ===== ADX (+DI/-DI) =====
    adx_cfg = indicator_config.get("adx")
    if adx_cfg and len(adx_cfg) >= 4 and bool(adx_cfg[0]):
        _, _, _, period = adx_cfg[:4]
        if period is None:
            raise ValueError("ADX включен, но period не задан")

        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=int(period))
        indicators[f"adx_{int(period)}"] = adx.adx()
        indicators[f"di_plus_{int(period)}"] = adx.adx_pos()
        indicators[f"di_minus_{int(period)}"] = adx.adx_neg()

    # ===== Bollinger Bands =====
    bb_cfg = indicator_config.get("bb")
    if bb_cfg and len(bb_cfg) >= 3 and bool(bb_cfg[0]):
        _, period, n_std = bb_cfg[:3]
        if period is None or n_std is None:
            raise ValueError("BB включен, но period/n_std не заданы")
        d = float(n_std)
        k = f"{d:g}"

        bb = ta.volatility.BollingerBands(close, window=p, window_dev=d)
        indicators[f"bb_m_{int(period)}_{k}"] = bb.bollinger_mavg()
        indicators[f"bb_u_{int(period)}_{k}"] = bb.bollinger_hband()
        indicators[f"bb_l_{int(period)}_{k}"] = bb.bollinger_lband()

    # ===== Volume MA + ratio =====
    vol_cfg = indicator_config.get("volume")
    if vol_cfg and len(vol_cfg) >= 4 and bool(vol_cfg[0]):
        _, _, ma_period, _ = vol_cfg[:4]
        if "volume" in df.columns and ma_period is not None:
            v = df["volume"].astype(float)
            indicators[f"vol_ma_{int(ma_period)}"] = v.rolling(int(ma_period)).mean()

    # ===== VWAP (интрадей) =====
    vwap_cfg = indicator_config.get("vwap")
    if vwap_cfg and len(vwap_cfg) >= 3 and bool(vwap_cfg[0]):
        if "volume" in df.columns:
            tp = (high + low + close) / 3.0
            vol = df["volume"].astype(float)
            day = pd.to_datetime(df["datetime"], utc=True).dt.floor("D")
            pv = (tp * vol).groupby(day).cumsum()
            vv = vol.groupby(day).cumsum()
            denom = vv.replace(0.0, np.nan)
            indicators["vwap"] = (pv / denom).replace([np.inf, -np.inf], np.nan)

    return indicators
