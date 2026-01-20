import pandas as pd
import numpy as np
import ta


def calculate_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    indicators = pd.DataFrame({"datetime": df["datetime"]})

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    # ===== EMA =====
    ema_cfg = indicator_config.get("ema")
    if ema_cfg and len(ema_cfg) >= 4 and ema_cfg[0]:
        _, _, fast, slow = ema_cfg
        if fast is None or slow is None:
            raise ValueError("EMA включен, но fast/slow не заданы")

        fast = int(fast)
        slow = int(slow)
        # считаем обе EMA, храним отдельно
        indicators[f"ema_{fast}"] = close.ewm(span=fast, adjust=False).mean()
        if slow != fast:
            indicators[f"ema_{slow}"] = close.ewm(span=slow, adjust=False).mean()

    # ===== RSI =====
    rsi_cfg = indicator_config.get("rsi")
    if rsi_cfg and len(rsi_cfg) >= 4 and rsi_cfg[0]:
        _, _, _, period = rsi_cfg
        if period is None:
            raise ValueError("RSI включен, но period не задан")
        indicators[f"rsi_{int(period)}"] = ta.momentum.RSIIndicator(close, int(period)).rsi()

    # # ===== ATR =====
    atr_cfg = indicator_config.get("atr")
    if atr_cfg and len(atr_cfg) >= 2 and atr_cfg[0]:
        _, period = atr_cfg[:2]
        if period is None:
            raise ValueError("ATR включен, но period не задан")
        
        period = int(period)

        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # ATR из ta (average_true_range)
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=period).average_true_range()
        indicators[f"atr_{period}"] = atr.shift(1)

    # ===== Donchian =====
    don_cfg = indicator_config.get("donchian")
    if don_cfg and len(don_cfg) >= 2 and don_cfg[0]:
        _, period = don_cfg[:2]
        period = int(period)

        high = df["high"].astype(float)
        low = df["low"].astype(float)

        # исключаем текущий бар (shift(1)) чтобы не было lookahead
        indicators[f"don_h_{period}"] = high.rolling(period).max().shift(1)
        indicators[f"don_l_{period}"] = low.rolling(period).min().shift(1)

    # # ===== ADX (+DI/-DI) =====
    # adx_cfg = indicator_config.get("adx")
    # if adx_cfg and len(adx_cfg) >= 4 and adx_cfg[0]:
    #     _, _, _, period = adx_cfg[:4]
    #     if period is None:
    #         raise ValueError("ADX включен, но period не задан")
    #     period = int(period)
    #     high = df["high"].astype(float)
    #     low = df["low"].astype(float)

    #     adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=period)
    #     indicators[f"adx_{period}"] = adx.adx()
    #     indicators[f"di_plus_{period}"] = adx.adx_pos()
    #     indicators[f"di_minus_{period}"] = adx.adx_neg()

    # # ===== MACD =====
    # macd_cfg = indicator_config.get("macd")
    # if macd_cfg and len(macd_cfg) >= 4 and macd_cfg[0]:
    #     _, fast, slow, signal = macd_cfg[:4]
    #     if fast is None or slow is None or signal is None:
    #         raise ValueError("MACD включен, но fast/slow/signal не заданы")
    #     fast = int(fast)
    #     slow = int(slow)
    #     signal = int(signal)
    #     m = ta.trend.MACD(close, window_fast=fast, window_slow=slow, window_sign=signal)
    #     indicators[f"macd_{fast}_{slow}_{signal}"] = m.macd()
    #     indicators[f"macds_{fast}_{slow}_{signal}"] = m.macd_signal()
    #     indicators[f"macdh_{fast}_{slow}_{signal}"] = m.macd_diff()

    # # ===== Bollinger Bands =====
    # bb_cfg = indicator_config.get("bb")
    # if bb_cfg and len(bb_cfg) >= 3 and bb_cfg[0]:
    #     _, period, n_std = bb_cfg[:3]
    #     if period is None or n_std is None:
    #         raise ValueError("BB включен, но period/n_std не заданы")
    #     period = int(period)
    #     n_std = float(n_std)
    #     bb = ta.volatility.BollingerBands(close, window=period, window_dev=n_std)
    #     indicators[f"bb_m_{period}_{n_std:g}"] = bb.bollinger_mavg()
    #     indicators[f"bb_u_{period}_{n_std:g}"] = bb.bollinger_hband()
    #     indicators[f"bb_l_{period}_{n_std:g}"] = bb.bollinger_lband()
    #     indicators[f"bb_p_{period}_{n_std:g}"] = bb.bollinger_pband()  # 0..1 положение цены в диапазоне

    # # ===== Volume MA + ratio =====
    # vol_cfg = indicator_config.get("volume")
    # if vol_cfg and len(vol_cfg) >= 4 and vol_cfg[0]:
    #     _, vol_sign, ma_period, vol_k = vol_cfg[:4]
    #     if "volume" in df.columns and ma_period is not None:
    #         ma_period = int(ma_period)
    #         v = df["volume"].astype(float)
    #         indicators[f"vol_ma_{ma_period}"] = v.rolling(ma_period).mean()
    #         denom = indicators[f"vol_ma_{ma_period}"].replace(0.0, np.nan)
    #         ratio = v / denom
    #         ratio = ratio.replace([np.inf, -np.inf], np.nan)
    #         indicators[f"vol_ratio_{ma_period}"] = ratio

    
    # # ===== VWAP (интрадей) =====
    # vwap_cfg = indicator_config.get("vwap")
    # if vwap_cfg and len(vwap_cfg) >= 1 and vwap_cfg[0]:
    #     if "volume" in df.columns:
    #         tp = (high + low + close) / 3.0
    #         vol = df["volume"].astype(float)
    #         day = pd.to_datetime(df["datetime"], utc=True).dt.floor("D")
    #         pv = (tp * vol).groupby(day).cumsum()
    #         vv = vol.groupby(day).cumsum()
    #         indicators["vwap"] = pv / vv

    # # ===== Swing highs/lows (fractals) =====
    # sr_cfg = indicator_config.get("sr")
    # if sr_cfg and len(sr_cfg) >= 2 and sr_cfg[0]:
    #     _, left = sr_cfg[:2]
    #     left = int(left) if left is not None else 2
    #     right = left

    #     # pivot high/low: максимум/минимум среди окна (2*left+1)
    #     w = left + right + 1
    #     roll_h = high.rolling(w, center=True).max()
    #     roll_l = low.rolling(w, center=True).min()
    #     indicators[f"pivot_h_{left}"] = (high == roll_h).astype(np.int8)
    #     indicators[f"pivot_l_{left}"] = (low == roll_l).astype(np.int8)

    return indicators
