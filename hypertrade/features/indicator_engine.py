from __future__ import annotations

import numpy as np
import pandas as pd
import ta

from hypertrade.config.strategy import indicator_enabled, indicator_settings
from hypertrade.data import ensure_market_history, load_indicator, save_indicator


def calculate_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    indicators = pd.DataFrame({"datetime": df["datetime"]})
    if not isinstance(indicator_config, dict):
        return indicators

    close = df["close"].astype(float)
    high = df["high"].astype(float)
    low = df["low"].astype(float)

    ema_periods = set()
    ema_cfg = indicator_settings(indicator_config, "ema")
    if indicator_enabled(indicator_config, "ema"):
        fast = ema_cfg.get("fast")
        slow = ema_cfg.get("slow")
        if fast is not None:
            ema_periods.add(int(fast))
        if slow is not None:
            ema_periods.add(int(slow))

    macd_cfg = indicator_settings(indicator_config, "macd")
    if indicator_enabled(indicator_config, "macd"):
        fast = macd_cfg.get("fast")
        slow = macd_cfg.get("slow")
        if fast is None or slow is None:
            raise ValueError("MACD enabled but fast/slow missing")
        ema_periods.add(int(fast))
        ema_periods.add(int(slow))

    for period in sorted(ema_periods):
        indicators[f"ema_{period}"] = close.ewm(span=int(period), adjust=False).mean()

    rsi_cfg = indicator_settings(indicator_config, "rsi")
    if indicator_enabled(indicator_config, "rsi"):
        period = rsi_cfg.get("period")
        if period is None:
            raise ValueError("RSI enabled but period missing")
        indicators[f"rsi_{int(period)}"] = ta.momentum.RSIIndicator(close, int(period)).rsi()

    atr_cfg = indicator_settings(indicator_config, "atr")
    if indicator_enabled(indicator_config, "atr"):
        period = atr_cfg.get("period")
        if period is None:
            raise ValueError("ATR enabled but period missing")
        atr = ta.volatility.AverageTrueRange(high=high, low=low, close=close, window=int(period)).average_true_range()
        indicators[f"atr_{int(period)}"] = atr.shift(1)

    don_cfg = indicator_settings(indicator_config, "donchian")
    if indicator_enabled(indicator_config, "donchian"):
        period = don_cfg.get("period")
        if period is None:
            raise ValueError("Donchian enabled but period missing")
        indicators[f"don_h_{int(period)}"] = high.rolling(int(period)).max().shift(1)
        indicators[f"don_l_{int(period)}"] = low.rolling(int(period)).min().shift(1)

    adx_cfg = indicator_settings(indicator_config, "adx")
    if indicator_enabled(indicator_config, "adx"):
        period = adx_cfg.get("period")
        if period is None:
            raise ValueError("ADX enabled but period missing")
        adx = ta.trend.ADXIndicator(high=high, low=low, close=close, window=int(period))
        indicators[f"adx_{int(period)}"] = adx.adx()
        indicators[f"di_plus_{int(period)}"] = adx.adx_pos()
        indicators[f"di_minus_{int(period)}"] = adx.adx_neg()

    bb_cfg = indicator_settings(indicator_config, "bb")
    if indicator_enabled(indicator_config, "bb"):
        period = bb_cfg.get("period")
        n_std = bb_cfg.get("std")
        if period is None or n_std is None:
            raise ValueError("BB enabled but period/n_std missing")
        d = float(n_std)
        k = f"{d:g}"
        bb = ta.volatility.BollingerBands(close, window=int(period), window_dev=d)
        indicators[f"bb_m_{int(period)}_{k}"] = bb.bollinger_mavg()
        indicators[f"bb_u_{int(period)}_{k}"] = bb.bollinger_hband()
        indicators[f"bb_l_{int(period)}_{k}"] = bb.bollinger_lband()

    vol_cfg = indicator_settings(indicator_config, "volume")
    if indicator_enabled(indicator_config, "volume"):
        ma_period = vol_cfg.get("ma_period")
        if "volume" in df.columns and ma_period is not None:
            volume = df["volume"].astype(float)
            indicators[f"vol_ma_{int(ma_period)}"] = volume.rolling(int(ma_period)).mean()

    if indicator_enabled(indicator_config, "vwap"):
        if "volume" in df.columns:
            typical_price = (high + low + close) / 3.0
            volume = df["volume"].astype(float)
            day = pd.to_datetime(df["datetime"], utc=True).dt.floor("D")
            pv = (typical_price * volume).groupby(day).cumsum()
            vv = volume.groupby(day).cumsum()
            denom = vv.replace(0.0, np.nan)
            indicators["vwap"] = (pv / denom).replace([np.inf, -np.inf], np.nan)

    return indicators


def _required_indicator_cols(indicator_config) -> list[str]:
    cols: list[str] = []
    if not isinstance(indicator_config, dict):
        return cols

    ema_cfg = indicator_settings(indicator_config, "ema")
    if indicator_enabled(indicator_config, "ema"):
        fast = ema_cfg.get("fast")
        slow = ema_cfg.get("slow")
        if fast is not None:
            cols.append(f"ema_{int(fast)}")
        if slow is not None and int(slow) != int(fast):
            cols.append(f"ema_{int(slow)}")

    macd_cfg = indicator_settings(indicator_config, "macd")
    if indicator_enabled(indicator_config, "macd"):
        fast = macd_cfg.get("fast")
        slow = macd_cfg.get("slow")
        if fast is not None:
            cols.append(f"ema_{int(fast)}")
        if slow is not None and int(slow) != int(fast):
            cols.append(f"ema_{int(slow)}")

    rsi_cfg = indicator_settings(indicator_config, "rsi")
    if indicator_enabled(indicator_config, "rsi"):
        period = rsi_cfg.get("period")
        if period is not None:
            cols.append(f"rsi_{int(period)}")

    atr_cfg = indicator_settings(indicator_config, "atr")
    if indicator_enabled(indicator_config, "atr"):
        period = atr_cfg.get("period")
        if period is not None:
            cols.append(f"atr_{int(period)}")

    don_cfg = indicator_settings(indicator_config, "donchian")
    if indicator_enabled(indicator_config, "donchian"):
        period = don_cfg.get("period")
        if period is not None:
            cols.append(f"don_h_{int(period)}")
            cols.append(f"don_l_{int(period)}")

    adx_cfg = indicator_settings(indicator_config, "adx")
    if indicator_enabled(indicator_config, "adx"):
        period = adx_cfg.get("period")
        if period is not None:
            cols.append(f"adx_{int(period)}")
            cols.append(f"di_plus_{int(period)}")
            cols.append(f"di_minus_{int(period)}")

    bb_cfg = indicator_settings(indicator_config, "bb")
    if indicator_enabled(indicator_config, "bb"):
        period = bb_cfg.get("period")
        n_std = bb_cfg.get("std")
        if period is not None and n_std is not None:
            k = f"{float(n_std):g}"
            cols.extend([f"bb_m_{int(period)}_{k}", f"bb_u_{int(period)}_{k}", f"bb_l_{int(period)}_{k}"])

    vol_cfg = indicator_settings(indicator_config, "volume")
    if indicator_enabled(indicator_config, "volume"):
        ma_period = vol_cfg.get("ma_period")
        if ma_period is not None:
            cols.append(f"vol_ma_{int(ma_period)}")

    if indicator_enabled(indicator_config, "vwap"):
        cols.append("vwap")

    seen = set()
    out = []
    for col in cols:
        if col not in seen:
            out.append(col)
            seen.add(col)
    return out


def _min_config_for_cols(required_cols: list[str]) -> dict:
    need = set(required_cols)
    cfg = {}

    ema_periods = []
    for col in need:
        if col.startswith("ema_"):
            try:
                ema_periods.append(int(col.split("_", 1)[1]))
            except Exception:
                pass
    if ema_periods:
        ema_periods = sorted(set(ema_periods))
        if len(ema_periods) == 1:
            cfg["ema"] = {"enabled": True, "sign": None, "fast": ema_periods[0], "slow": ema_periods[0]}
        else:
            cfg["ema"] = {"enabled": True, "sign": None, "fast": ema_periods[0], "slow": ema_periods[-1]}

    for col in need:
        if col.startswith("rsi_"):
            cfg["rsi"] = {"enabled": True, "sign": None, "level": None, "period": int(col.split("_", 1)[1])}
            break
    for col in need:
        if col.startswith("atr_"):
            cfg["atr"] = {"enabled": True, "period": int(col.split("_", 1)[1])}
            break
    for col in need:
        if col.startswith("don_h_") or col.startswith("don_l_"):
            cfg["donchian"] = {"enabled": True, "period": int(col.split("_", 2)[2])}
            break
    for col in need:
        if col.startswith("adx_") or col.startswith("di_plus_") or col.startswith("di_minus_"):
            cfg["adx"] = {"enabled": True, "mode": None, "minimum": None, "period": int(col.rsplit("_", 1)[1])}
            break
    for col in need:
        if col.startswith(("bb_m_", "bb_u_", "bb_l_")):
            parts = col.split("_")
            if len(parts) >= 4:
                cfg["bb"] = {"enabled": True, "sign": None, "period": int(parts[2]), "std": float(parts[3])}
            break
    for col in need:
        if col.startswith("vol_ma_"):
            cfg["volume"] = {"enabled": True, "sign": None, "ma_period": int(col.split("_", 2)[2]), "threshold": None}
            break
    if "vwap" in need:
        cfg["vwap"] = {"enabled": True, "sign": None, "threshold": None}
    return cfg


def _add_macd(ohlc: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    if not indicator_enabled(indicator_config, "macd"):
        return ohlc
    macd_cfg = indicator_settings(indicator_config, "macd")
    fast = macd_cfg.get("fast")
    slow = macd_cfg.get("slow")
    sig = macd_cfg.get("signal")
    if fast is None or slow is None or sig is None:
        return ohlc
    ef = f"ema_{int(fast)}"
    es = f"ema_{int(slow)}"
    if ef not in ohlc.columns or es not in ohlc.columns:
        return ohlc
    macd = ohlc[ef] - ohlc[es]
    macds = macd.ewm(span=int(sig), adjust=False).mean()
    ohlc[f"macd_{int(fast)}_{int(slow)}_{int(sig)}"] = macd
    ohlc[f"macds_{int(fast)}_{int(slow)}_{int(sig)}"] = macds
    ohlc[f"macdh_{int(fast)}_{int(slow)}_{int(sig)}"] = macd - macds
    return ohlc


def _add_bb_p(ohlc: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    if not indicator_enabled(indicator_config, "bb"):
        return ohlc
    bb_cfg = indicator_settings(indicator_config, "bb")
    period = bb_cfg.get("period")
    n_std = bb_cfg.get("std")
    if period is None or n_std is None:
        return ohlc
    k = f"{float(n_std):g}"
    m = f"bb_m_{int(period)}_{k}"
    u = f"bb_u_{int(period)}_{k}"
    l = f"bb_l_{int(period)}_{k}"
    if m not in ohlc.columns or u not in ohlc.columns or l not in ohlc.columns:
        return ohlc
    denom = (ohlc[u] - ohlc[l]).replace(0.0, np.nan)
    ohlc[f"bb_p_{int(period)}_{k}"] = ((ohlc["close"] - ohlc[l]) / denom).replace([np.inf, -np.inf], np.nan)
    return ohlc


def _add_vol_ratio(ohlc: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    if not indicator_enabled(indicator_config, "volume"):
        return ohlc
    vol_cfg = indicator_settings(indicator_config, "volume")
    ma_period = vol_cfg.get("ma_period")
    if ma_period is None:
        return ohlc
    ma_col = f"vol_ma_{int(ma_period)}"
    if "volume" not in ohlc.columns or ma_col not in ohlc.columns:
        return ohlc
    denom = ohlc[ma_col].replace(0.0, np.nan)
    ohlc[f"vol_ratio_{int(ma_period)}"] = (ohlc["volume"].astype(float) / denom).replace([np.inf, -np.inf], np.nan)
    return ohlc


def ensure_market_data(symbol: str, start: pd.Timestamp, indicator_config) -> pd.DataFrame | None:
    market_df = ensure_market_history(symbol, start)
    if market_df is None or market_df.empty:
        return None

    required_cols = _required_indicator_cols(indicator_config)
    if not required_cols:
        return market_df

    ind_df = load_indicator(symbol)
    if ind_df is None or ind_df.empty:
        ind_df = calculate_indicators(market_df, _min_config_for_cols(required_cols))
        save_indicator(symbol, ind_df)

    missing = list(set(required_cols) - set(ind_df.columns))
    if missing:
        new_df = calculate_indicators(market_df, _min_config_for_cols(missing))
        new_df = new_df[["datetime"] + [col for col in missing if col in new_df.columns]]
        ind_df = ind_df.merge(new_df, on="datetime", how="left")
        save_indicator(symbol, ind_df)

    use_cols = ["datetime"] + required_cols
    for col in use_cols:
        if col not in ind_df.columns:
            raise KeyError(f"Indicator column missing: {col}")

    ohlc = market_df.merge(ind_df[use_cols], on="datetime", how="left")
    ohlc = _add_macd(ohlc, indicator_config)
    ohlc = _add_bb_p(ohlc, indicator_config)
    ohlc = _add_vol_ratio(ohlc, indicator_config)
    return ohlc
