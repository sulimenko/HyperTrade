import pandas as pd
import numpy as np
from loader.market_loader import ensure_market_history
from loader.indicator_calc import calculate_indicators
from loader.indicator_store import load_indicator, save_indicator

def _required_indicator_cols(indicator_config) -> list[str]:
    cols = []

    if not isinstance(indicator_config, dict):
        return cols

    # --- EMA ---
    ema_cfg = indicator_config.get("ema")
    if ema_cfg and len(ema_cfg) >= 4 and bool(ema_cfg[0]):
        _, _, fast, slow = ema_cfg[:4]
        if fast is not None:
            cols.append(f"ema_{int(fast)}")
        if slow is not None and int(slow) != int(fast):
            cols.append(f"ema_{int(slow)}")
    
    # --- MACD ---
    macd_cfg = indicator_config.get("macd")
    if macd_cfg and len(macd_cfg) >= 5 and bool(macd_cfg[0]):
        _, _, fast, slow, _ = macd_cfg[:5]
        if fast is not None:
            cols.append(f"ema_{int(fast)}")
        if slow is not None and int(slow) != int(fast):
            cols.append(f"ema_{int(slow)}")
    
    # --- RSI ---
    rsi_cfg = indicator_config.get("rsi")
    if rsi_cfg and len(rsi_cfg) >= 4 and bool(rsi_cfg[0]):
        _, _, _, period = rsi_cfg[:4]
        if period is not None:
            cols.append(f"rsi_{int(period)}")

    # --- ATR ---
    atr_cfg = indicator_config.get("atr")
    if atr_cfg and len(atr_cfg) >= 2 and bool(atr_cfg[0]):
        _, period = atr_cfg[:2]
        if period is not None:
            cols.append(f"atr_{int(period)}")

    # --- Donchian ---
    don_cfg = indicator_config.get("donchian")
    if don_cfg and len(don_cfg) >= 2 and bool(don_cfg[0]):
        _, period = don_cfg[:2]
        if period is not None:
            cols.append(f"don_h_{int(period)}")
            cols.append(f"don_l_{int(period)}")

    # --- ADX (+DI/-DI) ---
    adx_cfg = indicator_config.get("adx")
    if adx_cfg and len(adx_cfg) >= 4 and bool(adx_cfg[0]):
        _, _, period, _ = adx_cfg[:4]
        if period is not None:
            cols.append(f"adx_{int(period)}")
            cols.append(f"di_plus_{int(period)}")
            cols.append(f"di_minus_{int(period)}")

    # --- Bollinger (store heavy m/u/l; pband on-the-fly) ---
    bb_cfg = indicator_config.get("bb")
    if bb_cfg and len(bb_cfg) >= 3 and bool(bb_cfg[0]):
        _, period, n_std = bb_cfg[:3]
        if period is not None and n_std is not None:
            d = float(n_std)
            k = f"{d:g}"
            cols.append(f"bb_m_{int(period)}_{k}")
            cols.append(f"bb_u_{int(period)}_{k}")
            cols.append(f"bb_l_{int(period)}_{k}")

    # --- Volume MA (store) ---
    vol_cfg = indicator_config.get("volume")
    if vol_cfg and len(vol_cfg) >= 4 and bool(vol_cfg[0]):
        _, _, ma_period, _ = vol_cfg[:4]
        if ma_period is not None:
            cols.append(f"vol_ma_{int(ma_period)}")

    # --- VWAP (store vwap) ---
    vwap_cfg = indicator_config.get("vwap")
    if vwap_cfg and len(vwap_cfg) >= 1 and bool(vwap_cfg[0]):
        cols.append("vwap")

    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            out.append(c)
            seen.add(c)
    return out

def _min_config_for_cols(indicator_config: dict, required_cols: list[str]) -> dict:
    """
    Собирает "урезанный" indicator_config, который включает только heavy-колонки,
    реально нужные для required_cols.

    Важно: derived (macd_*, bb_p_*, vol_ratio_*) сюда не добавляем —
    они считаются on-the-fly после merge.
    """
    if not isinstance(indicator_config, dict):
        return {}

    need = set(required_cols)
    cfg = {}

    # EMA periods can come from ema_* required cols
    ema_periods = []
    for c in need:
        if c.startswith("ema_"):
            try:
                ema_periods.append(int(c.split("_", 1)[1]))
            except Exception:
                pass
    if ema_periods:
        # calc_indicators умеет считать EMA по "ema" cfg,
        # для 2 периодов используем fast/slow (даже если это не "cross" фильтр)
        ema_periods = sorted(set(ema_periods))
        if len(ema_periods) == 1:
            cfg["ema"] = [True, None, ema_periods[0], ema_periods[0]]
        else:
            cfg["ema"] = [True, None, ema_periods[0], ema_periods[-1]]

    # RSI: rsi_{p}
    for c in need:
        if c.startswith("rsi_"):
            p = int(c.split("_", 1)[1])
            # format in your indicator_calc: [enabled, _, period, _]
            cfg["rsi"] = [True, None, None, p]
            break

    # ATR: atr_{p}
    for c in need:
        if c.startswith("atr_"):
            p = int(c.split("_", 1)[1])
            cfg["atr"] = [True, p]
            break

    # Donchian: don_h_{p}, don_l_{p}
    don_p = None
    for c in need:
        if c.startswith("don_h_") or c.startswith("don_l_"):
            don_p = int(c.split("_", 2)[2])
            break
    if don_p is not None:
        cfg["donchian"] = [True, don_p]

    # ADX: adx_{p}, di_plus_{p}, di_minus_{p}
    adx_p = None
    for c in need:
        if c.startswith("adx_") or c.startswith("di_plus_") or c.startswith("di_minus_"):
            # name like adx_14 OR di_plus_14
            adx_p = int(c.rsplit("_", 1)[1])
            break
    if adx_p is not None:
        # format assumed in indicator_calc: [enabled, sign, min_adx, period]
        # (sign/min_adx are used in filters; for calc only period matters)
        cfg["adx"] = [True, None, None, adx_p]

    # Bollinger heavy: bb_m_{p}_{k}, bb_u_{p}_{k}, bb_l_{p}_{k}
    bb_p = None
    bb_k = None
    for c in need:
        if c.startswith("bb_m_") or c.startswith("bb_u_") or c.startswith("bb_l_"):
            parts = c.split("_")
            # bb_m_{p}_{k}
            if len(parts) >= 4:
                bb_p = int(parts[2])
                bb_k = parts[3]
                break
    if bb_p is not None and bb_k is not None:
        # n_std in calc_indicators kept as float and formatted with :g in column name
        try:
            n_std = float(bb_k)
        except Exception:
            n_std = None
        if n_std is not None:
            cfg["bb"] = [True, bb_p, n_std]

    # Volume MA heavy: vol_ma_{p}
    vol_p = None
    for c in need:
        if c.startswith("vol_ma_"):
            vol_p = int(c.split("_", 2)[2])
            break
    if vol_p is not None:
        # format in your calc_indicators: [enabled, vol_sign, ma_period, vol_k]
        # (vol_sign/vol_k used in filters; for calc only ma_period matters)
        cfg["volume"] = [True, None, vol_p, None]

    # VWAP heavy: vwap
    if "vwap" in need:
        cfg["vwap"] = [True, None, None]

    return cfg


def _min_config_for_missing(indicator_config: dict, required_cols: list[str], missing_cols: list[str]) -> dict:
    """
    Как _min_config_for_cols, но строит конфиг только для реально missing heavy колонок.
    Это ускоряет пересчёт и гарантирует отсутствие лишних расчётов.
    """
    # Считаем только то, что не хватает; но EMA для MACD может понадобиться для derived позже.
    # Поэтому лучше брать missing_cols + "обязательные" EMA из required_cols (если MACD потом будет считаться).
    want = set(missing_cols)

    # Если в required_cols есть ema_*, а missing нет — не добавляем, нам они уже есть.
    # Если missing содержит macd derived — мы их тут не считаем вообще.
    # (macd_* добавится on-the-fly после merge, если cfg["macd"] включен)
    filtered = [c for c in required_cols if c in want]

    # Если filtered пуст, но missing есть (например, из-за странных колонок) — попробуем по missing
    base = filtered if filtered else missing_cols

    return _min_config_for_cols(indicator_config, base)


def _add_macd(ohlc: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    macd_cfg = indicator_config.get("macd") if isinstance(indicator_config, dict) else None
    if not (macd_cfg and len(macd_cfg) >= 5 and bool(macd_cfg[0])):
        return ohlc

    _, _, fast, slow, sig = macd_cfg[:5]
    if fast is None or slow is None or sig is None:
        return ohlc

    f = int(fast)
    s = int(slow)
    g = int(sig)

    ef = f"ema_{f}"
    es = f"ema_{s}"
    if ef not in ohlc.columns or es not in ohlc.columns:
        return ohlc  # дальше фильтр зареджектит, если MACD реально обязателен

    macd = ohlc[ef] - ohlc[es]
    macds = macd.ewm(span=g, adjust=False).mean()
    macdh = macd - macds

    ohlc[f"macd_{f}_{s}_{g}"] = macd
    ohlc[f"macds_{f}_{s}_{g}"] = macds
    ohlc[f"macdh_{f}_{s}_{g}"] = macdh
    return ohlc

def _add_bb_p(ohlc: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    bb_cfg = indicator_config.get("bb") if isinstance(indicator_config, dict) else None
    if not (bb_cfg and len(bb_cfg) >= 3 and bool(bb_cfg[0])):
        return ohlc

    _, period, n_std = bb_cfg[:3]
    if period is None or n_std is None:
        return ohlc

    d = float(n_std)
    k = f"{d:g}"

    m = f"bb_m_{int(period)}_{k}"
    u = f"bb_u_{int(period)}_{k}"
    l = f"bb_l_{int(period)}_{k}"
    if m not in ohlc.columns or u not in ohlc.columns or l not in ohlc.columns:
        return ohlc

    denom = (ohlc[u] - ohlc[l]).replace(0.0, np.nan)
    bbp = ((ohlc["close"] - ohlc[l]) / denom).replace([np.inf, -np.inf], np.nan)
    ohlc[f"bb_p_{int(period)}_{k}"] = bbp
    return ohlc

def _add_vol_ratio(ohlc: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    vol_cfg = indicator_config.get("volume") if isinstance(indicator_config, dict) else None
    if not (vol_cfg and len(vol_cfg) >= 4 and bool(vol_cfg[0])):
        return ohlc

    _, _, ma_period, _ = vol_cfg[:4]
    if ma_period is None:
        return ohlc

    ma_col = f"vol_ma_{int(ma_period)}"
    if "volume" not in ohlc.columns or ma_col not in ohlc.columns:
        return ohlc

    denom = ohlc[ma_col].replace(0.0, np.nan)
    ratio = (ohlc["volume"].astype(float) / denom).replace([np.inf, -np.inf], np.nan)
    ohlc[f"vol_ratio_{int(ma_period)}"] = ratio
    return ohlc

def ensure_market_data(symbol: str, start: pd.Timestamp, indicator_config) -> pd.DataFrame | None:
    market_df = ensure_market_history(symbol, start)
    if market_df is None or market_df.empty:
        return None

    required_cols = _required_indicator_cols(indicator_config)
    if not required_cols:
        return market_df

    min_cfg_full = _min_config_for_cols(indicator_config, required_cols)

    ind_df = load_indicator(symbol)
    if ind_df is None or ind_df.empty:
        ind_df = calculate_indicators(market_df, min_cfg_full)
        save_indicator(symbol, ind_df)

    missing = list(set(required_cols) - set(ind_df.columns))
    if missing:
        min_cfg_miss = _min_config_for_missing(indicator_config, required_cols, missing)
        new_df = calculate_indicators(market_df, min_cfg_miss)
        new_df = new_df[["datetime"] + [c for c in missing if c in new_df.columns]]
        ind_df = ind_df.merge(new_df, on="datetime", how="left")
        save_indicator(symbol, ind_df)

    need = ["datetime"] + required_cols
    for col in need:
        if col not in ind_df.columns:
            raise KeyError(f"Indicator column missing: {col}")

    ind_use = ind_df[need]
    ohlc = market_df.merge(ind_use, on="datetime", how="left")

    # --- derived/light (on-the-fly, do NOT store) ---
    ohlc = _add_macd(ohlc, indicator_config)
    ohlc = _add_bb_p(ohlc, indicator_config)
    ohlc = _add_vol_ratio(ohlc, indicator_config)

    return ohlc
