import pandas as pd
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
        fast, slow = ema_cfg[2], ema_cfg[3]
        if fast is not None:
            cols.append(f"ema_{int(fast)}")
        if slow is not None and int(slow) != int(fast):
            cols.append(f"ema_{int(slow)}")
    
    # --- RSI ---
    rsi_cfg = indicator_config.get("rsi")
    if rsi_cfg and len(rsi_cfg) >= 4 and bool(rsi_cfg[0]):
        period = rsi_cfg[3]
        if period is not None:
            cols.append(f"rsi_{int(period)}")

    # --- ATR ---
    atr_cfg = indicator_config.get("atr")
    if atr_cfg and len(atr_cfg) >= 2 and bool(atr_cfg[0]):
        period = atr_cfg[1]
        if period is not None:
            cols.append(f"atr_{int(period)}")

    # --- Donchian ---
    don_cfg = indicator_config.get("donchian")
    if don_cfg and len(don_cfg) >= 2 and bool(don_cfg[0]):
        period = don_cfg[1]
        if period is not None:
            p = int(period)
            cols.append(f"don_h_{p}")
            cols.append(f"don_l_{p}")

    # # --- ADX (+DI/-DI) ---
    # adx_cfg = indicator_config.get("adx")
    # if adx_cfg and len(adx_cfg) >= 4 and bool(adx_cfg[0]):
    #     period = adx_cfg[3]
    #     if period is not None:
    #         p = int(period)
    #         cols.append(f"adx_{p}")
    #         cols.append(f"di_plus_{p}")
    #         cols.append(f"di_minus_{p}")

    # # --- MACD ---
    # macd_cfg = indicator_config.get("macd")
    # if macd_cfg and len(macd_cfg) >= 5 and bool(macd_cfg[0]):
    #     _, _, fast, slow, sig = macd_cfg[:5]
    #     if fast is not None and slow is not None and sig is not None:
    #         f = int(fast)
    #         s = int(slow)
    #         g = int(sig)
    #         cols.append(f"macd_{f}_{s}_{g}")
    #         cols.append(f"macds_{f}_{s}_{g}")
    #         cols.append(f"macdh_{f}_{s}_{g}")

    # # --- Bollinger ---
    # bb_cfg = indicator_config.get("bb")
    # if bb_cfg and len(bb_cfg) >= 3 and bool(bb_cfg[0]):
    #     period, n_std = bb_cfg[1], bb_cfg[2]
    #     if period is not None and n_std is not None:
    #         p = int(period)
    #         d = float(n_std)
    #         k = f"{d:g}"
    #         cols.append(f"bb_m_{p}_{k}")
    #         cols.append(f"bb_u_{p}_{k}")
    #         cols.append(f"bb_l_{p}_{k}")
    #         cols.append(f"bb_p_{p}_{k}")

    # # --- Volume MA/Ratio ---
    # vol_cfg = indicator_config.get("volume")
    # if vol_cfg and len(vol_cfg) >= 4 and bool(vol_cfg[0]):
    #     ma_period = vol_cfg[2]
    #     if ma_period is not None:
    #         p = int(ma_period)
    #         cols.append(f"vol_ma_{p}")
    #         cols.append(f"vol_ratio_{p}")

    # # --- VWAP ---
    # vwap_cfg = indicator_config.get("vwap")
    # if vwap_cfg and len(vwap_cfg) >= 1 and bool(vwap_cfg[0]):
    #     cols.append("vwap")

    # # --- SR pivots/fractals ---
    # sr_cfg = indicator_config.get("sr")
    # if sr_cfg and len(sr_cfg) >= 2 and bool(sr_cfg[0]):
    #     left = sr_cfg[1]
    #     l = int(left) if left is not None else 2
    #     cols.append(f"pivot_h_{l}")
    #     cols.append(f"pivot_l_{l}")

    return cols

def ensure_market_data(symbol: str, start: pd.Timestamp, indicator_config) -> pd.DataFrame | None:
    market_df = ensure_market_history(symbol, start)
    if market_df is None or market_df.empty:
        return None

    required_cols = _required_indicator_cols(indicator_config)
    if not required_cols:
        return market_df

    ind_df = load_indicator(symbol)
    if ind_df is None or ind_df.empty:
        ind_df = calculate_indicators(market_df, indicator_config)
        save_indicator(symbol, ind_df)

    missing = list(set(required_cols) - set(ind_df.columns))

    if missing:
        new_df = calculate_indicators(market_df, indicator_config)
        new_df = new_df[["datetime"] + missing]
        ind_df = ind_df.merge(new_df, on="datetime", how="left")
        save_indicator(symbol, ind_df)

    need = ["datetime"] + required_cols
    for col in need:
        if col not in ind_df.columns:
            raise KeyError(f"Indicator column missing: {col}")

    ind_use = ind_df[need]
    return market_df.merge(ind_use, on="datetime", how="left")
