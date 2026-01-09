import pandas as pd
from loader.market_loader import ensure_market_history
from loader.indicator_calc import calculate_indicators
from loader.indicator_store import load_indicator, save_indicator

def _required_indicator_cols(indicator_config) -> list[str]:
    cols = []

    ema_cfg = indicator_config.get("ema")
    if ema_cfg and len(ema_cfg) >= 4 and bool(ema_cfg[0]):
        fast, slow = ema_cfg[2], ema_cfg[3]
        if fast is not None:
            cols.append(f"ema_{int(fast)}")
        if slow is not None and int(slow) != int(fast):
            cols.append(f"ema_{int(slow)}")

    rsi_cfg = indicator_config.get("rsi")
    if rsi_cfg and len(rsi_cfg) >= 4 and bool(rsi_cfg[0]):
        period = rsi_cfg[3]
        if period is not None:
            cols.append(f"rsi_{int(period)}")
    
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
