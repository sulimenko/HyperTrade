import pandas as pd
from core.time_utils import calc_entry_datetime

def simulate_trade(symbol, signal_time, params, ohlc):
    entry_dt = calc_entry_datetime(
        signal_time,
        params["dalay_open"]
    )

    ohlc["datetime"] = pd.to_datetime(ohlc["timestamp"], unit="ms", utc=True)
    ohlc = ohlc[ohlc["datetime"] >= entry_dt]

    if ohlc.empty:
        return 0.0

    entry_price = ohlc.iloc[0]["open"]

    sl = entry_price * (1 - params["sl"] / 100)
    tp = entry_price * (1 + params["tp"] / 100)
    for _, row in ohlc.iterrows():
        if row["low"] <= sl:
            return -params["sl"]
        if row["high"] >= tp:
            return params["tp"]

    return 0.0
