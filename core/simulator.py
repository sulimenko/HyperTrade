def simulate_trade(symbol, signal_time, params, ohlc):
    # !!!
    entry_price = ohlc.iloc[0]["close"]

    sl = entry_price * (1 - params.sl / 100)
    tp = entry_price * (1 + params.tp / 100)
    for _, row in ohlc.iterrows():
        if row["low"] <= sl:
            return -params.sl
        if row["high"] >= tp:
            return params.tp

    return 0.0
