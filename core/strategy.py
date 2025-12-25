def check_entry(df, params):
    close = df["close"]
    ma = close.rolling(params.entry_lookback).mean()

    return close.iloc[-1] > ma.iloc[-1]
