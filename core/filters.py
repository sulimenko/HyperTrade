def default_filters(row) -> bool:
    if not (
        row["ema_fast"] > row["ema_slow"]
        and row["rsi"] > 50
    ):
        return False

    if "volume" in row and "volume_ma" in row:
        return row["volume"] > row["volume_ma"]

    return True

# def ema_trend_filter(entry_time, ohlc: pd.DataFrame, span: int = 50) -> bool:
#     print("ENTRY:", entry_time, "EMA:", ema.loc[entry_time])
#     if entry_time not in ohlc.index:
#         return False

#     close = ohlc["close"]
#     ema = close.ewm(span=span, adjust=False).mean().shift(1)

#     return close.loc[entry_time] > ema.loc[entry_time]


# def atr_volatility_filter(entry_time, ohlc: pd.DataFrame, period: int = 14, min_ratio: float = 0.002) -> bool:
#     if entry_time not in ohlc.index:
#         return False
    
#     high = ohlc["high"]
#     low = ohlc["low"]
#     close = ohlc["close"]

#     tr = (high - low)
#     atr = tr.rolling(period).mean().shift(1)

#     return (atr.loc[entry_time] / close.loc[entry_time]) > min_ratio
