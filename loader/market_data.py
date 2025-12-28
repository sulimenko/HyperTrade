import os
import pandas as pd
from loader.api_client import fetch_market_data

_MARKET_CACHE = {}

def load_market_data(symbol: str, start) -> pd.DataFrame | None:
    if symbol in _MARKET_CACHE:
        return _MARKET_CACHE[symbol]
    
    path = f"data/market/{symbol}.csv"

    if os.path.exists(path):
        ohlc = pd.read_csv(path, sep=";")
        if not ohlc.empty and 'timestamp' in ohlc.columns:
            if ohlc["timestamp"].iloc[0] <= int(start.timestamp() * 1000):
                ohlc["datetime"] = pd.to_datetime(ohlc["timestamp"], unit="ms", utc=True)
                _MARKET_CACHE[symbol] = ohlc
                return ohlc

    ohlc = fetch_market_data(symbol)
    if ohlc is None:
        return None
    ohlc.to_csv(path, sep=";", index=False)
    ohlc["datetime"] = pd.to_datetime(ohlc["timestamp"], unit="ms", utc=True)
    _MARKET_CACHE[symbol] = ohlc
    return ohlc