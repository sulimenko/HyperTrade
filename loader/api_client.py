import os
import requests
import pandas as pd
import logging
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("DATA_BASE_URL")

USA_MARKETS = {
    "NASDAQ",
    "NYSE",
    "AMEX",
    "NYSE ARCA"
}

def search_symbol(symbol: str) -> Optional[dict]:
    """
    Возвращает instrument {symbol, source} для USA рынков
    """
    response = requests.get(
        f"{BASE_URL}/api/marketData/symbolSearch/",
        json={"data": symbol},
        timeout=15
    )

    response.raise_for_status()
    instruments = response.json().get("result", [])

    for instrument in instruments:
        if ( instrument.get("symbol") == symbol and instrument.get("source") in USA_MARKETS):
            return { "symbol": instrument["symbol"], "source": instrument["source"] }

    return None

def fetch_candles(instrument: dict) -> pd.DataFrame:
    payload = {
        "instruments": [instrument],
        "period": 60 * 15,
        "limit": 15000
    }

    response = requests.post(
        f"{BASE_URL}/api/marketData/addChartSymbols",
        json=payload,
        timeout=30
    )

    response.raise_for_status()
    data = response.json().get("result", [])
    if not data:
        logging.warning(f"Not found data " + str(instrument))
        return None
    ohlc = pd.DataFrame(data[instrument["symbol"]].get("chart", {}).get("full", []))

    required_cols = {"open", "high", "low", "close", "timestamp"}
    if not required_cols.issubset(ohlc.columns):
        raise ValueError(
            f"Invalid OHLC format, got columns: {ohlc.columns}"
        )

    return ohlc

def fetch_market_data(symbol: str) -> pd.DataFrame:
    instrument = search_symbol(symbol)

    if instrument is None:
        logging.warning(f"Symbol {symbol} not found on USA markets")
        return None

    return fetch_candles(instrument)
