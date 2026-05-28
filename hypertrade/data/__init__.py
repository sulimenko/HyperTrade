from __future__ import annotations

from functools import lru_cache
import logging
import os
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE_URL = os.getenv("DATA_BASE_URL")
MARKET_PATH = Path("data/ohlc")
INDICATOR_PATH = Path("data/indicators")
MARKET_PATH.mkdir(parents=True, exist_ok=True)
INDICATOR_PATH.mkdir(parents=True, exist_ok=True)

USA_MARKETS = {"NASDAQ", "NYSE", "AMEX", "NYSE ARCA", "CBOE"}


def search_symbol(symbol: str) -> Optional[dict]:
    response = requests.get(f"{BASE_URL}/api/marketData/symbolSearch/", json={"data": symbol}, timeout=15)
    response.raise_for_status()
    instruments = response.json().get("result", [])
    for instrument in instruments:
        if instrument.get("symbol") == symbol and instrument.get("source") in USA_MARKETS:
            return {"symbol": instrument["symbol"], "source": instrument["source"]}
    return None


def fetch_candles(instrument: dict, limit: int) -> pd.DataFrame | None:
    payload = {"instruments": [instrument], "period": 60 * 15, "limit": limit}
    response = requests.post(f"{BASE_URL}/api/marketData/addChartSymbols", json=payload, timeout=30)
    response.raise_for_status()
    data = response.json().get("result", [])
    if not data:
        logging.warning("Not found data %s", instrument)
        return None
    ohlc = pd.DataFrame(data[instrument["symbol"]].get("chart", {}).get("full", []))
    required_cols = {"open", "high", "low", "close", "timestamp"}
    if not required_cols.issubset(ohlc.columns):
        raise ValueError(f"Invalid OHLC format, got columns: {ohlc.columns}")
    ohlc["datetime"] = pd.to_datetime(ohlc["timestamp"], unit="ms", utc=True)
    cols = ["datetime"] + [col for col in ohlc.columns if col not in {"timestamp", "datetime"}]
    return ohlc[cols]


def fetch_market_data(symbol: str, limit: int) -> pd.DataFrame | None:
    instrument = search_symbol(symbol)
    if instrument is None:
        logging.warning("Symbol %s not found on USA markets", symbol)
        return None
    return fetch_candles(instrument, limit)


@lru_cache(maxsize=2048)
def _read_market(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_parquet(path_str, engine="pyarrow")


def load_market(symbol: str) -> pd.DataFrame | None:
    path = MARKET_PATH / f"{symbol}.parquet"
    if not path.exists():
        return None
    return _read_market(str(path), path.stat().st_mtime_ns)


def save_market(symbol: str, df: pd.DataFrame) -> None:
    path = MARKET_PATH / f"{symbol}.parquet"
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out.sort_values("datetime", inplace=True)
    out.reset_index(drop=True, inplace=True)
    out.to_parquet(path, engine="pyarrow", compression="zstd")


def ensure_market_history(symbol: str, start, api: bool = True) -> pd.DataFrame | None:
    df = load_market(symbol)
    if df is not None:
        if df["datetime"].min() <= start or not api:
            return df
    elif not api:
        return None

    limit = 15000
    if df is not None and len(df) > limit * 0.8:
        limit += 5000
    api_df = fetch_market_data(symbol, limit)
    if api_df is None or api_df.empty:
        return df
    save_market(symbol, api_df)
    return ensure_market_history(symbol, start, api=False)


def _indicator_file(symbol: str) -> Path:
    return INDICATOR_PATH / f"{symbol}.parquet"


@lru_cache(maxsize=2048)
def _read_indicator(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_parquet(path_str, engine="pyarrow")


def load_indicator(symbol: str) -> pd.DataFrame | None:
    path = _indicator_file(symbol)
    if not path.exists():
        return None
    return _read_indicator(str(path), path.stat().st_mtime_ns)


def save_indicator(symbol: str, df: pd.DataFrame) -> None:
    if df is None or df.empty:
        return
    path = _indicator_file(symbol)
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out.sort_values("datetime", inplace=True)
    out.reset_index(drop=True, inplace=True)
    cols = ["datetime"] + [col for col in out.columns if col not in {"timestamp", "datetime"}]
    out[cols].to_parquet(path, engine="pyarrow", compression="zstd")


__all__ = [
    "ensure_market_history",
    "fetch_market_data",
    "load_indicator",
    "load_market",
    "save_indicator",
    "save_market",
    "search_symbol",
]
