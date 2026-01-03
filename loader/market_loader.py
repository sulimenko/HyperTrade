from pathlib import Path
from functools import lru_cache
import pandas as pd
from loader.api_client import fetch_market_data

MARKET_PATH = Path("data/ohlc")
MARKET_PATH.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=512)
def load_market(symbol: str) -> pd.DataFrame | None:
    path = MARKET_PATH / f"{symbol}.parquet"
    if not path.exists():
        return None
    return pd.read_parquet(path, engine="pyarrow")

def save_market(symbol: str, df: pd.DataFrame):
    path = MARKET_PATH / f"{symbol}.parquet"
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_parquet(path, engine="pyarrow", compression="zstd")

def ensure_market_history(symbol: str, start, api=True) -> pd.DataFrame | None:
    df = load_market(symbol)
    if df is not None or not api:
        if df["datetime"].min() <= start:
            return df

    # ⬇️ догружаем через API
    api_df = fetch_market_data(symbol)
    if api_df is None:
        return None

    save_market(symbol, api_df)
    return ensure_market_history(symbol, start, api=False)
