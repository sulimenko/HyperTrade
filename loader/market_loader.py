from pathlib import Path
from functools import lru_cache
import pandas as pd
from loader.api_client import fetch_market_data

MARKET_PATH = Path("data/ohlc")
MARKET_PATH.mkdir(parents=True, exist_ok=True)

@lru_cache(maxsize=2048)
def _read_market(path_str: str, mtime_ns: int) -> pd.DataFrame:
    return pd.read_parquet(path_str, engine="pyarrow")

def load_market(symbol: str) -> pd.DataFrame | None:
    path = MARKET_PATH / f"{symbol}.parquet"
    if not path.exists():
        return None
    return _read_market(str(path), path.stat().st_mtime_ns)

def save_market(symbol: str, df: pd.DataFrame):
    path = MARKET_PATH / f"{symbol}.parquet"
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out.sort_values("datetime", inplace=True)
    out.reset_index(drop=True, inplace=True)
    out.to_parquet(path, engine="pyarrow", compression="zstd")

def ensure_market_history(symbol: str, start, api=True) -> pd.DataFrame | None:
    df = load_market(symbol)

    if df is not None:
        if df["datetime"].min() <= start or not api:
            return df
    else:
        if not api:
            return None

    # ⬇️ догружаем через API
    limit = 15000
    if df is not None and len(df) > limit * 0.8:
        limit += 5000
    api_df = fetch_market_data(symbol, limit)
    if api_df is None or api_df.empty:
        return df

    save_market(symbol, api_df)
    return ensure_market_history(symbol, start, api=False)
