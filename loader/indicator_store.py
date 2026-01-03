from functools import lru_cache
from pathlib import Path
import pandas as pd

INDICATOR_PATH = Path("data/indicators")
INDICATOR_PATH.mkdir(parents=True, exist_ok=True)

def _indicator_file(symbol: str) -> Path:
    return INDICATOR_PATH / f"{symbol}.parquet"

@lru_cache(maxsize=512)
def load_indicator(symbol: str) -> pd.DataFrame | None:
    path = _indicator_file(symbol)
    if not path.exists():
        return None
    df = pd.read_parquet(path, engine="pyarrow")
    return df

def save_indicator(symbol: str, df: pd.DataFrame):
    path = _indicator_file(symbol)
    df.sort_values("datetime", inplace=True)
    df.reset_index(drop=True, inplace=True)
    cols = ['datetime'] + [col for col in df.columns if col != 'timestamp' and col != 'datetime']
    df[cols].to_parquet(path, engine="pyarrow", compression="zstd")
