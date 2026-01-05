from functools import lru_cache
from pathlib import Path
import pandas as pd

INDICATOR_PATH = Path("data/indicators")
INDICATOR_PATH.mkdir(parents=True, exist_ok=True)

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

def save_indicator(symbol: str, df: pd.DataFrame):
    if df is None or df.empty:
        return
    path = _indicator_file(symbol)
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], utc=True)
    out.sort_values("datetime", inplace=True)
    out.reset_index(drop=True, inplace=True)

    cols = ["datetime"] + [col for col in out.columns if col not in ("timestamp", "datetime")]
    out[cols].to_parquet(path, engine="pyarrow", compression="zstd")
