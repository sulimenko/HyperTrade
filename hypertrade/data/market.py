"""Read-only source market data. Network acquisition is outside research runs."""
from functools import lru_cache
from pathlib import Path
import pandas as pd

MARKET_PATH = Path('data/ohlc')


@lru_cache(maxsize=128)
def _read(path: str, mtime_ns: int) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    if not isinstance(frame['datetime'].dtype, pd.DatetimeTZDtype):
        # Legacy naive parquet is explicitly UTC, never local machine time.
        frame['datetime'] = pd.to_datetime(frame['datetime']).dt.tz_localize('UTC')
    else:
        frame['datetime'] = frame['datetime'].dt.tz_convert('UTC')
    return frame.sort_values('datetime').drop_duplicates('datetime').reset_index(drop=True)


def _load(path: Path) -> pd.DataFrame | None:
    return _read(str(path), path.stat().st_mtime_ns) if path.exists() else None


def ensure_market_history(symbol: str, start) -> pd.DataFrame | None:
    if Path(symbol).name != symbol:
        raise ValueError('Invalid market symbol')
    return _load(MARKET_PATH / f'{symbol}.parquet')


