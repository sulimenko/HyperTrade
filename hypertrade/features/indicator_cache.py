"""Read legacy source columns; write additional columns only to the derived cache."""
from pathlib import Path
import pandas as pd
from hypertrade.data.market import _load
INDICATOR_PATH = Path('data/indicators')
DERIVED_PATH = Path('data/cache/indicators')

def load_indicator(symbol: str) -> pd.DataFrame | None:
    source = _load(INDICATOR_PATH / f'{symbol}.parquet')
    derived = _load(DERIVED_PATH / f'{symbol}.parquet')
    if derived is None:
        return source
    if source is None:
        return derived
    # Newly required columns take precedence only in the separate derived cache.
    return source.drop(columns=source.columns.intersection(derived.columns).difference(['datetime'])).merge(derived, on='datetime', how='outer')


def save_indicator(symbol: str, frame: pd.DataFrame) -> None:
    DERIVED_PATH.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(DERIVED_PATH / f'{symbol}.parquet', index=False, compression='zstd')
