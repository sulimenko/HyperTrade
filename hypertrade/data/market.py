"""Read-only research inputs and explicit, separately staged market preparation."""
from functools import lru_cache
import os
from pathlib import Path
import re

import pandas as pd
import requests

from hypertrade.data.signals import utc_timestamp

MARKET_PATH = Path('data/ohlc')
PREPARED_PATH = Path('data/cache/market')
USA_MARKETS = {'NASDAQ', 'NYSE', 'AMEX', 'NYSE ARCA', 'CBOE'}


class MissingMarketDataError(ValueError):
    def __init__(self, symbol, start, detail):
        super().__init__(f'Market data preparation required for {symbol} at {start}: {detail}. '
                         'Run .venv/bin/python -m hypertrade.data.market --signals <CSV> '
                         'with DATA_BASE_URL configured, or provide prepared 15-minute UTC '
                         'OHLC parquet in data/cache/market. Research never fetches per trial.')


def _symbol(symbol: str) -> str:
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9.^_-]*', symbol):
        raise ValueError('Invalid market symbol')
    return symbol


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


def _history(symbol: str) -> pd.DataFrame | None:
    _symbol(symbol)
    staged = _load(PREPARED_PATH / f'{symbol}.parquet')
    if staged is not None:
        staged = staged.copy()
        staged.attrs['prepared_market'] = True
        return staged
    return _load(MARKET_PATH / f'{symbol}.parquet')


def ensure_market_history(symbol: str, start) -> pd.DataFrame:
    frame = _history(symbol)
    if frame is None or frame.empty:
        raise MissingMarketDataError(symbol, start, 'missing/empty market history')
    if start is not None and frame['datetime'].min() > utc_timestamp(start):
        raise MissingMarketDataError(symbol, start, 'history starts after the requested signal')
    # Existing histories may end early (e.g. a delisted symbol); later signals retain
    # the explicit no-candles eligibility outcome, not a missing-instrument outcome.
    return frame


def fetch_market_data(symbol: str, base_url: str, limit: int = 20000) -> pd.DataFrame:
    """Previous configured market API contract; called only by explicit preparation."""
    response = requests.get(f'{base_url.rstrip("/")}/api/marketData/symbolSearch/',
                            json={'data': symbol}, timeout=15)
    response.raise_for_status()
    instrument = next(({'symbol': row['symbol'], 'source': row['source']}
                       for row in response.json().get('result', [])
                       if row.get('symbol') == symbol and row.get('source') in USA_MARKETS), None)
    if instrument is None:
        raise MissingMarketDataError(symbol, None, 'symbol not recognized on a US market by configured API')
    response = requests.post(f'{base_url.rstrip("/")}/api/marketData/addChartSymbols',
                             json={'instruments': [instrument], 'period': 900, 'limit': limit}, timeout=30)
    response.raise_for_status()
    payload = response.json().get('result') or {}
    frame = pd.DataFrame(payload.get(symbol, {}).get('chart', {}).get('full', []))
    if not {'timestamp', 'open', 'high', 'low', 'close'}.issubset(frame.columns):
        raise MissingMarketDataError(symbol, None, 'API returned no valid OHLC chart')
    frame['datetime'] = pd.to_datetime(frame.pop('timestamp'), unit='ms', utc=True)
    return frame


def prepare_market_history(symbol: str, start, end=None, *, base_url: str | None = None) -> Path:
    """Acquire missing coverage once, never overwrite protected source files."""
    frame = _history(symbol)
    start = utc_timestamp(start)
    end = utc_timestamp(end) if end is not None else start
    if end < start:
        raise ValueError('Market preparation end precedes start')

    def covered(data):
        return (data is not None and not data.empty and data.datetime.min() <= start
                and data.datetime.max() + pd.Timedelta(minutes=15) > end)

    if covered(frame):
        staged = PREPARED_PATH / f'{symbol}.parquet'
        return staged if staged.exists() else MARKET_PATH / f'{symbol}.parquet'
    if not base_url:
        raise MissingMarketDataError(symbol, start, 'coverage incomplete and DATA_BASE_URL is not configured')
    try:
        acquired = fetch_market_data(symbol, base_url)
    except requests.RequestException as exc:
        raise MissingMarketDataError(symbol, start, f'configured API unavailable ({type(exc).__name__})') from None
    combined = pd.concat([f for f in (frame, acquired) if f is not None], ignore_index=True)
    combined = combined.sort_values('datetime').drop_duplicates('datetime', keep='last').reset_index(drop=True)
    if not covered(combined):
        raise MissingMarketDataError(symbol, start, f'API coverage does not span {start} through {end}')
    prices = combined[['open', 'high', 'low', 'close']].apply(pd.to_numeric, errors='coerce')
    import numpy as np
    if not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any():
        raise MissingMarketDataError(symbol, start, 'API returned invalid prices')
    combined[prices.columns] = prices
    PREPARED_PATH.mkdir(parents=True, exist_ok=True)
    target = PREPARED_PATH / f'{symbol}.parquet'
    temporary = target.with_suffix('.parquet.tmp')
    combined.to_parquet(temporary, index=False, compression='zstd')
    temporary.replace(target)
    return target


def main():
    import argparse
    from hypertrade.data.signals import load_external_signals
    parser = argparse.ArgumentParser(description='Explicit market preparation; immutable source OHLC is preserved')
    parser.add_argument('--signals', required=True)
    args = parser.parse_args()
    requirements = {}
    for signal in load_external_signals(args.signals):
        for symbol in signal['long'] + signal['short']:
            requirements.setdefault(symbol, []).append(utc_timestamp(signal['datetime']))
    try:
        for symbol, times in sorted(requirements.items()):
            print(prepare_market_history(symbol, min(times), max(times), base_url=os.getenv('DATA_BASE_URL')))
    except (MissingMarketDataError, ValueError) as exc:
        parser.exit(1, f'{exc}\n')


if __name__ == '__main__':
    main()
