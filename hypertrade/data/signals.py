"""External CSV signals. Naive timestamps explicitly mean UTC; offsets retain their instant."""
from pathlib import Path
import pandas as pd


def utc_timestamp(value) -> pd.Timestamp:
    text = str(value).strip()
    stamp = pd.to_datetime(text, dayfirst=True) if len(text) > 2 and text[2] == '.' else pd.Timestamp(text)
    return stamp.tz_localize('UTC') if stamp.tzinfo is None else stamp.tz_convert('UTC')


def _symbols(value) -> list[str]:
    if pd.isna(value):
        return []
    return [s.strip() for s in str(value).split(',') if s.strip()]


def load_external_signals(path: str | Path) -> list[dict]:
    frame = pd.read_csv(path, sep=';', encoding='utf-8-sig')
    if 'datetime' not in frame:
        raise ValueError('CSV requires datetime column')
    if not {'long_symbols', 'symbols', 'short_symbols', 'shotr_symbols'} & set(frame):
        raise ValueError('CSV requires long_symbols/symbols or short_symbols')
    groups = {}
    for row in frame.to_dict('records'):
        stamp = utc_timestamp(row['datetime'])
        signal = groups.setdefault(stamp, {'datetime': stamp, 'long': [], 'short': []})
        for label, key in [('long', 'long_symbols' if 'long_symbols' in frame else 'symbols'),
                           ('short', 'short_symbols' if 'short_symbols' in frame else 'shotr_symbols')]:
            # The historical fixture has shotr_symbols. Input is never rewritten.
            signal[label].extend(s for s in _symbols(row.get(key)) if s not in signal[label])
    return [groups[t] for t in sorted(groups)]
