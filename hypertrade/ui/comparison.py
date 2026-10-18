"""Display-only views of saved trades. Never rewrite sequential or money curves."""
import pandas as pd


def calendar_domain(frames, period=()):
    stamps = list(period)
    for frame in frames:
        for name in ('datetime', 'signal_dt', 'entry_dt', 'exit_event_dt', 'exit_dt'):
            if name in frame:
                stamps.extend(frame[name].dropna().tolist())
    dates = pd.to_datetime(pd.Series(stamps), utc=True, errors='coerce').dropna()
    if dates.empty:
        raise ValueError('Нет сохранённых календарных дат для сравнения.')
    return dates.min(), dates.max()


def realized_comparison(trades_by_label, period=(), nominal=10000):
    """Group simultaneous exits, then carry realized PnL on a common union of times.

    Legacy bundles without exit_event_dt use exit_dt (candle timestamp). Missing
    both dates or PnL is an error, never a silently omitted trade.
    """
    start, end = calendar_domain(list(trades_by_label.values()), period)
    totals = {}
    for label, trades in trades_by_label.items():
        if trades.empty:
            totals[label] = pd.Series(dtype=float)
            continue
        exits = pd.to_datetime(trades.get('exit_event_dt', pd.Series(index=trades.index, dtype=object)), utc=True)
        if 'exit_dt' in trades:
            exits = exits.fillna(pd.to_datetime(trades['exit_dt'], utc=True))
        pnl = (trades['normalized_pnl_usd'].astype(float) if 'normalized_pnl_usd' in trades
               else trades['return_pct'].astype(float) * nominal / 100)
        if exits.isna().any() or not pnl.map(lambda v: pd.notna(v) and abs(v) != float('inf')).all():
            raise ValueError(f'{label}: отсутствует время закрытия или конечный PnL.')
        totals[label] = pd.Series(pnl.to_numpy(), index=exits).groupby(level=0).sum().sort_index().cumsum()
    first_exit = min((s.index.min() for s in totals.values() if not s.empty), default=end)
    # A zero precedes even a legacy instantaneous first close.
    if start >= first_exit:
        start = first_exit - pd.Timedelta(nanoseconds=1)
    axis = pd.DatetimeIndex([start, end])
    for series in totals.values():
        axis = axis.union(series.index)
    axis = axis.sort_values()
    return pd.concat([
        pd.DataFrame({'datetime': axis, 'cumulative_normalized_pnl': s.reindex(axis).ffill().fillna(0).to_numpy(), 'source': label})
        for label, s in totals.items()
    ], ignore_index=True)
