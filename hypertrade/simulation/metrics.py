"""Research quality metrics: fixed nominal exposure, never a capital return."""
from __future__ import annotations
import math
import numpy as np
import pandas as pd


def sequential_curve(trades: list[dict], nominal: float = 10_000) -> pd.DataFrame:
    if not trades:
        return pd.DataFrame(columns=['sequence', 'normalized_pnl', 'cumulative_normalized_pnl', 'proxy_drawdown_usd'])
    frame = pd.DataFrame(trades).sort_values(['entry_dt', 'symbol', 'direction'], kind='stable').reset_index(drop=True)
    pnl = frame['return_pct'].astype(float) * nominal / 100
    equity = pnl.cumsum()
    peak = equity.cummax().clip(lower=0)
    return pd.DataFrame({'sequence': np.arange(1, len(frame)+1), 'normalized_pnl': pnl,
                         'cumulative_normalized_pnl': equity, 'proxy_drawdown_usd': peak-equity})


def compute_metrics(trades: list[dict], params=None) -> dict[str, float | None]:
    admitted = [t for t in trades if not t.get('rejected')]
    nominal = float(getattr(params, 'normalized_notional', 10_000))
    returns = np.array([t['return_pct'] for t in admitted], dtype=float)
    if not np.isfinite(returns).all():
        raise ValueError('Non-finite trade return')
    wins = returns[returns > 0].sum()
    losses = -returns[returns < 0].sum()
    # None means undefined (+infinity when only gains); never an arbitrary capped PF.
    pf = float(wins / losses) if losses > 0 else None
    ordered = np.sort(returns)
    curve = sequential_curve(admitted, nominal)
    return {
        'trades': float(len(returns)),
        'sum_trade_returns_pct': float(returns.sum()),
        'mean_trade_return_pct': float(returns.mean()) if returns.size else 0.0,
        'median_trade_return_pct': float(np.median(returns)) if returns.size else 0.0,
        'win_rate_pct': float((returns > 0).mean()*100) if returns.size else 0.0,
        'profit_factor': pf,
        'normalized_pnl_usd': float(returns.sum()*nominal/100),
        'normalized_notional_usd': nominal,
        'proxy_drawdown_usd': float(curve['proxy_drawdown_usd'].max()) if not curve.empty else 0.0,
        'cvar_loss_5_pct': float(max(0, -ordered[:max(1, math.ceil(len(ordered)*.05))].mean())) if returns.size else 0.0,
        'avg_hold_minutes': float(np.mean([t['hold_minutes'] for t in admitted])) if admitted else 0.0,
    }
