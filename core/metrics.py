# core/metrics.py

import math
from typing import Optional, Any, Dict

import numpy as np
import pandas as pd


def _to_num(a, default=0.0):
    try:
        v = float(a)
        if math.isnan(v) or math.isinf(v):
            return float(default)
        return v
    except Exception:
        return float(default)


def _cvar_left_tail(x: np.ndarray, alpha: float = 0.05) -> float:
    if x.size == 0:
        return 0.0
    xs = np.sort(x)
    k = max(1, int(math.floor(alpha * xs.size)))
    return float(xs[:k].mean())


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    dd = peak - equity
    return float(dd.max())


def _log1p_pos(x: float) -> float:
    return float(math.log1p(max(0.0, _to_num(x, 0.0))))


def _clamp(x: float, lo: float, hi: float) -> float:
    v = _to_num(x, lo)
    if v < lo:
        return float(lo)
    if v > hi:
        return float(hi)
    return float(v)


def compute_metrics(
    trades: list[dict],
    params: Optional[Any] = None,
    *,
    delay_penalty_k: float = 0.0,
) -> Dict[str, float]:
    df = pd.DataFrame(trades)
    if df.empty:
        return {}

    if "rejected" in df.columns:
        df = df[df["rejected"] == False]  # noqa: E712
        if df.empty:
            return {}

    # --- pnl in % ---
    rp = pd.to_numeric(df.get("return_pct"), errors="coerce")
    rp = rp.replace([np.inf, -np.inf], np.nan).dropna()
    if rp.empty:
        return {}

    pnl_pct = rp.to_numpy(dtype=float)
    rets = pnl_pct / 100.0

    total_pnl_pct = float(pnl_pct.sum())

    wins = pnl_pct[pnl_pct > 0]
    losses = pnl_pct[pnl_pct < 0]
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(losses.sum()) if losses.size else 0.0

    if gross_loss < 0:
        profit_factor = float(gross_profit / abs(gross_loss)) if abs(gross_loss) > 0 else 10.0
    else:
        profit_factor = 10.0

    equity = np.cumsum(pnl_pct)
    max_dd_pct = _max_drawdown(equity)
    calmar = float(total_pnl_pct / max_dd_pct) if max_dd_pct > 0 else 0.0
    cvar_5 = _cvar_left_tail(rets, 0.05)

    # --- hold in market minutes (preferred) ---
    hold = None
    if "hold_minutes" in df.columns:
        hm = pd.to_numeric(df["hold_minutes"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        hm = hm[hm >= 0]
        if not hm.empty:
            hold = float(hm.mean())

    if hold is None and "hold_bars" in df.columns:
        hb = pd.to_numeric(df["hold_bars"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        hb = hb[hb >= 0]
        if not hb.empty:
            bar_minutes = int(getattr(params, "bar_minutes", 15)) if params is not None else 15
            hold = float(hb.mean() * float(bar_minutes))

    if hold is None:
        hold = 0.0

    # --- Variant B score ---
    score = 0.0
    score += 1.25 * _log1p_pos(total_pnl_pct)
    score += 0.20 * _log1p_pos(max(0.0, calmar))
    score += 0.5 * _clamp(profit_factor, 0.0, 5.0)

    score -= 0.50 * _log1p_pos(max_dd_pct)
    score -= 0.70 * _log1p_pos(abs(cvar_5))
    score -= 0.5 * _log1p_pos(hold)

    if delay_penalty_k and params is not None:
        score -= float(delay_penalty_k) * float(int(getattr(params, "delay_open", 0)))

    return {
        "trades": float(len(pnl_pct)),
        "total_pnl": float(total_pnl_pct),
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_dd_pct),
        "calmar": float(calmar),
        "cvar_5": float(cvar_5),
        "avg_hold_minutes": float(hold),
        "score": float(score),
    }
