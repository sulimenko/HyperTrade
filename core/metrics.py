from __future__ import annotations

from typing import Optional, Any, Dict

import numpy as np
import pandas as pd

def max_drawdown(equity: pd.Series) -> float:
    """Максимальная просадка по кривой equity (в тех же единицах, что equity)."""
    peak = equity.cummax()
    dd = equity - peak
    return float(-dd.min())

def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    return float(a / b) if b not in (0, 0.0, None) else float(default)

def _cvar(x: np.ndarray, alpha: float = 0.05) -> float:
    if x.size == 0:
        return 0.0
    x_sorted = np.sort(x)
    k = max(1, int(np.floor(alpha * x_sorted.size)))
    return float(x_sorted[:k].mean())

def _chunk_stability(x: np.ndarray, n_chunks: int = 4) -> float:
    """Нестабильность среднего по чанкам: std(chunk_means) / (abs(global_mean) + eps)."""
    if x.size < n_chunks * 5:
        return 1.0  # мало данных → считаем нестабильным

    chunks = np.array_split(x, n_chunks)
    means = np.array([c.mean() for c in chunks if c.size > 0], dtype=float)
    if means.size < 2:
        return 1.0
    global_mean = x.mean()
    eps = 1e-9
    return float(means.std(ddof=1) / (abs(global_mean) + eps))

def complexity_penalty(params: Optional[Any]) -> float:
    """Штраф за сложность (простая защита от переобучения): число включённых фильтров."""
    if params is None:
        return 0.0
    cfg = getattr(params, "indicator_config", None)
    if not isinstance(cfg, dict):
        return 0.0

    enabled = 0
    for _, v in cfg.items():
        if isinstance(v, (list, tuple)) and len(v) > 0 and bool(v[0]):
            enabled += 1

    return 0.05 * enabled

def compute_metrics(trades: list[dict], params: Optional[Any] = None) -> Dict[str, float]:
    df = pd.DataFrame(trades)
    if df.empty:
        return {}
    
    if "rejected" in df.columns:
        df = df[df["rejected"] == False]  # noqa: E712
    if df.empty:
        return {}
    
    sort_col = None
    for c in ("exit_time", "entry_time", "datetime"):
        if c in df.columns:
            sort_col = c
            break
    if sort_col:
        df = df.sort_values(sort_col)

    # mdd = max_drawdown(df["pnl"])
    # score = expectancy - mdd * 0.5

    pnl = df["pnl"].astype(float).to_numpy()
    
    if "return_pct" in df.columns:
        rets = df["return_pct"].astype(float).to_numpy() / 100.0
    else:
        rets = pnl.copy()

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(losses.sum()) if losses.size else 0.0

    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0

    win_rate = _safe_div(wins.size, pnl.size, 0.0)
    loss_rate = 1.0 - win_rate

    expectancy = win_rate * avg_win + loss_rate * avg_loss

    profit_factor = _safe_div(gross_profit, abs(gross_loss), 10.0) if gross_loss < 0 else 10.0

    equity = pd.Series(pnl).cumsum()
    mdd = max_drawdown(equity)
    total_pnl = float(pnl.sum())

        # Risk-adjusted
    ret_mean = float(np.mean(rets)) if rets.size else 0.0
    ret_std = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
    sharpe = _safe_div(ret_mean, ret_std, 0.0) * (np.sqrt(rets.size) if rets.size > 1 else 0.0)

    downside = rets[rets < 0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = _safe_div(ret_mean, downside_std, 0.0) * (np.sqrt(rets.size) if rets.size > 1 else 0.0)

    calmar = _safe_div(total_pnl, mdd, 0.0)
    cvar_5 = _cvar(rets, 0.05)

    # Overfit guards
    stability = _chunk_stability(rets, n_chunks=4)  # меньше = лучше
    comp_pen = complexity_penalty(params)

    n = pnl.size
    sample_pen = 0.0
    if n < 30:
        sample_pen = 0.30
    elif n < 60:
        sample_pen = 0.15
    elif n < 120:
        sample_pen = 0.05

    score = (
        expectancy
        - 0.50 * mdd
        - 0.20 * abs(cvar_5) * 100.0
        - 0.15 * stability
        - comp_pen
        - sample_pen
    )

    return {
        "trades": float(n),
        "win_rate": float(win_rate),
        "profit_factor": float(profit_factor),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "expectancy": float(expectancy),
        "total_pnl": float(total_pnl),
        "max_drawdown": float(mdd),
        "sharpe_trade": float(sharpe),
        "sortino_trade": float(sortino),
        "calmar": float(calmar),
        "cvar_5": float(cvar_5),
        "stability": float(stability),
        "complexity_penalty": float(comp_pen),
        "sample_penalty": float(sample_pen),
        "score": float(score),
    }
