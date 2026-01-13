from typing import Optional, Any, Dict, Literal
import math

import numpy as np
import pandas as pd


# ---------------------------
# Helpers
# ---------------------------

def max_drawdown(equity: pd.Series) -> float:
    """Максимальная просадка по кривой equity (в тех же единицах, что equity)."""
    if equity.empty:
        return 0.0
    peak = equity.cummax()
    dd = equity - peak
    return float(-dd.min())


def _is_bad(x: Any) -> bool:
    try:
        return x is None or (isinstance(x, float) and math.isnan(x))
    except Exception:
        return x is None


def _safe_float(x: Any, default: float = 0.0) -> float:
    if _is_bad(x):
        return float(default)
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x: Any, default: int = 0) -> int:
    if _is_bad(x):
        return int(default)
    try:
        return int(x)
    except Exception:
        return int(default)


def _safe_div(a: float, b: float, default: float = 0.0) -> float:
    a = _safe_float(a, default=0.0)
    b = _safe_float(b, default=0.0)
    if b == 0.0:
        return float(default)
    return float(a / b)


def clamp(x: float, lo: float, hi: float) -> float:
    x = _safe_float(x, default=lo)
    return float(max(lo, min(hi, x)))


def log1p_pos(x: float) -> float:
    """log(1+x) but safe for x<=0 -> 0"""
    x = _safe_float(x, default=0.0)
    return float(math.log1p(max(0.0, x)))


def _cvar_left_tail(x: np.ndarray, alpha: float = 0.05) -> float:
    """
    CVaR (Expected Shortfall) по левому хвосту.
    Возвращает среднее худших alpha-доли значений.
    """
    if x.size == 0:
        return 0.0
    x_sorted = np.sort(x)
    k = max(1, int(np.floor(alpha * x_sorted.size)))
    return float(x_sorted[:k].mean())


def _chunk_instability(x: np.ndarray, n_chunks: int = 4) -> float:
    """
    Нестабильность среднего по чанкам:
      std(chunk_means) / (abs(global_mean) + eps).
    Меньше = лучше.
    """
    if x.size < n_chunks * 5:
        return 1.0  # мало данных → считаем нестабильным

    chunks = np.array_split(x, n_chunks)
    means = np.array([c.mean() for c in chunks if c.size > 0], dtype=float)
    if means.size < 2:
        return 1.0
    global_mean = float(x.mean())
    eps = 1e-9
    return float(means.std(ddof=1) / (abs(global_mean) + eps))


def complexity_penalty(params: Optional[Any]) -> float:
    """
    Штраф за сложность (простая защита от переобучения): число включённых фильтров.
    Возвращает число в "баллах" (обычно маленькое).
    """
    if params is None:
        return 0.0
    cfg = getattr(params, "indicator_config", None)
    if not isinstance(cfg, dict):
        return 0.0

    enabled = 0
    for _, v in cfg.items():
        if isinstance(v, (list, tuple)) and len(v) > 0 and bool(v[0]):
            enabled += 1

    # Подбери коэффициент под масштаб твоего score.
    return 0.05 * enabled


def sample_penalty(n_trades: int) -> float:
    """
    Штраф за слишком маленькую выборку сделок, чтобы Optuna не “влюблялась”
    в случайные удачные конфиги.
    """
    n = int(n_trades)
    if n < 20:
        return 0.30
    if n < 40:
        return 0.15
    if n < 80:
        return 0.05
    return 0.0

def _compute_avg_hold_minutes(df: pd.DataFrame) -> float:
    if "hold_minutes" in df.columns:
        x = pd.to_numeric(df["hold_minutes"], errors="coerce")
        x = x.replace([np.inf, -np.inf], np.nan).dropna()
        x = x[x >= 0]
        return float(x.mean()) if len(x) else 0.0

    if "entry_dt" in df.columns and "exit_dt" in df.columns:
        entry = pd.to_datetime(df["entry_dt"], errors="coerce")
        exit_ = pd.to_datetime(df["exit_dt"], errors="coerce")
        dt = (exit_ - entry).dt.total_seconds() / 60.0
        dt = dt.replace([np.inf, -np.inf], np.nan).dropna()
        return float(dt.mean()) if len(dt) else 0.0

    return 0.0


# ---------------------------
# Variant A Objective
# ---------------------------

def objective_variant_a(
    summary: Dict[str, float],
    *,
    trades_target: int = 800,
    gates: Optional[Dict[str, float]] = None,
) -> float:
    """
    Risk-adjusted objective (Variant A).

    trades_target: "нормальное" число сделок; используется мягкий симметричный штраф.
    gates: опциональные пороги для hard-gates, напр:
      {
        "min_total_pnl": 0.0,
        "max_max_drawdown": 1e9,
        "min_trades": 30,
      }
    """
    gates = gates or {}
    total_pnl = _safe_float(summary.get("total_pnl", 0.0), 0.0)
    max_dd    = _safe_float(summary.get("max_drawdown", 0.0), 0.0)
    calmar    = _safe_float(summary.get("calmar", 0.0), 0.0)
    sharpe_t  = _safe_float(summary.get("sharpe_trade", 0.0), 0.0)
    sortino_t = _safe_float(summary.get("sortino_trade", 0.0), 0.0)
    cvar_5    = _safe_float(summary.get("cvar_5", 0.0), 0.0)
    instability = _safe_float(summary.get("instability", 0.0), 0.0)
    trades    = _safe_int(summary.get("trades", 0), 0)
    avg_hold = _safe_float(summary.get("avg_hold_minutes", 0.0), 0.0)
    expectancy = _safe_float(summary.get("expectancy", 0.0), 0.0)
    profit_factor = _safe_float(summary.get("profit_factor", 0.0), 0.0)

    # --- Hard gates ---
    min_total_pnl = _safe_float(gates.get("min_total_pnl", 0.0), 0.0)
    min_trades    = int(gates.get("min_trades", 1))
    max_mdd_gate  = _safe_float(gates.get("max_max_drawdown", float("inf")), float("inf"))

    if total_pnl <= min_total_pnl:
        return -1e9
    if trades < min_trades:
        return -1e9
    if max_dd > max_mdd_gate:
        return -1e9

    stability_good = clamp(1.0 - instability, 0.0, 1.0)

    # CVaR может быть отрицательным (левый хвост), для штрафа берём модуль.
    cvar_pen = abs(cvar_5)

    score = 0.0
    score += 1.25 * log1p_pos(total_pnl)
    score += 0.50 * log1p_pos(max(0.0, calmar))

    # качество входа/фильтра
    score += 0.70 * log1p_pos(max(0.0, expectancy))
    score += 0.25 * clamp(profit_factor, 0.0, 5.0)

    score += 0.4 * clamp(sharpe_t,  -2.0, 4.0)
    score += 0.4 * clamp(sortino_t, -2.0, 6.0)
    score += 0.45 * stability_good

    score -= 0.70 * log1p_pos(max_dd)
    score -= 0.60 * log1p_pos(cvar_pen)

    # Мягкий контроль trades вокруг таргета
    score -= 0.25 * abs(math.log1p(trades) - math.log1p(max(1, trades_target)))

    # k подбирается под масштаб score. Стартовое значение: 0.20–0.60 обычно норм.
    k_hold = _safe_float(gates.get("k_hold", 0.35), 0.35)
    score -= k_hold * math.log1p(max(0.0, avg_hold))

    return float(score)


# ---------------------------
# Metrics computation
# ---------------------------

ObjectiveName = Literal["legacy", "variant_a"]

def compute_metrics(
    trades: list[dict],
    params: Optional[Any] = None,
    *,
    objective: ObjectiveName = "variant_a",
    trades_target: int = 800,
    objective_gates: Optional[Dict[str, float]] = None,
    delay_penalty_k: float = 0.005,
) -> Dict[str, float]:
    """
    Возвращает summary-метрики + score.
    delay_penalty_k: мягко предпочесть delay_open=0, но не запрещать.
    """
    df = pd.DataFrame(trades)
    if df.empty:
        return {}

    # фильтр rejected (если есть)
    if "rejected" in df.columns:
        df = df[df["rejected"] == False]  # noqa: E712
    if df.empty:
        return {}

    # сортировка по времени для корректной equity/стабильности
    sort_col = None
    for col in ("exit_dt", "entry_dt"):
        if col in df.columns:
            sort_col = col
            break
    if sort_col:
        df = df.sort_values(sort_col)

    pnl = pd.to_numeric(df.get("pnl"), errors="coerce")
    pnl = pnl.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    n = int(pnl.size)
    if n == 0:
        return {}
    
    total_pnl = float(pnl.sum())

    # Доходности: prefer return_pct, иначе pnl как прокси (не идеально, но лучше чем ничего)
    if "return_pct" in df.columns:
        rets = pd.to_numeric(df["return_pct"], errors="coerce")
        rets = rets.replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float) / 100.0
        if rets.size != pnl.size:
            rets = pnl.copy()
    else:
        rets = pnl.copy()

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]

    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(losses.sum()) if losses.size else 0.0

    avg_win = float(wins.mean()) if wins.size else 0.0
    avg_loss = float(losses.mean()) if losses.size else 0.0

    win_rate = _safe_div(wins.size, n, 0.0)
    loss_rate = 1.0 - win_rate

    expectancy = win_rate * avg_win + loss_rate * avg_loss

    # Profit factor: если нет лоссов → capped
    if gross_loss < 0:
        profit_factor = _safe_div(gross_profit, abs(gross_loss), 10.0)
    else:
        profit_factor = 10.0

    equity = pd.Series(pnl).cumsum()
    mdd = max_drawdown(equity)

    # Risk-adjusted (на trade returns)
    ret_mean = float(np.mean(rets)) if rets.size else 0.0
    ret_std = float(np.std(rets, ddof=1)) if rets.size > 1 else 0.0
    sharpe = _safe_div(ret_mean, ret_std, 0.0) * (float(np.sqrt(rets.size)) if rets.size > 1 else 0.0)

    downside = rets[rets < 0]
    downside_std = float(np.std(downside, ddof=1)) if downside.size > 1 else 0.0
    sortino = _safe_div(ret_mean, downside_std, 0.0) * (float(np.sqrt(rets.size)) if rets.size > 1 else 0.0)

    calmar = _safe_div(total_pnl, mdd, 0.0)
    cvar_5 = _cvar_left_tail(rets, 0.05) # CVaR на левом хвосте returns (обычно будет отрицательный)

    # Overfit guards
    instability = _chunk_instability(rets, n_chunks=4)  # меньше = лучше
    comp_pen = complexity_penalty(params)
    samp_pen = sample_penalty(n)

    avg_hold_minutes = _compute_avg_hold_minutes(df)

    # Собираем summary
    summary: Dict[str, float] = {
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
        "instability": float(instability),
        "avg_hold_minutes": float(avg_hold_minutes),
        "complexity_penalty": float(comp_pen),
        "sample_penalty": float(samp_pen),
    }

    # --- Score ---
    if objective == "legacy":
        score = (
            expectancy
            - 0.50 * mdd
            - 0.20 * abs(cvar_5) * 100.0
            - 0.15 * instability
            - comp_pen
            - samp_pen
        )
    else:
        # Variant A (боевой), плюс штрафы за сложность/малую выборку прямо здесь
        score = objective_variant_a(
            summary,
            trades_target=trades_target,
            gates=objective_gates,
        )
        score -= comp_pen
        score -= samp_pen

    if delay_penalty_k and params is not None:
        delay_open = int(getattr(params, "delay_open", 0))
        score -= float(delay_penalty_k) * float(delay_open)

    summary["score"] = float(score)
    return summary
