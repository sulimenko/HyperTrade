from __future__ import annotations

import math
import time
from typing import Any

import numpy as np
import pandas as pd

from hypertrade.features.indicator_engine import ensure_market_data
from hypertrade.simulation.market_time import build_market_cache
from hypertrade.simulation.trade_simulator import simulate_trade


def _cvar_left_tail(values: np.ndarray, alpha: float = 0.05) -> float:
    if values.size == 0:
        return 0.0
    sorted_values = np.sort(values)
    k = max(1, int(math.floor(alpha * sorted_values.size)))
    return float(sorted_values[:k].mean())


def _max_drawdown(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    peak = np.maximum.accumulate(equity)
    return float((peak - equity).max())


def compute_metrics(trades: list[dict], params: Any | None = None) -> dict[str, float]:
    df = pd.DataFrame(trades)
    if df.empty:
        return {}
    if "rejected" in df.columns:
        df = df[df["rejected"] == False]  # noqa: E712
        if df.empty:
            return {}
    returns = pd.to_numeric(df.get("return_pct"), errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if returns.empty:
        return {}
    pnl_pct = returns.to_numpy(dtype=float)
    rets = pnl_pct / 100.0
    total_pnl = float(pnl_pct.sum())
    wins = pnl_pct[pnl_pct > 0]
    losses = pnl_pct[pnl_pct < 0]
    gross_profit = float(wins.sum()) if wins.size else 0.0
    gross_loss = float(losses.sum()) if losses.size else 0.0
    profit_factor = float(gross_profit / abs(gross_loss)) if gross_loss < 0 and abs(gross_loss) > 0 else 10.0
    equity = np.cumsum(pnl_pct)
    max_drawdown = _max_drawdown(equity)
    calmar = float(total_pnl / max_drawdown) if max_drawdown > 0 else 0.0
    cvar_5 = _cvar_left_tail(rets, 0.05)
    hold = 0.0
    if "hold_minutes" in df.columns:
        hm = pd.to_numeric(df["hold_minutes"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        hm = hm[hm >= 0]
        if not hm.empty:
            hold = float(hm.mean())
    if hold == 0.0 and "hold_bars" in df.columns:
        hb = pd.to_numeric(df["hold_bars"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        hb = hb[hb >= 0]
        if not hb.empty:
            hold = float(hb.mean() * float(int(getattr(params, "bar_minutes", 15)) if params is not None else 15))
    return {
        "trades": float(len(pnl_pct)),
        "total_pnl": total_pnl,
        "profit_factor": float(profit_factor),
        "max_drawdown": float(max_drawdown),
        "calmar": float(calmar),
        "cvar_5": float(cvar_5),
        "avg_hold_minutes": float(hold),
    }


def backtest(signals, params):
    from hypertrade.signals import LONG, SHORT, apply_signal_policy

    trades = []
    signal_stats = []
    start_time = time.time()
    filtered_signals, policy_stats = apply_signal_policy(signals, params)

    dts = [pd.Timestamp(signal["datetime"]) for signal in filtered_signals if signal.get("datetime") is not None]
    dt_min = min(dts) if dts else pd.Timestamp.utcnow()
    dt_max = max(dts) if dts else dt_min
    max_minutes = int(getattr(params, "holding_minutes", 0)) + int(getattr(params, "delay_open", 0))
    extra_days = max(60, int(max_minutes / 390) * 2 + 30)
    market_cache = build_market_cache(dt_min, dt_max, extra_days=extra_days)

    for signal, policy_stat in zip(filtered_signals, policy_stats):
        day_trades = []
        rejected = []
        for direction in ["long", "short"]:
            for symbol in signal.get(direction, []):
                ohlc = ensure_market_data(symbol, signal["datetime"], params.indicator_config)
                if ohlc is None:
                    rejected.append((symbol, "no_market_data"))
                    continue
                trade = simulate_trade(
                    symbol=symbol,
                    signal_time=signal["datetime"],
                    params=params,
                    ohlc=ohlc,
                    direction=(LONG if direction == "long" else SHORT),
                    market_cache=market_cache,
                )
                if trade.get("rejected"):
                    rejected.append((symbol, trade["reject_reason"]))
                else:
                    trades.append(trade)
                    day_trades.append(trade)
        signal_stats.append(
            {
                "datetime": signal["datetime"],
                "symbols_total": len(signal["long"]) + len(signal.get("short", [])),
                "symbols_traded": len(day_trades),
                "symbols_rejected": len(rejected),
                "total_pnl": sum(t["pnl"] for t in day_trades),
                "avg_pnl": (sum(t["pnl"] for t in day_trades) / len(day_trades)) if day_trades else 0,
                **policy_stat,
            }
        )

    print(f"Время: {(time.time() - start_time):.4f} секунд")
    return trades, signal_stats


def run_backtest(signals: list[dict], params) -> tuple[list[dict], list[dict], dict]:
    trades, signal_stats = backtest(signals, params)
    metrics = compute_metrics(trades, params)
    return trades, signal_stats, metrics
