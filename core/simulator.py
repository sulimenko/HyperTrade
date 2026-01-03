import pandas as pd

from datetime import timedelta
import numba as nb
import numpy as np
from core.market_time import compute_entry_time
from core.filters import filters


@nb.njit
def simulate_trade_core(
    open,
    high,
    low,
    close,
    entry_idx,
    sl_pct,
    tp_pct,
    max_bars
):
    entry_price = open[entry_idx]

    sl_price = entry_price * (1.0 - sl_pct / 100.0)
    tp_price = entry_price * (1.0 + tp_pct / 100.0)

    for i in range(entry_idx + 1, min(entry_idx + max_bars, len(close))):
        if low[i] <= sl_price:
            return sl_price, i, "sl"   # SL

        if high[i] >= tp_price:
            return tp_price, i, "tp"   # TP

    exit_idx = min(entry_idx + max_bars, len(close) - 1)
    return open[exit_idx], exit_idx, "time_exit"

def simulate_trade(symbol, signal_time, params, ohlc):
    entry = {}
    try:
        entry["datetime"] = compute_entry_time(
            signal_time,
            params.delay_open
        )

        mask = ohlc["datetime"] >= entry["datetime"]
        if not mask.any():
            return {
                "symbol": symbol,
                "rejected": True,
                "reject_reason": "no_candles_after_entry"
            }

        entry["idx"] = mask.idxmax()

        if not filters(ohlc.iloc[entry["idx"]], params):
            return {
                "symbol": symbol,
                "rejected": True,
                "reject_reason": "indicators_filter_failed"
            }

        max_bars = params.holding_minutes // params.bar_minutes

        exit_price, exit_idx, exit_reason = simulate_trade_core(
            ohlc["open"].values,
            ohlc["high"].values,
            ohlc["low"].values,
            ohlc["close"].values,
            entry["idx"],
            params.sl,
            params.tp,
            max_bars
        )

        entry["price"] = ohlc.iloc[entry["idx"]]["open"]
        exit_datetime = ohlc.iloc[exit_idx]["datetime"]

        entry["price"] *= (1 + params.slippage)
        exit_price  *= (1 - params.slippage)

        commission = params.commission * 2

        pnl = (exit_price - entry["price"]) - commission
        return_pct = pnl / entry["price"] * 100

        return {
            "symbol": symbol,
            "entry_dt": entry["datetime"],
            "exit_dt": exit_datetime,
            "entry_price": entry["price"],
            "exit_price": exit_price,
            "pnl": pnl,
            "return_pct": return_pct,
            "is_win": pnl > 0,
            "exit_reason": exit_reason,
            "rejected": False
        }

    except Exception as e:
        return {
            "symbol": symbol,
            "rejected": True,
            "reject_reason": str(e)
        }



def simulate_trade_old(symbol, signal_time, params, ohlc: pd.DataFrame) -> object:
    try:
        entry_dt = compute_entry_time(
            signal_dt = signal_time,
            delay_minutes = params["delay_open"]
        )

        chart = ohlc[ohlc["datetime"] >= entry_dt]
        if chart.empty:
            return {
                "symbol": symbol,
                "rejected": True,
                "reject_reason": "no_candles_after_entry"
            }

        entry_price = chart.iloc[0]["open"]

        sl_price = entry_price * (1 - params["sl"] / 100)
        tp_price = entry_price * (1 + params["tp"] / 100)

        exit_price = entry_price
        exit_dt = chart.iloc[0]["datetime"]
        timeout = entry_dt + timedelta(minutes=params["holding_minutes"])
        exit_reason = "time_exit"

        for _, row in chart.iterrows():
            if row["datetime"] > timeout:
                exit_price = row["open"]
                exit_dt = row["datetime"]
                break

            if row["low"] <= sl_price:
                exit_price = sl_price
                exit_dt = row["datetime"]
                exit_reason = "sl"
                break

            if row["high"] >= tp_price:
                exit_price = tp_price
                exit_dt = row["datetime"]
                exit_reason = "tp"
                break

        commission = params["commission"]
        slippage = params["slippage"]

        entry_price *= (1 + slippage)
        exit_price  *= (1 - slippage)

        gross_pnl = exit_price - entry_price
        # fees = (entry_price + exit_price) - commission

        pnl = gross_pnl - commission * 2
        return_pct = pnl / entry_price * 100

        return {
            "symbol": symbol,
            "entry_dt": entry_dt,
            "exit_dt": exit_dt,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "return_pct": return_pct,
            "is_win": pnl > 0,
            "exit_reason": exit_reason,
            "rejected": False
        }
    
    except Exception as e:
        return {
            "symbol": symbol,
            "rejected": True,
            "reject_reason": str(e)
        }
