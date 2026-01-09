import pandas as pd

from datetime import timedelta
import numba as nb
import numpy as np
from core.market_time import compute_entry_time
from core.filters import filters

# =====================
# Constants / Enums
# =====================

EXIT_REASON = {0: "sl", 1: "tp", 2: "time_exit", 3: "psar"}

DT    = 0
OPEN  = 1
HIGH  = 2
LOW   = 3
CLOSE = 4

LONG  = 1
SHORT = -1

@nb.njit
def simulate_trade_core(
    dt_ns,
    ohlc,
    entry_idx,
    entry_ts,
    holding_ns,
    direction,
    sl_pct,
    tp_pct,
    psar_enabled,
    psar_step,
    psar_max,
    slippage,
    commission
):
        # prices
    entry_open = ohlc[entry_idx, 0]
    entry_high = ohlc[entry_idx, 1]
    entry_low  = ohlc[entry_idx, 2]

    entry_price = ohlc[entry_idx, 0] * (1.0 + slippage * direction)

    if direction == LONG:
        sl_price = entry_price * (1.0 - sl_pct / 100.0)
        tp_price = entry_price * (1.0 + tp_pct / 100.0)
    else:
        sl_price = entry_price * (1.0 + sl_pct / 100.0)
        tp_price = entry_price * (1.0 - tp_pct / 100.0)

    exit_deadline = entry_ts + holding_ns

    exit_price = entry_price
    exit_idx = entry_idx
    reason = 2

    # --- PSAR state (per-trade, starts at entry) ---
    psar = 0.0
    ep = 0.0
    af = psar_step
    bull = True

    prev_low1 = entry_low
    prev_low2 = entry_low
    prev_high1 = entry_high
    prev_high2 = entry_high

    if psar_enabled:
        if direction == LONG:
            bull = True
            psar = entry_low
            ep = entry_high
        else:
            bull = False
            psar = entry_high
            ep = entry_low

    for i in range(entry_idx + 1, ohlc.shape[0]):
        ts = dt_ns[i]
        o  = ohlc[i, 0]
        h  = ohlc[i, 1]
        l  = ohlc[i, 2]
        c  = ohlc[i, 3]

        # ---------- TIME EXIT ----------
        if ts > exit_deadline:
            exit_price = o
            exit_idx = i
            reason = 2
            break

        # ---------- UPDATE PSAR ----------
        if psar_enabled:
            # PSAR base update
            psar = psar + af * (ep - psar)

            if bull:
                # constrain below last two lows
                if psar > prev_low1:
                    psar = prev_low1
                if psar > prev_low2:
                    psar = prev_low2

                # reversal
                if l <= psar:
                    bull = False
                    psar = ep
                    ep = l
                    af = psar_step
                    # exit this bar at stop level (gap-aware)
                    exit_price = min(psar, o)
                    exit_idx = i
                    reason = 3
                    break
                else:
                    # update EP/AF
                    if h > ep:
                        ep = h
                        af = af + psar_step
                        if af > psar_max:
                            af = psar_max
            else:
                # bear: constrain above last two highs
                if psar < prev_high1:
                    psar = prev_high1
                if psar < prev_high2:
                    psar = prev_high2

                if h >= psar:
                    bull = True
                    psar = ep
                    ep = h
                    af = psar_step
                    exit_price = max(psar, o)
                    exit_idx = i
                    reason = 3
                    break
                else:
                    if l < ep:
                        ep = l
                        af = af + psar_step
                        if af > psar_max:
                            af = psar_max

        # ---------- DIRECTIONAL CANDLE MODEL ----------
        bullish = c >= o

        if direction == LONG:
            if bullish:
                # open → low → high
                if l <= sl_price:
                    exit_price = min(sl_price, o)
                    exit_idx = i
                    reason = 0
                    break
                if h >= tp_price:
                    exit_price = max(tp_price, o)
                    exit_idx = i
                    reason = 1
                    break
            else:
                # open → high → low
                if h >= tp_price:
                    exit_price = max(tp_price, o)
                    exit_idx = i
                    reason = 1
                    break
                if l <= sl_price:
                    exit_price = min(sl_price, o)
                    exit_idx = i
                    reason = 0
                    break

        else:  # ---------- SHORT ----------
            if bullish:
                # open → low → high
                if l <= tp_price:
                    exit_price = min(tp_price, o)
                    exit_idx = i
                    reason = 1
                    break
                if h >= sl_price:
                    exit_price = max(sl_price, o)
                    exit_idx = i
                    reason = 0
                    break
            else:
                # open → high → low
                if h >= sl_price:
                    exit_price = max(sl_price, o)
                    exit_idx = i
                    reason = 0
                    break
                if l <= tp_price:
                    exit_price = min(tp_price, o)
                    exit_idx = i
                    reason = 1
                    break
        
        # update prev extrema for PSAR constraints
        prev_low2 = prev_low1
        prev_low1 = l
        prev_high2 = prev_high1
        prev_high1 = h

    # ---------- APPLY SLIPPAGE + COMMISSION ----------
    exit_price = exit_price * (1.0 - slippage * direction)

    pnl = (exit_price - entry_price) * direction
    pnl -= commission * 2

    return pnl, entry_price, exit_price, exit_idx, reason

def simulate_trade(symbol, signal_time, params, ohlc, direction=LONG):
    try:
        
        entry_dt = compute_entry_time(signal_time,params.delay_open)
        entry_idx = ohlc["datetime"].searchsorted(entry_dt)

        if entry_idx >= len(ohlc):
            return {"symbol": symbol, "rejected": True, "reject_reason": "no_candles_after_entry"}
        
        if not filters(ohlc.iloc[entry_idx], params):
            return {"symbol": symbol, "rejected": True, "reject_reason": "indicators_filter_failed"}

        dt_ns = ohlc["datetime"].values.astype("datetime64[ns]").astype(np.int64)

        ohlc_np = np.column_stack([
            ohlc["open"].values,
            ohlc["high"].values,
            ohlc["low"].values,
            ohlc["close"].values,
        ]).astype(np.float64)

        entry_ts = np.int64(entry_dt.value)
        holding_ns = np.int64(params.holding_minutes * 60_000_000_000)
        
        pnl, entry_price, exit_price, exit_idx, reason = simulate_trade_core(
            dt_ns,
            ohlc_np,
            entry_idx,
            entry_ts,
            holding_ns,
            direction,
            params.sl,
            params.tp,
            params.psar_enabled,
            params.psar_step,
            params.psar_max,
            params.slippage,
            params.commission
        )

        exit_dt = ohlc.iloc[exit_idx]["datetime"]

        return_pct = pnl / entry_price * 100

        return {
            "symbol": symbol,
            "direction": int(direction),
            "entry_dt": entry_dt,
            "exit_dt": exit_dt,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pnl": float(pnl),
            "return_pct": float(return_pct),
            "is_win": pnl > 0,
            "exit_reason": EXIT_REASON[reason],
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
