import pandas as pd

import numba as nb
import numpy as np
from core.market_time import compute_entry_time, add_market_minutes
from core.market_time import compute_entry_time_cached, add_market_minutes_cached
from core.filters import filters

# =====================
# Constants / Enums
# =====================

EXIT_REASON = {0: "sl", 1: "tp", 2: "time_exit", 3: "psar", 4: "ts"}

LONG = 1
SHORT = -1


@nb.njit
def _check_time_exit(ts, exit_deadline, o):
    if ts >= exit_deadline:
        return True, o, 2
    return False, 0.0, -1

@nb.njit
def _check_sl_tp(direction, bullish, o, h, l, sl_price, tp_price):
    # returns: hit, exit_price, reason
    if direction == LONG:
        if bullish:
            # open → low → high
            if l <= sl_price:
                return True, min(sl_price, o), 0
            if h >= tp_price:
                return True, max(tp_price, o), 1
        else:
            # open → high → low
            if h >= tp_price:
                return True, max(tp_price, o), 1
            if l <= sl_price:
                return True, min(sl_price, o), 0
    else:
        # SHORT
        if bullish:
            # open → low → high
            if l <= tp_price:
                return True, min(tp_price, o), 1
            if h >= sl_price:
                return True, max(sl_price, o), 0
        else:
            # open → high → low
            if h >= sl_price:
                return True, max(sl_price, o), 0
            if l <= tp_price:
                return True, min(tp_price, o), 1

    return False, 0.0, -1

@nb.njit
def _ts_init(direction, entry_price, ts_dist):
    # returns initial trailing stop level + best extreme
    if direction == LONG:
        best = entry_price
        trail = entry_price * (1.0 - ts_dist / 100.0)
    else:
        best = entry_price
        trail = entry_price * (1.0 + ts_dist / 100.0)
    return trail, best

@nb.njit
def _ts_update_and_check(direction, o, h, l, ts_dist, trail, best):
    if direction == LONG:
        if h > best:
            best = h
        new_trail = best * (1.0 - ts_dist / 100.0)
        if new_trail > trail:
            trail = new_trail

        if l <= trail:
            return True, best, trail, min(trail, o), 4
        return False, best, trail, 0.0, -1
    else:
        if l < best:
            best = l
        new_trail = best * (1.0 + ts_dist / 100.0)
        if new_trail < trail:
            trail = new_trail

        if h >= trail:
            return True, best, trail, max(trail, o), 4
        return False, best, trail, 0.0, -1
    
@nb.njit
def _psar_init(direction, entry_high, entry_low, psar_step):
    psar = 0.0
    ep = 0.0
    af = psar_step
    bull = True

    if direction == LONG:
        bull = True
        psar = entry_low
        ep = entry_high
    else:
        bull = False
        psar = entry_high
        ep = entry_low

    return psar, ep, af, bull


@nb.njit
def _psar_update_and_check(
    direction,
    o, h, l,
    psar, ep, af, bull,
    prev_low1, prev_low2,
    prev_high1, prev_high2,
    psar_step, psar_max
):
    psar = psar + af * (ep - psar)

    if bull:
        if psar > prev_low1:
            psar = prev_low1
        if psar > prev_low2:
            psar = prev_low2

        if l <= psar:
            bull = False
            psar = ep
            ep = l
            af = psar_step
            return True, psar, ep, af, bull, min(psar, o), 3
        else:
            if h > ep:
                ep = h
                af = af + psar_step
                if af > psar_max:
                    af = psar_max
    else:
        if psar < prev_high1:
            psar = prev_high1
        if psar < prev_high2:
            psar = prev_high2

        if h >= psar:
            bull = True
            psar = ep
            ep = h
            af = psar_step
            return True, psar, ep, af, bull, max(psar, o), 3
        else:
            if l < ep:
                ep = l
                af = af + psar_step
                if af > psar_max:
                    af = psar_max

    return False, psar, ep, af, bull, 0.0, -1

@nb.njit
def simulate_trade_core(
    dt_ns,
    ohlc,
    entry_idx,
    exit_deadline_ts,
    direction,
    sl_pct,
    tp_pct,
    psar_enabled,
    psar_step,
    psar_max,
    ts_enabled,
    ts_dist,
    slippage,
    commission
):
    entry_high = ohlc[entry_idx, 1]
    entry_low  = ohlc[entry_idx, 2]

    entry_price = ohlc[entry_idx, 0] * (1.0 + slippage * direction)

    if direction == LONG:
        sl_price = entry_price * (1.0 - sl_pct / 100.0)
        tp_price = entry_price * (1.0 + tp_pct / 100.0)
    else:
        sl_price = entry_price * (1.0 + sl_pct / 100.0)
        tp_price = entry_price * (1.0 - tp_pct / 100.0)

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
        psar, ep, af, bull = _psar_init(direction, entry_high, entry_low, psar_step)

    # --- Trailing Stop state ---
    trail = 0.0
    best = 0.0
    if ts_enabled:
        trail, best = _ts_init(direction, entry_price, ts_dist)

    had_exit = False
    for i in range(entry_idx + 1, ohlc.shape[0]):
        ts = dt_ns[i]
        o  = ohlc[i, 0]
        h  = ohlc[i, 1]
        l  = ohlc[i, 2]
        c  = ohlc[i, 3]

        # ---------- TIME EXIT ----------
        hit, px, rsn = _check_time_exit(ts, exit_deadline_ts, o)
        if hit:
            exit_price = px
            exit_idx = i
            reason = rsn
            had_exit = True
            break

        # ---------- PSAR ----------
        if psar_enabled:
            hit, psar, ep, af, bull, px, rsn = _psar_update_and_check(
                direction, o, h, l,
                psar, ep, af, bull,
                prev_low1, prev_low2,
                prev_high1, prev_high2,
                psar_step, psar_max
            )
            if hit:
                exit_price = px
                exit_idx = i
                reason = rsn
                had_exit = True
                break

        # ---------- Trailing Stop ----------
        if ts_enabled:
            hit, best, trail, px, rsn = _ts_update_and_check(direction, o, h, l, ts_dist, trail, best)
            if hit:
                exit_price = px
                exit_idx = i
                reason = rsn
                had_exit = True
                break

        # ---------- SL/TP candle model ----------
        bullish = c >= o
        hit, px, rsn = _check_sl_tp(direction, bullish, o, h, l, sl_price, tp_price)
        if hit:
            exit_price = px
            exit_idx = i
            reason = rsn
            had_exit = True
            break

        # update PSAR constraints
        prev_low2 = prev_low1
        prev_low1 = l
        prev_high2 = prev_high1
        prev_high1 = h

    if not had_exit:
        exit_idx = ohlc.shape[0] - 1
        exit_price = ohlc[exit_idx, 3]
        reason = 2

    # ---------- APPLY SLIPPAGE + COMMISSION ----------
    exit_price = exit_price * (1.0 - slippage * direction)

    pnl = (exit_price - entry_price) * direction
    pnl -= commission * 2

    return pnl, entry_price, exit_price, exit_idx, reason

def simulate_trade(symbol, signal_time, params, ohlc, direction=LONG, market_cache=None):
    try:
        if market_cache is not None:
            entry_dt = compute_entry_time_cached(signal_time, params.delay_open, market_cache)
        else:
            entry_dt = compute_entry_time(signal_time, params.delay_open)

        entry_idx = ohlc["datetime"].searchsorted(entry_dt)

        if entry_idx >= len(ohlc):
            return {"symbol": symbol, "rejected": True, "reject_reason": "no_candles_after_entry"}

        # if not filters(ohlc.iloc[entry_idx], params):
        if not filters(ohlc, entry_idx, params, direction):
            return {"symbol": symbol, "rejected": True, "reject_reason": "indicators_filter_failed"}

        dt_ns = ohlc["datetime"].values.astype("datetime64[ns]").astype(np.int64)

        if market_cache is not None:
            entry_ns = int(pd.Timestamp(entry_dt).tz_localize("UTC").value) if pd.Timestamp(entry_dt).tzinfo is None else int(pd.Timestamp(entry_dt).tz_convert("UTC").value)
            exit_deadline_ts = add_market_minutes_cached(entry_ns, int(params.holding_minutes), market_cache)
            exit_deadline_ts = np.int64(exit_deadline_ts)
        else:
            entry_ts_utc = pd.Timestamp(entry_dt)
            exit_deadline_dt = add_market_minutes(entry_ts_utc, int(params.holding_minutes))
            exit_deadline_ts = np.int64(pd.Timestamp(exit_deadline_dt).value)

        ohlc_np = np.column_stack([
            ohlc["open"].values,
            ohlc["high"].values,
            ohlc["low"].values,
            ohlc["close"].values,
        ]).astype(np.float64)

        entry_open = float(ohlc.iloc[entry_idx]["open"])
        entry_price_for_levels = entry_open * (1.0 + float(params.slippage) * float(direction))

        if bool(getattr(params, "atr_use", False)):
            # ATR обязателен
            atr_period = int(getattr(params, "atr_period", 14))
            atr_col = f"atr_{atr_period}"
            if atr_col not in ohlc.columns:
                return {"symbol": symbol, "rejected": True, "reject_reason": f"missing_{atr_col}"}

            atr_v = float(ohlc.iloc[entry_idx][atr_col])
            if not np.isfinite(atr_v) or atr_v <= 0:
                return {"symbol": symbol, "rejected": True, "reject_reason": "atr_nan_or_zero"}

            atr_sl = float(getattr(params, "atr_sl", 1.0))
            atr_tp = float(getattr(params, "atr_tp", 1.0))

            # ATR-only уровни (работают без SR)
            sl_price = entry_price_for_levels - float(direction) * atr_sl * atr_v
            tp_price = entry_price_for_levels + float(direction) * atr_tp * atr_v

            # Если Donchian включен — используем как SR-ограничитель уровней (опционально)
            don_cfg = getattr(params, "indicator_config", {}).get("donchian")
            if don_cfg and len(don_cfg) >= 2 and bool(don_cfg[0]):
                _, period = don_cfg[:2]
                hcol = f"don_h_{int(period)}"
                lcol = f"don_l_{int(period)}"
                if hcol in ohlc.columns and lcol in ohlc.columns:
                    don_h = ohlc.iloc[entry_idx][hcol]
                    don_l = ohlc.iloc[entry_idx][lcol]
                    if pd.notna(don_h) and pd.notna(don_l):
                        pad = 0.002
                        don_h = float(don_h)
                        don_l = float(don_l)

                        if direction == LONG:
                            # SL дальше (ниже) из {ATR, support}; TP ближе (ниже) к resistance
                            sl_price = min(sl_price, don_l * (1.0 - pad))
                            tp_price = min(tp_price, don_h * (1.0 - pad))
                        else:
                            # SHORT: SL выше, TP ниже
                            sl_price = max(sl_price, don_h * (1.0 + pad))
                            tp_price = max(tp_price, don_l * (1.0 + pad))

            # Переводим абсолютные уровни в проценты для simulate_trade_core
            sl_pct = abs((entry_price_for_levels - sl_price) / entry_price_for_levels) * 100.0
            tp_pct = abs((tp_price - entry_price_for_levels) / entry_price_for_levels) * 100.0

        else:
            # Старый режим: % SL/TP
            sl_pct = float(params.sl)
            tp_pct = float(params.tp)

        pnl, entry_price, exit_price, exit_idx, reason = simulate_trade_core(
            dt_ns,
            ohlc_np,
            entry_idx,
            exit_deadline_ts,
            direction,
            sl_pct,
            tp_pct,
            psar_enabled=bool(getattr(params, "psar_enabled", False)),
            psar_step=float(getattr(params, "psar_step", 0.005) or 0.005),
            psar_max=float(getattr(params, "psar_max", 0.1) or 0.1),
            ts_enabled=bool(getattr(params, "ts_enabled", False)),
            ts_dist=float(getattr(params, "ts_dist", 1.0) or 1.0),
            slippage=float(getattr(params, "slippage", 0.0)),
            commission=float(getattr(params, "commission", 0.0))
        )

        exit_dt = ohlc.iloc[exit_idx]["datetime"]
        return_pct = pnl / entry_price * 100

        hold_bars = int(exit_idx - entry_idx) if exit_idx >= entry_idx else 0
        bar_minutes = int(getattr(params, "bar_minutes", 15))
        hold_minutes = float(hold_bars * bar_minutes)

        return {
            "symbol": symbol,
            "direction": int(direction),
            "entry_dt": entry_dt,
            "exit_dt": exit_dt,
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "pnl": float(pnl),
            "return_pct": float(return_pct),
            "hold_bars": float(hold_bars),
            "hold_minutes": float(hold_minutes),
            "is_win": pnl > 0,
            "exit_reason": EXIT_REASON[reason],
            "rejected": False
        }

    except Exception as e:
        return {"symbol": symbol, "rejected": True, "reject_reason": str(e)}