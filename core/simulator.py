import pandas as pd
from datetime import timedelta
from core.market_time import compute_entry_time

def simulate_trade(symbol, signal_time, params, ohlc: pd.DataFrame) -> object:
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
