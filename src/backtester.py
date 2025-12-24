import numpy as np
import pandas as pd


def evaluate_strategy(
    ohlc: pd.DataFrame,
    signals: pd.DataFrame,
    entry_delay: int,
    sl_pct: float,
    tp_pct: float,
    max_holding: int,
    commission: float = 0.0005,
    slippage: float = 0.0002,
    filters_fn=None
):
    equity = 1.0
    trades = []

    for _, sig in signals.iterrows():
        signal_time = sig["signal_time"]
        entry_time = signal_time + pd.Timedelta(minutes=15 * entry_delay)

        try:
            entry_idx = ohlc.index.get_indexer(
                [entry_time],
                method="bfill"
            )[0]
        except Exception:
            continue

        if entry_idx == -1:
            continue

        entry_row = ohlc.iloc[entry_idx]

        if filters_fn and not filters_fn(entry_row.name, ohlc):
            continue

        entry_price = entry_row["open"] * (1 + slippage)
        sl_price = entry_price * (1 - sl_pct)
        tp_price = entry_price * (1 + tp_pct)

        exit_price = None

        for i in range(max_holding):
            idx = entry_idx + i
            if idx >= len(ohlc):
                break

            row = ohlc.iloc[idx]

            if row["low"] <= sl_price:
                exit_price = sl_price * (1 - slippage)
                break

            if row["high"] >= tp_price:
                exit_price = tp_price * (1 + slippage)
                break

        if exit_price is None:
            exit_price = ohlc.iloc[min(
                entry_idx + max_holding - 1,
                len(ohlc) - 1
            )]["close"]

        ret = (exit_price / entry_price) - 1
        ret -= commission * 2

        equity *= (1 + ret)
        trades.append(ret)

    if not trades:
        return {
            "sharpe": -999,
            "return": 0,
            "trades": 0
        }

    returns = np.array(trades)
    sharpe = returns.mean() / (returns.std() + 1e-9)

    return {
        "sharpe": sharpe,
        "return": equity - 1,
        "trades": len(trades)
    }
