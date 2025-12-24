import numpy as np
import pandas as pd


def evaluate_strategy(
    ohlc: pd.DataFrame,
    signals: pd.DataFrame,
    entry_delay: int,
    sl: float,
    tp: float,
    holding_minutes: int,
    commission: float = 0.001,
    slippage: float = 0.0002,
    filters_fn=None
):
    equity = 1.0
    trades = []

    holding_delta = pd.Timedelta(minutes=holding_minutes)

    for signal_time in signals.index:
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

        entry = ohlc.iloc[entry_idx]

        if filters_fn and not filters_fn(entry):
            continue

        entry_price = entry["open"] * (1 + slippage)
        sl_price = entry["open"] * (1 - sl)
        tp_price = entry["open"] * (1 + tp)

        expiry_time = entry.name + holding_delta
        exit_price = None
        exit_time = None

        # print(f"Entry at {entry.name}, price: {entry_price:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")
        for idx in range(entry_idx + 1, len(ohlc)):
            row = ohlc.iloc[idx]
            current_time = row.name

            if current_time > expiry_time:
                exit_price = ohlc.iloc[idx - 1]["close"]
                exit_time = current_time
                break

            if row["low"] <= sl_price:
                exit_price = sl_price * (1 - slippage)
                exit_time = current_time
                break

            if row["high"] >= tp_price:
                exit_price = tp_price * (1 + slippage)
                exit_time = current_time
                break

        if exit_price is None:
            last_row = ohlc.iloc[-1]
            exit_price = last_row["close"]
            exit_time = last_row.name

        # print(f"Exit at {current_time}, price: {exit_price:.2f}, percent: {(exit_price/entry_price - 1) * 100:.2f} %")
        ret = (exit_price / entry_price) - 1
        ret -= commission * 2

        equity *= (1 + ret)
        trades.append(ret)

    if not trades:
        return {
            "score": -999,
            "return": 0,
            "trades": 0
        }

    returns = np.array(trades)
    score = returns.mean() / (returns.std() + 1e-9)

    return {
        "score": score,
        "return": equity - 1,
        "trades": len(trades)
    }
