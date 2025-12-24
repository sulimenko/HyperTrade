import numpy as np
import pandas as pd


def evaluate_strategy(
    ohlc: pd.DataFrame,
    signals: pd.DataFrame,
    entry_delay: int,
    sl: float,
    tp: float,
    holding_minutes: int,
    commission: float = 0.0005,
    slippage: float = 0.0002,
    filters_fn=None
):
    equity = 1.0
    trades = []

    holding_delta = pd.Timedelta(minutes=holding_minutes)

    for _, sig in signals.iterrows():
        signal_time = pd.Timestamp(sig["signal_time"]).tz_localize('UTC')
        entry_time = signal_time + pd.Timedelta(minutes=15 * entry_delay)
        expiry_time = entry_time + holding_delta

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
        sl_price = entry_row["open"] * (1 - sl)
        tp_price = entry_row["open"] * (1 + tp)

        exit_price = None

        print(f"Entry at {entry_row.name}, price: {entry_price:.2f}, SL: {sl_price:.2f}, TP: {tp_price:.2f}")
        for i in range(len(ohlc) - entry_idx):
            idx = entry_idx + i
            row = ohlc.iloc[idx]
            current_time = row.name

            if current_time > expiry_time:
                exit_price = ohlc.iloc[max(entry_idx, idx-1)]["close"]
                exit_time = current_time
                break

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

        print(f"Exit at {current_time}, price: {exit_price:.2f}, percent: {(exit_price/entry_price - 1) * 100:.2f} %")
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
    sharpe = returns.mean() / (returns.std() + 1e-9)

    return {
        "score": sharpe,
        "return": equity - 1,
        "trades": len(trades)
    }
