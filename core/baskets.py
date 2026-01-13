import time
import pandas as pd


from loader.ensure_data import ensure_market_data
from core.simulator import simulate_trade, LONG, SHORT
from core.market_time import build_market_cache


def backtest(signals, params):
    trades = []
    signal_stats = []
    start_time = time.time()

    dts = [pd.Timestamp(s["datetime"]) for s in signals if s.get("datetime") is not None]
    if dts:
        dt_min = min(dts)
        dt_max = max(dts)
    else:
        dt_min = pd.Timestamp.utcnow()
        dt_max = dt_min

    max_minutes = int(getattr(params, "holding_minutes", 0)) + int(getattr(params, "delay_open", 0))
    extra_days = max(60, int(max_minutes / 390) * 2 + 30)

    market_cache = build_market_cache(dt_min, dt_max, extra_days=extra_days)

    for signal in signals:
        day_trades = []
        rejected = []

        for direction in ["long", "short"]:
            for symbol in signal.get(direction, []):
                # start_time = time.time()
                ohlc = ensure_market_data(symbol, signal["datetime"], params.indicator_config)
                # print(f"Время {symbol} ohlc: {(time.time() - start_time):.4f} секунд")
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
                # print(f"Время {symbol} ohlc + trade: {(time.time() - start_time):.4f} секунд")
                if trade.get("rejected"):
                    rejected.append((symbol, trade["reject_reason"]))
                else:
                    trades.append(trade)
                    day_trades.append(trade)

        signal_stats.append({
            "datetime": signal["datetime"],
            "symbols_total": len(signal["long"]) + len(signal.get("short", [])),
            "symbols_traded": len(day_trades),
            "symbols_rejected": len(rejected),
            "total_pnl": sum(t["pnl"] for t in day_trades),
            "avg_pnl": (
                sum(t["pnl"] for t in day_trades) / len(day_trades)
                if day_trades else 0
            ),
        })

    print(f"Время: {(time.time() - start_time):.4f} секунд")
    return trades, signal_stats
