import time
from loader.ensure_data import ensure_market_data
from core.simulator import simulate_trade

def backtest(signals, params):
    trades = []
    signal_stats = []
    start_time = time.time()

    for signal in signals:
        day_trades = []
        rejected = []

        for symbol in signal["symbols"]:
            # start_time = time.time()
            ohlc = ensure_market_data(symbol, start=signal["datetime"], indicator_config=params.indicator_config)
            # print(f"Время {symbol} ohlc: {(time.time() - start_time):.4f} секунд")
            if ohlc is None:
                rejected.append((symbol, "no_market_data"))
                continue

            trade = simulate_trade(
                symbol=symbol,
                signal_time=signal["datetime"],
                params=params,
                ohlc=ohlc
            )
            # print(f"Время {symbol} ohlc + trade: {(time.time() - start_time):.4f} секунд")
            if trade.get("rejected"):
                rejected.append((symbol, trade["reject_reason"]))
            else:
                trades.append(trade)
                day_trades.append(trade)

        signal_stats.append({
            "datetime": signal["datetime"],
            "symbols_total": len(signal["symbols"]),
            "symbols_traded": len(day_trades),
            "symbols_rejected": len(rejected),
            "total_pnl": sum(t["pnl"] for t in day_trades),
            "avg_pnl": (
                sum(t["pnl"] for t in day_trades) / len(day_trades)
                if day_trades else 0
            ),
            # "total_pnl": sum(t["pnl"] for t in day_trades)
        })

    print(f"Время: {(time.time() - start_time):.4f} секунд")
    return trades, signal_stats
