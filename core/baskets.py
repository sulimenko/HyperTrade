# from loader.market_data import load_market_data
from loader.ensure_data import ensure_market_data
from core.simulator import simulate_trade

def backtest(signals, params):
    trades = []
    signal_stats = []

    for signal in signals:
        day_trades = []
        rejected = []

        for symbol in signal["symbols"]:
            # ohlc = load_market_data(symbol, start=signal["datetime"])
            ohlc = ensure_market_data(symbol, start=signal["datetime"], indicator_config=params.indicator_config)
            if ohlc is None:
                rejected.append((symbol, "no_market_data"))
                continue

            trade = simulate_trade(
                symbol=symbol,
                signal_time=signal["datetime"],
                params=params,
                ohlc=ohlc
            )
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

    return trades, signal_stats
