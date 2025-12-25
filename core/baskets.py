from loader.market_data import load_market_data
from core.simulator import simulate_trade

def backtest(signals, params):
    results = []

    for signal in signals:
        pnls = []

        for symbol in signal["symbols"]:
            ohlc = load_market_data(symbol)

            pnl = simulate_trade(
                symbol=symbol,
                signal_time=signal["datetime"],
                params=params,
                ohlc=ohlc
            )
            pnls.append(pnl)

        results.append({
            "datetime": signal["datetime"],
            "symbols_count": len(pnls),
            "avg_pnl": sum(pnls) / len(pnls),
            "total_pnl": sum(pnls)
        })

    return results
