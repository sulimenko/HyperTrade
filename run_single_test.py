from loader.signals import load_signals
from core.baskets import backtest
from config.params import StrategyParams

SIGNALS_PATH = "data/signals/signals.csv"

if __name__ == "__main__":
    signals = load_signals(SIGNALS_PATH)

    params = StrategyParams(
        dalay_open=60,
        sl=2.5,
        tp=4.0
    )

    results = backtest(signals, params)

    for r in results:
        print(f"{r['datetime']}, {r['symbols_count']}, {r['avg_pnl']:.4f}, {r['total_pnl']:.4f}")
