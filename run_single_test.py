from loader.signals import load_signals
from core.baskets import backtest
from config.params import StrategyParams
# from src.data_loader import load_ohlc, load_signals
# from src.features import add_indicators
# from src.filters import default_filters
# from src.backtester import evaluate_strategy

SIGNALS_PATH = "data/signals/signals.csv"

if __name__ == "__main__":
    signals = load_signals(SIGNALS_PATH)

    params = StrategyParams(
        entry_lookback=20,
        sl=3.0,
        tp=8.0
    )

    results = backtest(signals, params)

    for r in results:
        print(
            r["datetime"],
            r["symbols_count"],
            round(r["avg_pnl"], 2),
            round(r["total_pnl"], 2)
        )

# ohlc = add_indicators(load_ohlc("data/raw/PLTR.csv"))
# signals = load_signals("data/raw/PLTR_signals.csv")

# result = evaluate_strategy(
#     ohlc=ohlc,
#     # signals=signals.head(5),  # ТЕСТ: первые 5 сигналов
#     signals=signals,
#     entry_delay=6,
#     sl=0.025,
#     tp=0.035,
#     # holding_minutes=60*24*5,
#     holding_minutes=60*24,
#     commission=0.001,
#     slippage=0.0002,
#     filters_fn=default_filters
# )

# print(result)
