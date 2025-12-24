from src.backtester import evaluate_strategy
from src.data_loader import load_ohlc, load_signals

params = {
    "entry_delay": 1,
    "sl_pct": 0.015,
    "tp_pct": 0.06,
    "max_holding": 32
}

ohlc = load_ohlc("data/raw/tsla_ohlc_15m.csv")
signals = load_signals("data/raw/tsla_signals.csv")

result = evaluate_strategy(
    ohlc,
    signals,
    **params
)

print(result)
