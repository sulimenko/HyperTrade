from src.data_loader import load_ohlc, load_signals
from src.backtester import evaluate_strategy

ohlc = load_ohlc("data/raw/TSLA.csv")
signals = load_signals("data/raw/TSLA_signals.csv")

result = evaluate_strategy(
    ohlc=ohlc,
    signals=signals.head(5),  # ТЕСТ: первые 5 сигналов
    entry_delay=1,
    sl_pct=0.02,
    tp_pct=0.06,
    max_holding=32
)

print(result)
