from src.data_loader import load_ohlc, load_signals
from src.backtester import evaluate_strategy

ohlc = load_ohlc("data/raw/TSLA.csv")
signals = load_signals("data/raw/TSLA_signals.csv")

result = evaluate_strategy(
    ohlc=ohlc,
    # signals=signals.head(5),  # ТЕСТ: первые 5 сигналов
    signals=signals,
    entry_delay=0,
    sl=0.05,
    tp=0.05,
    holding_minutes=60*24*7,
    commission=0.001,
    slippage=0.0005,
)

print(result)
