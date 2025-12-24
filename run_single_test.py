from src.data_loader import load_ohlc, load_signals
from src.features import add_indicators
from src.filters import default_filters
from src.backtester import evaluate_strategy

ohlc = add_indicators(load_ohlc("data/raw/PLTR.csv"))
signals = load_signals("data/raw/PLTR_signals.csv")

result = evaluate_strategy(
    ohlc=ohlc,
    # signals=signals.head(5),  # ТЕСТ: первые 5 сигналов
    signals=signals,
    entry_delay=3,
    sl=0.03,
    tp=0.04,
    # holding_minutes=60*24*5,
    holding_minutes=60*24*5,
    commission=0.001,
    slippage=0.0002,
    filters_fn=default_filters
)

print(result)
