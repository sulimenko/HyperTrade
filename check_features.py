from src.data_loader import load_ohlc
from src.features import add_indicators

df = load_ohlc("data/raw/TSLA.csv")
df = add_indicators(df)

print(df.tail())
