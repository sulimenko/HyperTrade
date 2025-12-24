from src.data_loader import load_ohlc

df = load_ohlc("data/raw/PLTR.csv")

print(df.head())
print(df.index)
print(df.index.tz)
