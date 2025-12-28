import pandas as pd
import ta

def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=20)
    df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=50)
    df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    df["atr"] = ta.volatility.average_true_range(
        df["high"], df["low"], df["close"], window=14
    )

    if "volume" in df.columns:
        df["volume_ma"] = df["volume"].rolling(20).mean()
    else:
        df["volume_ma"] = 1.0  # нейтральное значение

    return df.dropna()
