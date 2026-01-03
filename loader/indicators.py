import pandas as pd
import ta

def add_indicators(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    df = df.copy()

    for name, periods in config.items():
        for p in periods:
            col = f"{name}_{p}"

            if col in df.columns:
                continue

            if name == "ema":
                df[col] = df["close"].ewm(span=p, adjust=False).mean()

            elif name == "rsi":
                delta = df["close"].diff()
                gain = delta.clip(lower=0)
                loss = -delta.clip(upper=0)
                rs = gain.rolling(p).mean() / loss.rolling(p).mean()
                df[col] = 100 - (100 / (1 + rs))

            elif name == "atr":
                high_low = df["high"] - df["low"]
                high_close = (df["high"] - df["close"].shift()).abs()
                low_close = (df["low"] - df["close"].shift()).abs()
                tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
                df[col] = tr.rolling(p).mean()

    return df
    # df["ema_fast"] = ta.trend.ema_indicator(df["close"], window=20)
    # df["ema_slow"] = ta.trend.ema_indicator(df["close"], window=50)
    # df["rsi"] = ta.momentum.rsi(df["close"], window=14)

    # df["atr"] = ta.volatility.average_true_range(
    #     df["high"], df["low"], df["close"], window=14
    # )

    # if "volume" in df.columns:
    #     df["volume_ma"] = df["volume"].rolling(20).mean()
    # else:
    #     df["volume_ma"] = 1.0  # нейтральное значение

    # return df.dropna()
