import pandas as pd

def load_ohlc(path: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        sep=";",
        dtype={
            "open": float,
            "high": float,
            "low": float,
            "close": float,
            "volume": float,
            "timestamp": "int64"
        })
    
    df["datetime"] = pd.to_datetime(df["timestamp"],unit="ms",utc=True)
    df = df.set_index("datetime")
    # приведение типов (на всякий случай)
    df = df.astype({
        "open": float,
        "high": float,
        "low": float,
        "close": float,
        "volume": float,
    })

    df = df.sort_index()

    return df[["open", "high", "low", "close", "volume"]]


def load_signals(path: str) -> pd.DataFrame:
    ts = pd.read_csv(path)
    idx = pd.to_datetime(ts["time"], utc=True)
    return pd.DataFrame(index=idx, data={"signal": 1})
