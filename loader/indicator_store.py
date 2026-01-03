import os
import pandas as pd

INDICATOR_PATH = "data/indicators"
os.makedirs(INDICATOR_PATH, exist_ok=True)

def _indicator_file(symbol: str) -> str:
    return f"{INDICATOR_PATH}/{symbol}.csv"

def load_indicator_csv(symbol: str) -> pd.DataFrame | None:
    path = _indicator_file(symbol)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=";")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True, errors="coerce")
    df = df.dropna(subset=["datetime"])
    return df

def save_indicator_csv(symbol: str, df: pd.DataFrame):
    path = _indicator_file(symbol)
    
    out = df.copy()
    out["timestamp"] = (pd.to_datetime(df["datetime"], utc=True).astype("int64") // 10**6)
    # out["timestamp"] = (pd.to_datetime(out["datetime"]).astype("int64") // 10**6).astype("int64")

    out = out.drop(columns=["datetime"])
    out.to_csv(path, sep=";", index=False)
