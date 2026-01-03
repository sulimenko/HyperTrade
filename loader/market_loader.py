import os
import pandas as pd
from loader.api_client import fetch_market_data

MARKET_PATH = "data/market"
os.makedirs(MARKET_PATH, exist_ok=True)

def load_market_csv(symbol: str) -> pd.DataFrame | None:
    path = f"{MARKET_PATH}/{symbol}.csv"
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, sep=";")
    df["datetime"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    return df

def save_market_csv(symbol: str, df: pd.DataFrame):
    path = f"{MARKET_PATH}/{symbol}.csv"
    df.to_csv(path, sep=";", index=False)

def ensure_market_history(symbol: str, start) -> pd.DataFrame | None:
    df = load_market_csv(symbol)
    if df is not None:
        if df["datetime"].min() <= start:
            return df

    # ⬇️ догружаем через API
    api_df = fetch_market_data(symbol)
    if api_df is None:
        return None

    api_df["datetime"] = pd.to_datetime(api_df["timestamp"], unit="ms", utc=True)

    if df is not None:
        df = pd.concat([api_df, df]).drop_duplicates("timestamp")
    else:
        df = api_df

    df = df.sort_values("timestamp")
    save_market_csv(symbol, df)
    return df
