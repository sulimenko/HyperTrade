import os
import pandas as pd
from loader.api_client import fetch_market_data

MIN_ROWS = 7000

def load_market_data(symbol: str):
    path = f"data/market/{symbol}.csv"

    if os.path.exists(path):
        df = pd.read_csv(path, sep=";")
        if len(df) >= MIN_ROWS:
            return df

    # если данных нет или мало — POST
    df = fetch_market_data(symbol)
    df.to_csv(path, sep=";", index=False)
    return df
