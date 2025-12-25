import requests
import pandas as pd

def fetch_market_data(symbol: str) -> pd.DataFrame:
    payload = {
        "symbol": symbol,
        "timeframe": "5m",
        "limit": 500
    }

    response = requests.post(
        "https://API_URL_PLACEHOLDER/market-data",
        json=payload,
        timeout=30
    )

    data = response.json()["data"]

    return pd.DataFrame(data)
