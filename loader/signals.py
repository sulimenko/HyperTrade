import pandas as pd


def _split_symbols(x) -> list[str]:
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return []
    return [s.strip() for s in str(x).split(",") if s.strip()]

def load_signals(path: str):
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], format='%d.%m.%Y %H:%M:%S', utc=True)

    has_long = "long_symbols" in df.columns
    has_short = "short_symbols" in df.columns
    
    signals = []
    for _, row in df.iterrows():
        long_symbols = _split_symbols(row["long_symbols"]) if has_long else _split_symbols(row.get("symbols"))
        short_symbols = _split_symbols(row["short_symbols"]) if has_short else []

        signals.append({
            "datetime": row["datetime"],
            "long": long_symbols,
            "short": short_symbols,
        })
    return signals
