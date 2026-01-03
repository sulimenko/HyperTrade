import pandas as pd

def load_signals(path: str):
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"])

    df["datetime"] = df["datetime"].dt.tz_localize("UTC")
    
    signals = []
    for _, row in df.iterrows():
        symbols = [s.strip() for s in row["symbols"].split(",") if s.strip()]
        signals.append({
            "datetime": row["datetime"],
            "symbols": symbols
        })

    return signals
