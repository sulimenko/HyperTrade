from pathlib import Path
from datetime import datetime
import json
import pandas as pd

def create_results_dir(base="data/results"):
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(base) / ts
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_csv(df, path, name):
    df.to_csv(path / name, index=False)

def save_json(data, path, name):
    with open(path / name, "w") as f:
        json.dump(data, f, indent=2, default=str)
