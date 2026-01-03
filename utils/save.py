import os
import json
import pandas as pd
from pathlib import Path
from datetime import datetime

RESULT_PATH = "results"
os.makedirs(RESULT_PATH, exist_ok=True)

def _result_file(type="optuna") -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{RESULT_PATH}/{type}/{ts}.csv"

def save_optimization_results(study):
    path = _result_file("optuna")

    df = study.trials_dataframe()
    df.to_csv(path / "trials.csv", index=False)

    df_sorted = df.sort_values("value", ascending=False)
    top = df_sorted.head(max(1, int(len(df) * 0.1)))
    top.to_csv(path / "top_trials.csv", index=False)

    print(f"Saved Optuna results to {path}")

    with open(path / "best.txt", "w") as f:
        f.write(f"Best value: {study.best_value}\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")

    print(f"\nOptuna results saved to: {path}")

def save_csv(df, path, name):
    df.to_csv(path / name, index=False)

def save_json(data, path, name):
    with open(path / name, "w") as f:
        json.dump(data, f, indent=2, default=str)
