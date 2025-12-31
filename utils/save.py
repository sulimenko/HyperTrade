import json
import pandas as pd
from pathlib import Path
from datetime import datetime


def save_walk_forward_results(results, path="results/walk/walk_forward.csv"):
    """
    results: list[dict]
    """
    df = pd.DataFrame(results)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_optimization_results(study, base_path="results/optuna"):
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(base_path) / ts
    path.mkdir(parents=True, exist_ok=True)

    # все трейлы
    df = study.trials_dataframe(attrs=("number", "value", "params", "state"))
    df = df[df["state"] == "COMPLETE"]
    df.to_csv(path / "trials.csv", index=False)

    with open(f"{path}/best.txt", "w") as f:
        f.write(f"Best value: {study.best_value}\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")

    print(f"\nOptuna results saved to: {path}")

def save_csv(df, path, name):
    df.to_csv(path / name, index=False)

def save_json(data, path, name):
    with open(path / name, "w") as f:
        json.dump(data, f, indent=2, default=str)

def create_results_dir(base="results/single"):
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(base) / ts
    path.mkdir(parents=True, exist_ok=True)
    return path
