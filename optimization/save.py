import os
import pandas as pd
from pathlib import Path
from datetime import datetime


def save_walk_forward_results(results, path="data/results/walk_forward.csv"):
    """
    results: list[dict]
    """
    df = pd.DataFrame(results)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def save_optimization_results(study, base_path="data/optuna"):
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
