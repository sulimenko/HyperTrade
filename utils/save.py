import json
import pandas as pd
from pathlib import Path
from datetime import datetime

RESULT_PATH = Path("results")

def _result_file(type="optuna") -> str:
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = RESULT_PATH / type / ts
    path.mkdir(parents=True, exist_ok=True)
    return path

def save_optimization_results(study):
    path = _result_file("optuna")

    df = study.trials_dataframe()
    for col in df.columns:
        df[col] = df[col].apply(
            lambda x: str(x) if isinstance(x, (dict, list, set)) else x
        )

    df.to_csv(path / "trials.csv", index=False)

    df_sorted = df.sort_values("value", ascending=False)
    top = df_sorted.head(max(1, int(len(df) * 0.1)))
    top.to_csv(path / "top_trials.csv", index=False)

    best_row = {
        "best_value": study.best_value,
        **study.best_params,
    }
    pd.DataFrame([best_row]).to_csv(path / "best.csv", index=False)

    print(f"\n✅ Optuna results saved to: {path}")

def save_csv(df, path, name):
    df.to_csv(path / name, index=False)

def save_json(data, path, name):
    with open(path / name, "w") as f:
        json.dump(data, f, indent=2, default=str)
