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
    """
    Сохраняет:
    - все trials
    - лучшие параметры
    """
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = Path(base_path) / ts
    path.mkdir(parents=True, exist_ok=True)

    # все трейлы
    trials_df = study.trials_dataframe()
    trials_df.to_csv(path / "trials.csv", index=False)

    # лучшие параметры
    best = {
        "best_value": study.best_value,
        **study.best_params,
    }
    pd.DataFrame([best]).to_csv(path / "best_params.csv", index=False)

    print(f"\nOptuna results saved to: {path}")
