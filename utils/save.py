import pandas as pd
from pathlib import Path
from datetime import datetime

RESULT_PATH = Path("results")


def _result_dir(kind: str) -> Path:
    ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")
    path = RESULT_PATH / kind / ts
    path.mkdir(parents=True, exist_ok=True)
    return path

def _to_jsonable(x):
    try:
        import numpy as np
        if isinstance(x, (np.floating,)):
            return float(x)
        if isinstance(x, (np.integer,)):
            return int(x)
        if isinstance(x, (np.ndarray,)):
            return x.tolist()
    except Exception:
        pass

    if isinstance(x, (dict, list, tuple, set)):
        return str(x)

    return x


def save_optimization_results(study):
    path = _result_dir("optuna")

    df = study.trials_dataframe(
        attrs=(
            "number",
            "value",
            "state",
            "datetime_start",
            "datetime_complete",
            "duration",
            "params",
            "user_attrs",
        )
    )

    for col in df.columns:
        df[col] = df[col].apply(_to_jsonable)

    df.to_csv(path / "trials.csv", index=False)

    if "state" in df.columns:
        df_ok = df[df["state"] == "COMPLETE"].copy()
    else:
        df_ok = df.copy()

    if "value" in df_ok.columns and len(df_ok):
        df_sorted = df_ok.sort_values("value", ascending=False)
        top = df_sorted.head(max(1, int(len(df_sorted) * 0.1)))
        top.to_csv(path / "top_trials.csv", index=False)

    best = study.best_trial
    best_row = {
        "best_value": float(best.value),
        **best.params,
        **{f"user_{k}": _to_jsonable(v) for k, v in (best.user_attrs or {}).items()},
    }
    pd.DataFrame([best_row]).to_csv(path / "best.csv", index=False)

    print(f"\n✅ Optuna results saved to: {path}")

def save_csv(df, name: str):
    path = _result_dir("single")
    df.to_csv(path / name, index=False)
    print(f"\n✅ Single results saved to: {path / name}")