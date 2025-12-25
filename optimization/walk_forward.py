import pandas as pd
import optuna
from core.baskets import backtest
from optimization.objective import objective

def walk_forward_optimization(
    signals,
    train_days=90,
    test_days=30,
    step_days=30
):
    results = []

    signals_df = pd.DataFrame(signals)
    signals_df = signals_df.sort_values("datetime")

    start = 0
    while start + train_days + test_days < len(signals_df):
        train = signals_df.iloc[start : start + train_days]
        test = signals_df.iloc[start + train_days : start + train_days + test_days]

        study = optuna.create_study(direction="maximize")
        study.optimize(lambda t: objective(t, train.to_dict("records")), n_trials=50)

        best_params = study.best_params

        test_results = backtest(
            test.to_dict("records"),
            params_from_dict(best_params)
        )

        results.append({
            "start": train.iloc[0]["datetime"],
            "end": test.iloc[-1]["datetime"],
            "avg_pnl": sum(r["avg_pnl"] for r in test_results) / len(test_results)
        })

        start += step_days

    return results
