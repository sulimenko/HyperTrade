import optuna

from loader.signals import load_signals
from core.baskets import backtest
from analytics.metrics import compute_metrics
from config.params import StrategyParams
from optimization.save import save_optimization_results

SIGNALS_PATH = "data/signals/signals.csv"


def objective(trial):
    params = StrategyParams(
        sl=trial.suggest_float("sl", 1.5, 5.0, step=0.5),
        tp=trial.suggest_float("tp", 3.0, 10.0, step=0.5),
        delay_open=trial.suggest_int("delay_open", 0, 300, step=10),
        holding_minutes=trial.suggest_int(
            "holding_minutes", 60, 60 * 24 * 7, step=60
        ),
        commission=trial.suggest_float("commission", 0.0002, 0.001),
        slippage=trial.suggest_float("slippage", 0.0, 0.0005),
    )

    signals = load_signals(SIGNALS_PATH)
    trades, _ = backtest(signals, params)

    metrics = compute_metrics(trades)

    # целевая функция
    score = metrics["expectancy"]
    score -= metrics.get("max_drawdown", 0) * 0.5

    return score


def run():
    optuna.logging.disable_default_handler()

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=100)

    save_optimization_results(study)

    print("\n=== BEST PARAMS ===")
    for k, v in study.best_params.items():
        print(f"{k:25}: {v}")
    print("Best score:", study.best_value)


if __name__ == "__main__":
    run()
