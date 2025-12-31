import argparse
import optuna

import warnings
warnings.filterwarnings(
    "ignore",
    category=optuna.exceptions.ExperimentalWarning
)

from loader.signals import load_signals
from core.baskets import backtest
from core.metrics import compute_metrics
from config.params import StrategyParams
from utils.save import save_optimization_results
from core.early_stopping import EarlyStopper

SIGNALS_PATH = "data/signals/signals.csv"

# def make_objective(args):
def objective(trial, args):
    params = StrategyParams(
        signals=args.signals,
        sl=trial.suggest_float("sl", args.sl_min, args.sl_max, step=args.sl_step),
        tp=trial.suggest_float("tp", args.tp_min, args.tp_max, step=args.tp_step),
        delay_open=trial.suggest_int("delay_open", args.delay_open_min, args.delay_open_max, step=args.delay_open_step),
        holding_minutes=trial.suggest_int("holding_minutes", args.holding_minutes_min, args.holding_minutes_max, step=args.holding_minutes_step),
    )

    signals = load_signals(args.signals)
    trades, _ = backtest(signals, params)

    metrics = compute_metrics(trades)
    return metrics.get("total_pnl", 0)

def objective_narrow(trial, best_params):
    params = StrategyParams(
        sl=trial.suggest_float("sl", max(0.1, best_params["sl"] - 0.5), best_params["sl"] + 0.5, step=0.1),
        tp=trial.suggest_float("tp", max(0.5, best_params["tp"] - 1.0), best_params["tp"] + 1.0, step=0.2),
        delay_open=trial.suggest_int("delay_open", max(0, best_params["delay_open"] - 30), best_params["delay_open"] + 30, step=5),
        holding_minutes=best_params["holding_minutes"],
    )

    signals = load_signals(SIGNALS_PATH)
    trades, _ = backtest(signals, params)
    metrics = compute_metrics(trades)

    return metrics.get("total_pnl", 0)

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--signals", type=str, default=SIGNALS_PATH)

    parser.add_argument("--sl_min", type=float, default=2)
    parser.add_argument("--sl_max", type=float, default=7.0)
    parser.add_argument("--sl_step", type=float, default=0.5)
    
    parser.add_argument("--tp_min", type=float, default=3.0)
    parser.add_argument("--tp_max", type=float, default=15.0)
    parser.add_argument("--tp_step", type=float, default=0.5)

    parser.add_argument("--delay_open_min", type=int, default=0)
    parser.add_argument("--delay_open_max", type=int, default=600)
    parser.add_argument("--delay_open_step", type=int, default=30)

    parser.add_argument("--holding_minutes_min", type=int, default=60*24)
    parser.add_argument("--holding_minutes_max", type=int, default=60*24*4)
    parser.add_argument("--holding_minutes_step", type=int, default=60*3)

    parser.add_argument("--n_trials", type=int, default=300)
    parser.add_argument("--dd_penalty", type=float, default=0.5)
    
    args = parser.parse_args()

    # optuna.logging.disable_default_handler()

    sampler = optuna.samplers.TPESampler(
       n_startup_trials=30,   # минимум случайных
        multivariate=True,     # учитывает связи параметров
        group=True,            # группирует параметры
        consider_prior=True,
        consider_magic_clip=True,
        seed=42
    )

    study = optuna.create_study(direction="maximize", sampler=sampler)
    # study.optimize(make_objective(args), n_trials=args.n_trials)
    study.optimize(
        lambda t: objective(t, args), 
        callbacks=[EarlyStopper(patience=50, warmup=40)], 
        n_trials=args.n_trials
    )

    save_optimization_results(study)

    print("\n=== BEST PARAMS ===")
    for k, v in study.best_params.items():
        print(f"{k:25}: {v}")
    
    print("Score:", study.best_value)

    best_params = study.best_params

    narrow_study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            n_startup_trials=10,
            multivariate=True,
            seed=1337,
        ),
    )

    narrow_study.optimize(
        lambda t: objective_narrow(t, best_params),
        callbacks=[EarlyStopper(patience=30, warmup=10)],
        n_trials=150,
    )

    print("\n=== FINAL BEST (NARROW) ===")
    for k, v in narrow_study.best_params.items():
        print(f"{k:25}: {v}")
    print("Final score:", narrow_study.best_value)

    save_optimization_results(narrow_study)


if __name__ == "__main__":
    run()
