import argparse
import optuna

from loader.signals import load_signals
from core.baskets import backtest
from analytics.metrics import compute_metrics
from config.params import StrategyParams
from optimization.save import save_optimization_results
from optimization.early_stopping import EarlyStopper

SIGNALS_PATH = "data/signals/signals.csv"

# def make_objective(args):
def objective(trial, args):
    params = StrategyParams(
        sl=trial.suggest_float("sl", args.sl_min, args.sl_max, step=args.sl_step),
        tp=trial.suggest_float("tp", args.tp_min, args.tp_max, step=args.tp_step),
        delay_open=trial.suggest_int("delay_open", args.delay_open_min, args.delay_open_max, step=args.delay_open_step),
        holding_minutes=trial.suggest_int("holding_minutes", args.holding_minutes_min, args.holding_minutes_max, step=args.holding_minutes_step),
    )

    signals = load_signals(SIGNALS_PATH)
    trades, _ = backtest(signals, params)

    metrics = compute_metrics(trades)
    return metrics.get("total_pnl", 0)

def run():
    parser = argparse.ArgumentParser()

    parser.add_argument("--sl_min", type=float, default=2)
    parser.add_argument("--sl_max", type=float, default=7.0)
    parser.add_argument("--sl_step", type=float, default=0.5)
    
    parser.add_argument("--tp_min", type=float, default=3.0)
    parser.add_argument("--tp_max", type=float, default=15.0)
    parser.add_argument("--tp_step", type=float, default=0.5)

    parser.add_argument("--delay_open_min", type=int, default=0)
    parser.add_argument("--delay_open_max", type=int, default=300)
    parser.add_argument("--delay_open_step", type=int, default=30)

    parser.add_argument("--holding_minutes_min", type=int, default=60*24)
    parser.add_argument("--holding_minutes_max", type=int, default=60*24*4)
    parser.add_argument("--holding_minutes_step", type=int, default=60*3)

    parser.add_argument("--n_trials", type=int, default=50)
    parser.add_argument("--dd_penalty", type=float, default=0.5)
    
    args = parser.parse_args()

    # optuna.logging.disable_default_handler()

    study = optuna.create_study(direction="maximize")
    # study.optimize(make_objective(args), n_trials=args.n_trials)
    study.optimize(
        lambda t: objective(t, args), 
        callbacks=[EarlyStopper(patience=15)], 
        n_trials=args.n_trials
    )

    save_optimization_results(study)

    print("\n=== BEST PARAMS ===")
    for k, v in study.best_params.items():
        print(f"{k:25}: {v}")
    
    print("Best score:", study.best_value)


if __name__ == "__main__":
    run()
