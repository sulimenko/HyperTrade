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
from config.params import build_optuna_params
from utils.save import save_optimization_results
from core.early_stopping import EarlyStopper

SIGNALS_PATH = "data/signals/signals.csv"

def objective(trial: optuna.trial.Trial, args, signals):
    params = build_optuna_params(trial, args)

    trades, _ = backtest(signals, params)

    if not trades:
        trial.set_user_attr("no_trades", 1.0)
        return -1e9

    # Считаем метрики + score (Variant A по умолчанию)
    metrics = compute_metrics(
        trades,
        params,
        objective="variant_a",
        trades_target=args.trades_target,
        objective_gates={
            "min_total_pnl": args.gate_min_total_pnl,
            "min_trades": args.gate_min_trades,
            "max_max_drawdown": args.gate_max_drawdown,
            "k_hold": args.k_hold,
        },
        delay_penalty_k=args.k_delay,
    )

    # Сохраняем компоненты score, чтобы потом было видно в trials.csv
    keys_to_save = [
        "score",
        "total_pnl",
        "max_drawdown",
        "cvar_5",
        "instability",
        "trades",
        "calmar",
        "sharpe_trade",
        "sortino_trade",
        "avg_hold_minutes",
        "complexity_penalty",
        "sample_penalty",
    ]
    for k in keys_to_save:
        if k in metrics and metrics[k] is not None:
            try:
                trial.set_user_attr(k, float(metrics[k]))
            except Exception:
                pass

    return float(metrics.get("score", -1e9))


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

    # --- Objective tuning knobs ---
    parser.add_argument("--trades_target", type=int, default=800)

    # gates for hard filtering
    parser.add_argument("--gate_min_total_pnl", type=float, default=0.0)
    parser.add_argument("--gate_min_trades", type=int, default=100)
    parser.add_argument("--gate_max_drawdown", type=float, default=1e18)

    # penalties
    parser.add_argument("--k_hold", type=float, default=0.35)     # штраф за log1p(avg_hold_minutes)
    parser.add_argument("--k_delay", type=float, default=0.005)   # мягкий штраф за delay_open (предпочесть 0)
    
    args = parser.parse_args()

    # Загружаем сигналы один раз
    signals = load_signals(args.signals)

    # optuna.logging.disable_default_handler()

    sampler = optuna.samplers.TPESampler(
        n_startup_trials=30,   # минимум случайных
        multivariate=True,     # учитывает связи параметров
        group=True,            # группирует параметры
        consider_prior=True,
        consider_magic_clip=True,
        constant_liar=True,
        seed=42
    )

    study = optuna.create_study(direction="maximize", sampler=sampler)
    # study.optimize(make_objective(args), n_trials=args.n_trials)
    study.optimize(
        lambda t: objective(t, args, signals), 
        callbacks=[EarlyStopper(patience=50, warmup=40)], 
        n_trials=args.n_trials
    )

    try:
        save_optimization_results(study)
    except Exception as e:
        print("⚠️ Failed to save Optuna results:", e)

    print("\n=== BEST PARAMS ===")
    for k, v in study.best_trial.params.items():
        print(f"{k:25}: {v}")
    print("Score:", study.best_value)


if __name__ == "__main__":
    run()
