import optuna

from loader.signals import load_signals
from optimization.objective import objective
from optimization.save import save_optimization_results

optuna.logging.disable_default_handler()

SIGNALS_PATH = "data/signals/signals.csv"

if __name__ == "__main__":
    signals = load_signals(SIGNALS_PATH)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, signals), n_trials=100)

    save_optimization_results(study)

    print("BEST:", study.best_params, study.best_value)