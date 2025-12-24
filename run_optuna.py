import optuna
from src.data_loader import load_ohlc, load_signals
from optuna.objective import objective

ohlc = load_ohlc("data/raw/TSLA.csv")
signals = load_signals("data/raw/TSLA_signals.csv")

study = optuna.create_study(direction="maximize")
study.optimize(
    lambda t: objective(t, ohlc, signals),
    n_trials=300
)

print("BEST PARAMS:", study.best_params)
print("BEST SHARPE:", study.best_value)
