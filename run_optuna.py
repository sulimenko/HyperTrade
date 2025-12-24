import optuna

from src.data_loader import load_ohlc, load_signals
from src.features import add_indicators
from src.filters import default_filters
from src.backtester import evaluate_strategy

ohlc = add_indicators(load_ohlc("data/raw/PLTR.csv"))
signals = load_signals("data/raw/PLTR_signals.csv")


def objective(trial):
    params = {
        # вход через N 15-мин свечей после сигнала
        "entry_delay": trial.suggest_int("entry_delay", 1, 6, step=1),
        
        # стоп-лосс и тейк-профит в %
        "sl": trial.suggest_float("sl", 0.01, 0.04, step=0.001),
        "tp": trial.suggest_float("tp", 0.03, 0.04, step=0.001),

        # максимальное удержание позиции
        "holding_minutes": trial.suggest_int(
            "holding_minutes",
            60 * 5,         # 5 часов
            60 * 24 * 5,     # до 5 торговых дней
            step=60 * 5
        ),
    }

    result = evaluate_strategy(
        ohlc=ohlc,
        signals=signals,
        commission=0.001,
        slippage=0.0002,
        filters_fn=default_filters,
        # holding_minutes=60*24*5,
        **params
    )

    # # --- ЖЁСТКИЕ ФИЛЬТРЫ КАЧЕСТВА ---
    # if result["trades"] < 20:
    #     return -10

    # if result["return"] < 0:
    #     return -5

    return result["return"]


study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=100)

print("\nBEST PARAMS")
print(study.best_params)

print("\nBEST RETURN")
print(study.best_value)
