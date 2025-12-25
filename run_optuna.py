import optuna

from loader.signals import load_signals
from optimization.objective import objective
from optimization.save import save_optimization_results

# Отключаем стандартный логгер Optuna
optuna.logging.disable_default_handler()

SIGNALS_PATH = "data/signals/signals.csv"

if __name__ == "__main__":
    signals = load_signals(SIGNALS_PATH)

    study = optuna.create_study(direction="maximize")
    study.optimize(lambda t: objective(t, signals), n_trials=100)

    save_optimization_results(study)

    print("BEST:", study.best_params, study.best_value)



# import optuna

# from src.data_loader import load_ohlc, load_signals
# from src.features import add_indicators
# from src.filters import default_filters
# from src.backtester import evaluate_strategy

# ohlc = add_indicators(load_ohlc("data/raw/PLTR.csv"))
# signals = load_signals("data/raw/PLTR_signals.csv")


# def objective(trial):
#     params = {
#         # вход через N 15-мин свечей после сигнала
#         "entry_delay": trial.suggest_int("entry_delay", 1, 10, step=1),
        
#         # стоп-лосс и тейк-профит в %
#         "sl": trial.suggest_float("sl", 0.01, 0.06, step=0.001),
#         "tp": trial.suggest_float("tp", 0.03, 0.08, step=0.001),

#         # максимальное удержание позиции
#         "holding_minutes": trial.suggest_int(
#             "holding_minutes",
#             60 * 5,         # 5 часов
#             60 * 24 * 5,     # до 5 торговых дней
#             step=60 * 5
#         ),
#     }

#     result = evaluate_strategy(
#         ohlc=ohlc,
#         signals=signals,
#         commission=0.001,
#         slippage=0.0002,
#         filters_fn=default_filters,
#         # holding_minutes=60*24*5,
#         **params
#     )

#     # # --- ЖЁСТКИЕ ФИЛЬТРЫ КАЧЕСТВА ---
#     # if result["trades"] < 20:
#     #     return -10

#     # if result["return"] < 0:
#     #     return -5

#     return result["return"]


# study = optuna.create_study(direction="maximize")
# study.optimize(objective, n_trials=1000)

# print("\nBEST PARAMS")
# print(study.best_params)

# print("\nBEST RETURN")
# print(study.best_value)
