# from src.backtester import evaluate_strategy
# from src.filters import ema_trend_filter, atr_volatility_filter


# def objective(trial, ohlc, signals):
#     entry_delay = trial.suggest_int("entry_delay", 0, 4)
#     sl_pct = trial.suggest_float("sl_pct", 0.003, 0.05)
#     tp_pct = trial.suggest_float("tp_pct", 0.005, 0.15)
#     max_holding = trial.suggest_int("max_holding", 4, 64)

#     def filters(entry_time, ohlc_df):
#         return (
#             ema_trend_filter(entry_time, ohlc_df, 50)
#             and atr_volatility_filter(entry_time, ohlc_df)
#         )

#     result = evaluate_strategy(
#         ohlc=ohlc,
#         signals=signals,
#         entry_delay=entry_delay,
#         sl_pct=sl_pct,
#         tp_pct=tp_pct,
#         max_holding=max_holding,
#         filters_fn=filters
#     )

#     # if result["trades"] < 10:
#         # return -999

#     return result["sharpe"]
