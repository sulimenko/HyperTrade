def objective(trial):
    params = StrategyParams(
        entry_lookback=trial.suggest_int("entry_lookback", 5, 50),
        sl_pct=trial.suggest_float("sl_pct", 1, 8),
        tp_pct=trial.suggest_float("tp_pct", 2, 20),
    )

    results = backtest(signals, params)

    return sum(r["avg_pnl"] for r in results) / len(results)
