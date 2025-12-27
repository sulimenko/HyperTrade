def objective(trial):
    params = StrategyParams(
        delay_open=60,
        sl=2.5,
        tp=4.0
    )

    results = backtest(signals, params)

    return sum(r["avg_pnl"] for r in results) / len(results)
