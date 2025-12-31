def StrategyParams(signals, sl, tp, delay_open, holding_minutes, commission = 0.02, slippage = 0.0002):
    return {
        "signals": signals,
        "sl": sl,
        "tp": tp,
        "delay_open": delay_open,
        "holding_minutes": holding_minutes,
        "commission": commission,
        "slippage": slippage,
    }