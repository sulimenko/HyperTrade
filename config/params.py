def StrategyParams(delay_open, sl, tp, holding_minutes, commission, slippage):
    return {
        "sl": sl,
        "tp": tp,
        "delay_open": delay_open,
        "holding_minutes": holding_minutes,
        "commission": commission,
        "slippage": slippage,
    }