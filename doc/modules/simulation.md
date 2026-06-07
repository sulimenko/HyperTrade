# Simulation Module

Code location: `hypertrade/simulation`.

The simulation module turns accepted signals into trade outcomes. It handles market-time entry and exit calculations, delayed opens, holding windows, stop loss, take profit, trailing stop, PSAR exit, slippage, and commission.

`trade_simulator.py` contains the trade core and backtest flow. `market_time.py` provides exchange-session aware timestamp helpers used by both signal acceptance and trade exits.

Important boundaries:

- Simulation evaluates a chosen parameter set; it does not choose Optuna parameters.
- The module reports rejected signals and trade metrics for analysis.
- Live execution, capital sizing, and portfolio allocation are outside phase 1.
