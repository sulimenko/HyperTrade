# Features Module

Code location: `hypertrade/features`.

The features module prepares indicator columns used by filters and simulation. `indicator_engine.py` calculates EMA, RSI, ATR, Donchian, ADX, Bollinger Bands, VWAP, and volume-derived fields from OHLC input.

The module also determines which indicator columns are required for a strategy configuration and hydrates missing cached indicators through the data layer.

Important boundaries:

- Indicator definitions are driven by strategy config.
- Feature output is dataframe-based and keyed by `datetime`.
- The module should not choose strategy parameters or run studies.
