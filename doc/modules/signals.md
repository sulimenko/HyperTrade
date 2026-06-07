# Signals Module

Code location: `hypertrade/signals`.

The signals module loads external signal CSV files and converts each row into long and short symbol lists. Phase 1 treats these signals as input from an existing strategy or client robot.

It also applies indicator filters, time gates, cooldown rules, confirmation bars, ranking, and per-side signal caps before simulation evaluates accepted symbols.

`hypertrade/signals/filter_space.py` converts an Optuna trial plus a `FilterSearchSpace` into concrete strategy parameters.

Important boundaries:

- Native signal generation is not the primary workflow in phase 1.
- Filters reject or accept existing signals; they do not allocate capital.
- Search-space choices belong here, while objective scoring belongs to optimization.
