# Data Module

Code location: `hypertrade/data`.

The data layer loads and stores market context needed to evaluate external signals. It searches symbols through the configured market-data API, fetches OHLC candles, and stores local parquet caches under `data/ohlc/`.

It also owns indicator cache persistence under `data/indicators/`. Feature and simulation code should use these helpers instead of reading or writing cache files directly.

Important boundaries:

- `DATA_BASE_URL` is read from the environment.
- Market and indicator parquet files are runtime data, not repository artifacts.
- The module does not define trading logic; it provides normalized dataframes to downstream modules.
