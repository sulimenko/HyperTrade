# from functools import lru_cache
import pandas as pd
from loader.market_loader import ensure_market_history
from loader.indicator_calc import calculate_indicators
from loader.indicator_store import load_indicator, save_indicator

# @lru_cache(maxsize=2048)
# def ensure_market_data(symbol: str, start: pd.Timestamp, indicator_key: tuple) -> pd.DataFrame | None:
def ensure_market_data(symbol: str, start: pd.Timestamp, indicator_config) -> pd.DataFrame | None:
    market_df = ensure_market_history(symbol, start)
    if market_df is None or market_df.empty:
        return None
    
    # indicator_config = { k: list(v) for k, v in indicator_key }

    ind_df = load_indicator(symbol)
    if ind_df is None or ind_df.empty:
        ind_df = calculate_indicators(market_df, indicator_config)
        save_indicator(symbol, ind_df)
        return market_df.merge(ind_df, on="datetime", how="left")

    sample_ind = calculate_indicators(market_df.iloc[:1], indicator_config)
    required_cols = set(sample_ind.columns) - {"datetime"}
    existing_cols = set(ind_df.columns)

    missing_cols = list(required_cols - existing_cols)

    if missing_cols:
        full_ind = calculate_indicators(market_df, indicator_config)
        new_ind = full_ind[["datetime"] + missing_cols]

        ind_df = ind_df.merge(new_ind, on="datetime", how="left")
        save_indicator(symbol, ind_df)

    return market_df.merge(ind_df, on="datetime", how="left")

    # if not calc:
    #     return None
    
    # # считаем индикаторы
    # ind_df = calculate_indicators(total_df, indicator_config)
    # save_indicator_csv(symbol, hash_, ind_df)

    # return ensure_market_data(symbol, start, indicator_config, calc=False)