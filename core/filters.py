import pandas as pd

def filters(row, params) -> bool:
    ema_cfg = params.indicator_config.get("ema")
    if ema_cfg and ema_cfg[0]:
        _, sign, fast, slow = ema_cfg
        fast = int(fast)
        slow = int(slow)

        col_fast = f"ema_{fast}"
        col_slow = f"ema_{slow}"

        if col_fast not in row or col_slow not in row:
            return False
        if pd.isna(row[col_fast]) or pd.isna(row[col_slow]):
            return False

        fast_v = float(row[col_fast])
        slow_v = float(row[col_slow])

        # above => fast > slow ; below => fast < slow
        if sign == "above" and not (fast_v > slow_v):
            return False
        if sign == "below" and not (fast_v < slow_v):
            return False

    rsi_cfg = params.indicator_config.get("rsi")
    if rsi_cfg and rsi_cfg[0]:
        _, sign, level, period = rsi_cfg
        col = f"rsi_{period}"

        if col not in row or pd.isna(row[col]):
            return False

        if sign == "above" and row[col] <= level:
            return False
        elif sign == "below" and row[col] >= level:
            return False

    # if params.indicator_config["volume"][0]:
    #     if "volume" in row and row["volume"] <= row["volume_ma"]:
    #         return False

    return True
