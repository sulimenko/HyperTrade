import pandas as pd

def default_filters(row, params) -> bool:
    ema_cfg = params.indicator_config.get("ema")
    if ema_cfg and ema_cfg[0]:
        _, fast, slow = ema_cfg
        col = f"ema_{fast}_{slow}"

        # если по какой-то причине колонки нет — не торгуем
        if col not in row or pd.isna(row[col]):
            return False

        if row[col] <= 0:
            return False

    rsi_cfg = params.indicator_config.get("rsi")
    if rsi_cfg and rsi_cfg[0]:
        _, period = rsi_cfg
        col = f"rsi_{period}"

        if col not in row or pd.isna(row[col]):
            return False

        if row[col] <= 0:
            return False

    # if params.indicator_config["volume"][0]:
    #     if "volume" in row and row["volume"] <= row["volume_ma"]:
    #         return False

    return True
