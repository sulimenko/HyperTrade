import pandas as pd
import ta


def calculate_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    indicators = pd.DataFrame({"datetime": df["datetime"]})

    # ===== EMA =====
    ema_cfg = indicator_config.get("ema")
    if ema_cfg and ema_cfg[0]:
        _, fast, slow = ema_cfg
        indicators[f"ema_{fast}_{slow}"] = df["close"].ewm(span=fast, adjust=False).mean() - df["close"].ewm(span=slow, adjust=False).mean()

    # ===== RSI =====
    rsi_cfg = indicator_config.get("rsi")
    if rsi_cfg and rsi_cfg[0]:
        _, period = rsi_cfg
        indicators[f"rsi_{period}"] = ta.momentum.RSIIndicator(df["close"], period).rsi()

    # # ===== VOLUME =====
    # volume_cfg = indicator_config.get("volume")
    # if volume_cfg and volume_cfg[0]:
    #     indicators["volume_ma"] = df["volume"].rolling(20).mean()

    return indicators
