import pandas as pd
import ta


def calculate_indicators(df: pd.DataFrame, indicator_config: dict) -> pd.DataFrame:
    indicators = pd.DataFrame({"datetime": df["datetime"]})

    close = df["close"].astype(float)
    
    # ===== EMA spread (EMA_fast - EMA_slow) =====
    ema_cfg = indicator_config.get("ema")
    if ema_cfg and len(ema_cfg) >= 4 and ema_cfg[0]:
        _, _, fast, slow = ema_cfg
        if fast is None or slow is None:
            raise ValueError("EMA включен, но fast/slow не заданы")
        ema_fast = close.ewm(span=int(fast), adjust=False).mean()
        ema_slow = close.ewm(span=int(slow), adjust=False).mean()
        indicators[f"ema_{int(fast)}_{int(slow)}"] = ema_fast - ema_slow

    # ===== RSI =====
    rsi_cfg = indicator_config.get("rsi")
    if rsi_cfg and len(rsi_cfg) >= 4 and rsi_cfg[0]:
        _, _, _, period = rsi_cfg
        if period is None:
            raise ValueError("RSI включен, но period не задан")
        indicators[f"rsi_{int(period)}"] = ta.momentum.RSIIndicator(close, int(period)).rsi()

    # # ===== VOLUME =====
    # volume_cfg = indicator_config.get("volume")
    # if volume_cfg and volume_cfg[0]:
    #     indicators["volume_ma"] = df["volume"].rolling(20).mean()

    return indicators
