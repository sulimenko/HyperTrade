from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from hypertrade.config.strategy import indicator_enabled, indicator_settings
from hypertrade.features.indicator_engine import ensure_market_data
from hypertrade.simulation.market_time import compute_entry_time

LONG = 1
SHORT = -1


def _split_symbols(value) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(",") if part.strip()]


def load_signals(path: str) -> list[dict]:
    df = pd.read_csv(path, sep=";", parse_dates=["datetime"])
    df["datetime"] = pd.to_datetime(df["datetime"], format="%d.%m.%Y %H:%M:%S", utc=True)

    has_long = "long_symbols" in df.columns
    has_short = "short_symbols" in df.columns
    signals = []
    for _, row in df.iterrows():
        long_symbols = _split_symbols(row["long_symbols"]) if has_long else _split_symbols(row.get("symbols"))
        short_symbols = _split_symbols(row["short_symbols"]) if has_short else []
        signals.append({"datetime": row["datetime"], "long": long_symbols, "short": short_symbols})
    return signals


def load_external_signals(path: str) -> list[dict]:
    return load_signals(path)


def _get_confirm_slice(ohlc: pd.DataFrame, entry_idx: int, confirm_bars: int) -> pd.DataFrame:
    if confirm_bars is None or int(confirm_bars) <= 1:
        return ohlc.iloc[entry_idx : entry_idx + 1]
    cb = int(confirm_bars)
    start = max(0, entry_idx - cb + 1)
    return ohlc.iloc[start : entry_idx + 1]


def filters(ohlc: pd.DataFrame, entry_idx: int, params, direction: int) -> bool:
    cfg = params.indicator_config
    win = _get_confirm_slice(ohlc, entry_idx, int(getattr(params, "confirm_bars", 1)))

    ema_cfg = indicator_settings(cfg, "ema")
    if indicator_enabled(cfg, "ema"):
        sign = ema_cfg.get("sign")
        fast = ema_cfg.get("fast")
        slow = ema_cfg.get("slow")
        col_fast = f"ema_{int(fast)}"
        col_slow = f"ema_{int(slow)}"
        if col_fast not in win.columns or col_slow not in win.columns:
            return False
        if win[col_fast].isna().any() or win[col_slow].isna().any():
            return False
        ok = (win[col_fast] > win[col_slow]).all() if sign == "above" else (win[col_fast] < win[col_slow]).all()
        if not ok:
            return False

    rsi_cfg = indicator_settings(cfg, "rsi")
    if indicator_enabled(cfg, "rsi"):
        sign = rsi_cfg.get("sign")
        level = rsi_cfg.get("level")
        period = rsi_cfg.get("period")
        col = f"rsi_{int(period)}"
        if col not in win.columns or win[col].isna().any():
            return False
        if sign == "above" and not (win[col] > float(level)).all():
            return False
        if sign != "above" and not (win[col] < float(level)).all():
            return False

    adx_cfg = indicator_settings(cfg, "adx")
    if indicator_enabled(cfg, "adx"):
        adx_sign = adx_cfg.get("mode")
        adx_min = adx_cfg.get("minimum")
        period = int(adx_cfg.get("period"))
        cols = [f"adx_{period}", f"di_plus_{period}", f"di_minus_{period}"]
        for col in cols:
            if col not in win.columns or win[col].isna().any():
                return False
        adx_min = float(adx_min) if adx_min is not None else 0.0
        if adx_sign == "trend":
            if not (win[f"adx_{period}"] >= adx_min).all():
                return False
        elif adx_sign == "range":
            if not (win[f"adx_{period}"] <= adx_min).all():
                return False
        else:
            return False
        if direction == LONG and not (win[f"di_plus_{period}"] > win[f"di_minus_{period}"]).all():
            return False
        if direction == SHORT and not (win[f"di_minus_{period}"] > win[f"di_plus_{period}"]).all():
            return False

    macd_cfg = indicator_settings(cfg, "macd")
    if indicator_enabled(cfg, "macd"):
        sign = macd_cfg.get("sign")
        fast = macd_cfg.get("fast")
        slow = macd_cfg.get("slow")
        sig = macd_cfg.get("signal")
        col_m = f"macd_{int(fast)}_{int(slow)}_{int(sig)}"
        col_s = f"macds_{int(fast)}_{int(slow)}_{int(sig)}"
        if col_m not in win.columns or col_s not in win.columns:
            return False
        if win[col_m].isna().any() or win[col_s].isna().any():
            return False
        if direction == LONG:
            ok = (win[col_m] > win[col_s]).all() if sign == "above" else (win[col_m] < win[col_s]).all()
        else:
            ok = (win[col_m] < win[col_s]).all() if sign == "above" else (win[col_m] > win[col_s]).all()
        if not ok:
            return False

    bb_cfg = indicator_settings(cfg, "bb")
    if indicator_enabled(cfg, "bb"):
        period = bb_cfg.get("period")
        n_std = bb_cfg.get("std")
        col_p = f"bb_p_{int(period)}_{float(n_std):g}"
        if col_p not in win.columns or win[col_p].isna().any():
            return False
        bb_max_p = float(getattr(params, "bb_max_p", 0.95))
        bb_min_p = float(getattr(params, "bb_min_p", 0.05))
        if direction == LONG and not (win[col_p] < bb_max_p).all():
            return False
        if direction == SHORT and not (win[col_p] > bb_min_p).all():
            return False

    vwap_cfg = indicator_settings(cfg, "vwap")
    if indicator_enabled(cfg, "vwap"):
        vwap_sign = vwap_cfg.get("sign")
        vwap_k = vwap_cfg.get("threshold")
        if "vwap" not in win.columns or win["vwap"].isna().any():
            return False
        ratio = (win["close"].astype(float) / win["vwap"].astype(float).replace(0.0, np.nan)).replace([np.inf, -np.inf], np.nan)
        if ratio.isna().any():
            return False
        vwap_k = float(vwap_k) if vwap_k is not None else 1.0
        if direction == LONG:
            if vwap_sign == "above" and not (ratio >= vwap_k).all():
                return False
            if vwap_sign != "above" and not (ratio <= vwap_k).all():
                return False
        else:
            if vwap_sign == "below" and not (ratio <= vwap_k).all():
                return False
            if vwap_sign != "below" and not (ratio >= vwap_k).all():
                return False

    vol_cfg = indicator_settings(cfg, "volume")
    if indicator_enabled(cfg, "volume"):
        vol_sign = vol_cfg.get("sign")
        ma_period = vol_cfg.get("ma_period")
        vol_k = vol_cfg.get("threshold")
        col = f"vol_ratio_{int(ma_period)}"
        if col not in win.columns or win[col].isna().any():
            return False
        if vol_sign == "above" and not (win[col] >= float(vol_k)).all():
            return False
        if vol_sign != "above" and not (win[col] <= float(vol_k)).all():
            return False

    atr_cfg = indicator_settings(cfg, "atr")
    if indicator_enabled(cfg, "atr"):
        period = atr_cfg.get("period")
        col = f"atr_{int(period)}"
        if col not in win.columns or win[col].isna().any():
            return False
        atr_pct = (win[col] / win["close"]).astype(float)
        if not ((atr_pct >= float(getattr(params, "atr_min_pct", 0.0))) & (atr_pct <= float(getattr(params, "atr_max_pct", 1e9)))).all():
            return False

    don_cfg = indicator_settings(cfg, "donchian")
    if indicator_enabled(cfg, "donchian"):
        period = don_cfg.get("period")
        hcol = f"don_h_{int(period)}"
        lcol = f"don_l_{int(period)}"
        if hcol not in win.columns or lcol not in win.columns:
            return False
        if win[hcol].isna().any() or win[lcol].isna().any():
            return False
        pad = 0.002
        last = win.iloc[-1]
        if direction == LONG and last["close"] >= last[hcol] * (1.0 - pad):
            return False
        if direction == SHORT and last["close"] <= last[lcol] * (1.0 + pad):
            return False

    return True


@dataclass
class AcceptedSignal:
    symbol: str
    direction: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    rank_score: float


def _minutes_from_open(ts: pd.Timestamp) -> int | None:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    ny = stamp.tz_convert("America/New_York")
    open_time = ny.normalize() + pd.Timedelta(hours=9, minutes=30)
    return int((ny - open_time).total_seconds() // 60)


def _passes_time_gate(signal_time, params) -> bool:
    start = getattr(params, "accept_start_minute", None)
    end = getattr(params, "accept_end_minute", None)
    if start is None or end is None:
        return True
    minutes = _minutes_from_open(pd.Timestamp(signal_time))
    if minutes is None:
        return True
    return int(start) <= minutes <= int(end)


def _rank_score(ohlc: pd.DataFrame, entry_idx: int, params, direction: int) -> float:
    ranking_mode = getattr(params, "ranking_mode", "none")
    if ranking_mode == "none":
        return 0.0

    score = 0.0
    row = ohlc.iloc[entry_idx]
    cfg = params.indicator_config

    ema_cfg = indicator_settings(cfg, "ema")
    if indicator_enabled(cfg, "ema"):
        fast = ema_cfg.get("fast")
        slow = ema_cfg.get("slow")
        fast_col = f"ema_{int(fast)}"
        slow_col = f"ema_{int(slow)}"
        if fast_col in ohlc.columns and slow_col in ohlc.columns:
            diff = float(row[fast_col] - row[slow_col])
            score += diff if direction == LONG else -diff

    rsi_cfg = indicator_settings(cfg, "rsi")
    if indicator_enabled(cfg, "rsi"):
        level = rsi_cfg.get("level")
        period = rsi_cfg.get("period")
        col = f"rsi_{int(period)}"
        if col in ohlc.columns:
            score += abs(float(row[col]) - float(level)) * 0.1

    adx_cfg = indicator_settings(cfg, "adx")
    if indicator_enabled(cfg, "adx"):
        period = adx_cfg.get("period")
        col = f"adx_{int(period)}"
        if col in ohlc.columns:
            score += float(row[col]) * 0.05

    vol_cfg = indicator_settings(cfg, "volume")
    if indicator_enabled(cfg, "volume"):
        ma_period = vol_cfg.get("ma_period")
        col = f"vol_ratio_{int(ma_period)}"
        if col in ohlc.columns and pd.notna(row[col]):
            score += float(row[col]) * 0.2

    return score


def apply_signal_policy(signals: list[dict], params) -> tuple[list[dict], list[dict]]:
    accepted_signals = []
    policy_stats = []
    last_entry_per_symbol: dict[str, pd.Timestamp] = {}

    for signal in signals:
        processed = {"datetime": signal["datetime"], "long": [], "short": []}
        signal_policy_stats = {
            "datetime": signal["datetime"],
            "input_long": len(signal.get("long", [])),
            "input_short": len(signal.get("short", [])),
            "accepted_long": 0,
            "accepted_short": 0,
            "rejected_time_gate": 0,
            "rejected_cooldown": 0,
            "rejected_no_data": 0,
        }

        for label, direction in (("long", LONG), ("short", SHORT)):
            candidates: list[AcceptedSignal] = []
            for symbol in signal.get(label, []):
                if not _passes_time_gate(signal["datetime"], params):
                    signal_policy_stats["rejected_time_gate"] += 1
                    continue

                entry_time = pd.Timestamp(compute_entry_time(signal["datetime"], int(getattr(params, "delay_open", 0))))
                previous_entry = last_entry_per_symbol.get(symbol)
                cooldown_minutes = int(getattr(params, "cooldown_minutes", 0) or 0)
                if previous_entry is not None and cooldown_minutes > 0:
                    delta_minutes = (entry_time - previous_entry).total_seconds() / 60.0
                    if delta_minutes < cooldown_minutes:
                        signal_policy_stats["rejected_cooldown"] += 1
                        continue

                ohlc = ensure_market_data(symbol, signal["datetime"], params.indicator_config)
                if ohlc is None or ohlc.empty:
                    signal_policy_stats["rejected_no_data"] += 1
                    continue
                entry_idx = ohlc["datetime"].searchsorted(entry_time)
                if entry_idx >= len(ohlc):
                    signal_policy_stats["rejected_no_data"] += 1
                    continue

                candidates.append(
                    AcceptedSignal(
                        symbol=symbol,
                        direction=direction,
                        signal_time=pd.Timestamp(signal["datetime"]),
                        entry_time=entry_time,
                        rank_score=_rank_score(ohlc, int(entry_idx), params, direction),
                    )
                )

            candidates.sort(key=lambda item: item.rank_score, reverse=True)
            max_per_side = getattr(params, "max_signals_per_side", None)
            if max_per_side is not None:
                candidates = candidates[: int(max_per_side)]
            processed[label] = [candidate.symbol for candidate in candidates]
            for candidate in candidates:
                last_entry_per_symbol[candidate.symbol] = candidate.entry_time

            signal_policy_stats[f"accepted_{label}"] = len(candidates)

        accepted_signals.append(processed)
        policy_stats.append(signal_policy_stats)

    return accepted_signals, policy_stats
