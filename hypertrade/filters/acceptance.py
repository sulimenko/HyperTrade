import pandas as pd

def _minutes_from_open(ts: pd.Timestamp) -> int | None:
    stamp = pd.Timestamp(ts)
    if stamp.tzinfo is None:
        stamp = stamp.tz_localize("UTC")
    ny = stamp.tz_convert("America/New_York")
    open_time = ny.normalize() + pd.Timedelta(hours=9, minutes=30)
    return int((ny - open_time).total_seconds() // 60)


def passes_time_gate(signal_time, params) -> bool:
    start = getattr(params, "accept_start_minute", None)
    end = getattr(params, "accept_end_minute", None)
    if start is None or end is None:
        return True
    minutes = _minutes_from_open(pd.Timestamp(signal_time))
    if minutes is None:
        return True
    return int(start) <= minutes <= int(end)


