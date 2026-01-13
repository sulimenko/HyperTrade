from datetime import datetime, time, timedelta
import pandas as pd
import pytz
import numpy as np
import exchange_calendars as xcals

XNYS = xcals.get_calendar("XNYS")
UTC = pytz.UTC


def _to_utc_ts(dt: datetime) -> pd.Timestamp:
    ts = pd.Timestamp(dt)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.tz_convert("UTC")


def build_market_cache(dt_min, dt_max, extra_days=30):
    """
    Готовит кэш расписания XNYS на диапазон дат (один раз на прогон).
    dt_min/dt_max: datetime|Timestamp (любые, приводим к UTC).
    extra_days: запас по датам (чтобы хватило на holding/delay).
    Возвращает dict с numpy-массивами ns:
      opens_ns, closes_ns, sess_minutes, cum_minutes
    """
    t0 = _to_utc_ts(dt_min).normalize() - pd.Timedelta(days=10)
    t1 = _to_utc_ts(dt_max).normalize() + pd.Timedelta(days=int(extra_days))

    # schedule DataFrame на диапазон
    sched = XNYS.schedule.loc[t0.date():t1.date()]

    # в exchange_calendars встречаются разные имена колонок
    if "open" in sched.columns and "close" in sched.columns:
        opens = sched["open"]
        closes = sched["close"]
    else:
        opens = sched["market_open"]
        closes = sched["market_close"]

    # один раз конвертируем в UTC и в int64 ns
    opens_ns = opens.dt.tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)
    closes_ns = closes.dt.tz_convert("UTC").values.astype("datetime64[ns]").astype(np.int64)

    sess_minutes = ((closes_ns - opens_ns) // 60_000_000_000).astype(np.int64)

    # cum_minutes[i] = минут до начала i-й сессии (i по opens_ns)
    cum_minutes = np.empty(sess_minutes.size + 1, dtype=np.int64)
    cum_minutes[0] = 0
    np.cumsum(sess_minutes, out=cum_minutes[1:])

    return {
        "opens_ns": opens_ns,
        "closes_ns": closes_ns,
        "sess_minutes": sess_minutes,
        "cum_minutes": cum_minutes,
    }


def add_market_minutes_cached(t_ns: int, minutes: int, cache) -> int:
    """
    Быстро добавляет market minutes к t_ns (UTC ns) по готовому cache.
    Возвращает UTC ns.
    """
    remaining = int(minutes)
    if remaining <= 0:
        return int(t_ns)

    opens_ns = cache["opens_ns"]
    closes_ns = cache["closes_ns"]
    cum = cache["cum_minutes"]

    # первая сессия, у которой close > t
    i = int(np.searchsorted(closes_ns, t_ns, side="right"))
    if i >= closes_ns.size:
        return int(t_ns)

    seg_start = t_ns
    if seg_start < opens_ns[i]:
        seg_start = int(opens_ns[i])

    avail = int((closes_ns[i] - seg_start) // 60_000_000_000)
    if avail < 0:
        avail = 0

    if remaining <= avail:
        return int(seg_start + remaining * 60_000_000_000)

    remaining -= avail

    base = i + 1
    if base > opens_ns.size:
        return int(closes_ns[-1])

    target = int(cum[base] + remaining)

    j = int(np.searchsorted(cum, target, side="right") - 1)
    if j >= opens_ns.size:
        return int(closes_ns[-1])

    offset = int(target - cum[j])
    if offset == 0 and j > 0:
        return int(closes_ns[j - 1])
    
    return int(opens_ns[j] + offset * 60_000_000_000)


def compute_entry_time_cached(signal_dt: datetime, delay_minutes: int, cache) -> datetime:
    """
    Быстрый entry_dt по cache:
      - если сигнал до открытия: base=open
      - если после закрытия/неторговое: base=следующее open
      - если внутри: base=signal
    delay_minutes добавляется как market minutes.
    """
    signal_ns = int(_to_utc_ts(signal_dt).value)

    opens_ns = cache["opens_ns"]
    closes_ns = cache["closes_ns"]

    i = int(np.searchsorted(closes_ns, signal_ns, side="right"))
    if i >= closes_ns.size:
        # вне диапазона кэша
        return pd.Timestamp(signal_ns, tz="UTC").to_pydatetime()

    open_ns = int(opens_ns[i])
    close_ns = int(closes_ns[i])

    if signal_ns < open_ns:
        base_ns = open_ns
    elif signal_ns >= close_ns:
        # следующая сессия
        if i + 1 < opens_ns.size:
            base_ns = int(opens_ns[i + 1])
        else:
            base_ns = close_ns
    else:
        base_ns = signal_ns

    entry_ns = add_market_minutes_cached(base_ns, int(delay_minutes), cache)
    return pd.Timestamp(entry_ns, tz="UTC").to_pydatetime()


# -----------------
# Оставляем старые (fallback) на всякий случай
# -----------------

def add_market_minutes(entry_ts_utc: pd.Timestamp, minutes: int) -> pd.Timestamp:
    # старый медленный вариант оставляем как fallback
    t = pd.Timestamp(entry_ts_utc)
    if t.tzinfo is None:
        t = t.tz_localize("UTC")
    else:
        t = t.tz_convert("UTC")

    remaining = int(minutes)
    if remaining <= 0:
        return t

    while remaining > 0:
        horizon_days = max(360, int(remaining / 300) + 30)
        start = t.normalize() - pd.Timedelta(days=10)
        end = t.normalize() + pd.Timedelta(days=horizon_days)
        sched = XNYS.schedule.loc[start.date():end.date()]

        advanced = False
        for _, row in sched.iterrows():
            if "open" in row:
                open_utc = row["open"].tz_convert("UTC")
                close_utc = row["close"].tz_convert("UTC")
            else:
                open_utc = row["market_open"].tz_convert("UTC")
                close_utc = row["market_close"].tz_convert("UTC")

            if close_utc <= t:
                continue

            seg_start = open_utc if t < open_utc else t
            seg_end = close_utc

            seg_minutes = int((seg_end - seg_start).total_seconds() // 60)
            if seg_minutes <= 0:
                continue

            if remaining <= seg_minutes:
                return seg_start + pd.Timedelta(minutes=remaining)

            remaining -= seg_minutes
            t = close_utc
            advanced = True

        if not advanced:
            break

    return t


def compute_entry_time(signal_dt: datetime, delay_minutes: int) -> datetime:
    # старый вариант по XNYS (медленнее) оставляем
    signal_utc = _to_utc_ts(signal_dt)
    day = signal_utc.normalize()

    if not XNYS.is_session(day.date()):
        next_sess = XNYS.next_session(day.date())
        open_utc = XNYS.session_open(next_sess).tz_convert("UTC")
        close_utc = XNYS.session_close(next_sess).tz_convert("UTC")
    else:
        open_utc = XNYS.session_open(day.date()).tz_convert("UTC")
        close_utc = XNYS.session_close(day.date()).tz_convert("UTC")

    if signal_utc < open_utc:
        base = open_utc
    elif signal_utc >= close_utc:
        next_sess = XNYS.next_session(open_utc.date())
        base = XNYS.session_open(next_sess).tz_convert("UTC")
    else:
        base = signal_utc

    entry_utc = add_market_minutes(base, int(delay_minutes))
    return entry_utc.to_pydatetime()
