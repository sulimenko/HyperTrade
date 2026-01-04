from datetime import datetime, time, timedelta
import pytz


def compute_entry_time(signal_dt: datetime, delay_minutes: int) -> datetime:
    eastern = pytz.timezone("US/Eastern")
    utc = pytz.UTC

    signal_et = signal_dt.astimezone(eastern)
    trade_date_et = signal_et.date()

    # trade_date = signal_dt.date() + timedelta(days=1)
    market_open_et = eastern.localize(datetime.combine(trade_date_et, time(9, 30)))
    market_close_et = eastern.localize(datetime.combine(trade_date_et, time(16, 0)))

    if signal_et < market_open_et:
        # сигнал ДО открытия → считаем от открытия
        entry_et = market_open_et + timedelta(minutes=delay_minutes)
    elif signal_et > market_close_et:
        # сигнал ПОСЛЕ закрытия → перенос на следующее открытие
        next_day = trade_date_et + timedelta(days=1)
        next_open = eastern.localize(datetime.combine(next_day, time(9, 30)))
        entry_et = next_open + timedelta(minutes=delay_minutes)
    else:
        # рынок уже работает → считаем от сигнала
        entry_et = signal_et + timedelta(minutes=delay_minutes)

    return entry_et.astimezone(utc)