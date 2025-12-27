from datetime import datetime, timedelta
from config.market_time import NYSE_OPEN, NYSE_TZ, UTC

def calc_entry_datetime(signal_dt, minutes_from_open=0):
    ny_date = signal_dt.tz_localize('UTC').astimezone(NYSE_TZ).date()

    ny_open_dt = datetime.combine(ny_date, NYSE_OPEN)
    ny_open_dt = NYSE_TZ.localize(ny_open_dt)

    entry_dt = ny_open_dt + timedelta(minutes=minutes_from_open)

    return entry_dt.astimezone(UTC)
