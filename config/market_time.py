from datetime import datetime, time, timedelta
import pytz


def compute_entry_time(signal_dt, minutes_from_open):
    eastern = pytz.timezone("US/Eastern")

    trade_date = signal_dt.date() + timedelta(days=1)
    market_open = eastern.localize(
        datetime.combine(trade_date, time(9, 30))
    )

    return market_open.astimezone(pytz.UTC) + timedelta(
        minutes=minutes_from_open
    )
