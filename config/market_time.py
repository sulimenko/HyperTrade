from datetime import time
import pytz

NYSE_TZ = pytz.timezone("America/New_York")
UTC = pytz.utc

NYSE_OPEN = time(9, 30)
NYSE_CLOSE = time(16, 0)
