from datetime import datetime
from datetime import timedelta
import pytz

def get_latest_market_close():
    now = datetime.now(pytz.timezone('US/Eastern'))
    today = now.date()
    market_close = now.replace(hour=16, minute=10, second=0, microsecond=0)
    
    if now < market_close:
        # return today - timedelta(days=1)
        return today 
    else:
        return today