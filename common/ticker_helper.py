
import pandas as pd 
import datetime
from datetime import datetime

def back_days(df):
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    end_date = df['timestamp'].max()
    back_days = (datetime.now() - end_date).days

    return back_days
