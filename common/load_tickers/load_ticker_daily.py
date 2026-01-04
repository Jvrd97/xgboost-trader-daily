import requests
import pandas as pd
import time
from typing import Optional, List, Dict, Union
from tqdm import tqdm
import os
from pathlib import Path
from dotenv import load_dotenv
from common.path_man import *
from common.load_tickers.ticker_helper import back_days
from common.load_concat_merge import concat_single_ticker
# Load environment variables
load_dotenv()


# Base data directory (adjust to your setup)
BASE_DATA_PATH = Path("data")

# ==========================================
# Interval Configuration Dictionary
# ==========================================
INTERVAL_CONFIG = {

    "D": {
        "key": "D",
        "folder": DATA_DAY,
        "description": "Daily",
        "default_lookback": 365 * 4,
    }
}


def get_data_path(ticker: str, interval: str, base_path: Path = BASE_DATA_PATH) -> dict:
    """
    Dynamically create paths based on ticker and interval.
    
    Returns dict with path info and creates directory if needed.
    """
    interval_str = str(interval).upper() if str(interval).upper() in ["D", "W"] else str(interval)
    
    if interval_str not in INTERVAL_CONFIG:
        raise ValueError(f"Unknown interval: {interval}. Valid: {list(INTERVAL_CONFIG.keys())}")
    
    config = INTERVAL_CONFIG[interval_str]
    
    # Build paths
    data_dir = base_path / config["folder"] / ticker
    data_dir.mkdir(parents=True, exist_ok=True)
    
    return {
        "ticker": ticker,
        "interval": interval_str,
        "timeframe_key": config["key"],
        "description": config["description"],
        "default_lookback": config["default_lookback"],
        "data_dir": data_dir,
        "csv_path": data_dir / f"{ticker}_{config['key']}.csv",
    }



# ==========================================
# USER SETTINGS

# Make production look ticks 


TICKER = ""       # e.g., "WLD", "BTC", "ETH"
INTERVAL = "D"     # "60" (Hourly) or "D" (Daily)
LOOKBACK_DAYS = ""  # How far back to fetch
# ==========================================


paths = get_data_path(TICKER, INTERVAL)
# --- Logic to determine Environment Variable and Path ---
# Map Interval to the Key used in your .env file (60 -> H, D -> D)
if str(INTERVAL) == "60":
    TIMEFRAME_KEY = "H"
elif str(INTERVAL).upper() == "D":
    TIMEFRAME_KEY = "D"
else:
    TIMEFRAME_KEY = paths["timeframe_key"] 


ENV_VAR_NAME = f"temp_{TIMEFRAME_KEY}_{TICKER}"

# Get the folder path from .env
SAVE_FOLDER = BASE_TMP_PATH / INTERVAL / TICKER

# Construct Configuration Dictionary
CONFIG = {
    'symbol': f'{TICKER}USDT',
    'category': 'linear',
    'interval': str(INTERVAL),
    'use_bybit': True,
    'bybit_testnet': False,
    'lookback_days': LOOKBACK_DAYS,
    'save_folder': SAVE_FOLDER or str(paths["data_dir"])  # fallback to paths dict
}

class DataFetcher:
    BASE_URL = 'https://api.bybit.com'
    TESTNET_URL = 'https://api-testnet.bybit.com'

    def __init__(self, symbol: str, category: str = 'linear', testnet: bool = False):
        self.symbol = symbol
        self.category = category
        self.base_url = self.TESTNET_URL if testnet else self.BASE_URL
        self.kline_url = f"{self.base_url}/v5/market/kline"

    def _get_klines(self, params: Dict[str, Union[str, int]]) -> List[List[str]]:
        try:
            r = requests.get(self.kline_url, params=params, timeout=10)
            r.raise_for_status()
            data = r.json()
            if data.get('retCode') != 0:
                print(f"Bybit API Error: {data.get('retMsg')}")
                return []
            return data.get('result', {}).get('list', [])
        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}")
            return []

    def fetch_bybit_klines(self, interval: str, start: int, end: int, limit: int = 1000) -> pd.DataFrame:
        params = {
            'category': self.category,
            'symbol': self.symbol,
            'interval': interval,
            'limit': limit
        }
        
        # Calculate interval in milliseconds
        interval_map = {"D": 24 * 60, "W": 7 * 24 * 60, "M": 30 * 24 * 60}
        if interval.isnumeric():
            interval_ms = int(interval) * 60 * 1000
        else:
            interval_ms = interval_map.get(interval.upper(), 60) * 60 * 1000

        if start is None or end is None: return pd.DataFrame()

        klines = []
        chunk_size_ms = limit * interval_ms
        time_chunks = range(start, end, chunk_size_ms)
        
        print(f"Fetching {self.symbol} [{interval}] in {len(time_chunks)} batches...")

        for chunk_start in tqdm(time_chunks):
            chunk_end = min(chunk_start + chunk_size_ms, end)
            params['start'] = chunk_start
            params['end'] = chunk_end
            klines.extend(self._get_klines(params))
            time.sleep(0.1) # Rate limit protection

        if not klines: return pd.DataFrame()

        cols = ['open_time', 'open', 'high', 'low', 'close', 'volume', 'turnover']
        if len(klines[0]) > len(cols): klines = [k[:len(cols)] for k in klines]
            
        df = pd.DataFrame(klines, columns=cols)
        df["open_time"] = pd.to_datetime(df["open_time"].astype(int), unit='ms')
        for c in ['open', 'high', 'low', 'close', 'volume']:
            df[c] = df[c].astype(float)
            
        df = df.set_index('open_time').sort_index()
        df.index.name = 'timestamp'
        return df[~df.index.duplicated(keep='first')][['open', 'high', 'low', 'close', 'volume']]

    def save_csv(self, df: pd.DataFrame, folder: str, filename: str):
        if not folder:
            print(f"CRITICAL ERROR: Folder path is None. Check .env for {ENV_VAR_NAME}")
            return
            
        path_obj = Path(folder)
        path_obj.mkdir(parents=True, exist_ok=True) # Create folder if missing
        
        full_path = path_obj / filename
        df.to_csv(full_path)
        print(f"✅ Data saved to: {full_path}")



# to make it usable the way we can do it is wrap our class and other functions into function
# At the end of load_ticker_daily.py, replace the __main__ block with:

def fetch_and_update_ticker(ticker: str, interval: str) -> pd.DataFrame:
    """
    Fetches latest data from Bybit and updates concat file.
    
    Args:
        ticker: e.g., "BTC", "WLD"
        interval: e.g., "D", "60", "5"
    
    Returns:
        Updated DataFrame with complete history
    """
    print(f"\n{'='*50}")
    print(f"FETCHING {ticker} [{interval}]")
    print(f"{'='*50}")
    
    # Calculate days back
    file_to_read = BASE_CONCAT_PATH / interval / ticker / f"{ticker}_{interval}_concat.csv"
    
    if not file_to_read.exists():

        print(f"✗ Concat file not found: {file_to_read}")
        fallback_folder = BASE_CONCAT_PATH / interval / ticker
        fallback_name = f"{ticker}_{interval}_concat.csv"
        print(f"downloading file into {fallback_folder}")
        fall_back_days = 365 * 4
        # Fallbaack
        end_ts_full = int(time.time() * 1000)
        start_ts_full = end_ts_full - (fall_back_days * 24 * 60 * 60 * 1000)
        config2 = {
        'symbol': f'{ticker}USDT',
        'category': 'linear',
        'interval': str(interval)
    }        
        fetcher2 = DataFetcher(symbol=config2['symbol'], category=config2['category'])

        

        df_new_full = fetcher2.fetch_bybit_klines(
        interval=config2['interval'],
        start=start_ts_full,
        end=end_ts_full
    )
        fetcher2.save_csv(df_new_full, fallback_folder,fallback_name )
    
    file_to_read = BASE_CONCAT_PATH / interval / ticker / f"{ticker}_{interval}_concat.csv"

    df_existing = pd.read_csv(file_to_read)
    days_back = back_days(df_existing)
    
    print(f"📅 Days to fetch: {days_back}")
    
    # Setup paths
    save_folder = BASE_TMP_PATH / interval / ticker
    save_folder.mkdir(parents=True, exist_ok=True)
    
    # Config
    config = {
        'symbol': f'{ticker}USDT',
        'category': 'linear',
        'interval': str(interval),
        'lookback_days': days_back,
        'save_folder': str(save_folder)
    }
    
    # Fetch data
    fetcher = DataFetcher(symbol=config['symbol'], category=config['category'])
    
    end_ts = int(time.time() * 1000)
    start_ts = end_ts - (days_back * 24 * 60 * 60 * 1000)
    
    df = fetcher.fetch_bybit_klines(
        interval=config['interval'],
        start=start_ts,
        end=end_ts
    )
    
    if df.empty:
        print("✗ No data fetched from Bybit")
        return df_existing
    
    # Save to TMP
    file_name = f"{ticker}_{interval}_raw.csv"
    fetcher.save_csv(df, config['save_folder'], file_name)
    
    print("\nNew data preview:")
    print(df.tail(3))
    
    # Update concat file
    print("\n" + "="*50)
    print("UPDATING CONCAT FILE...")
    print("="*50)
    
    df_updated, tmp_file = concat_single_ticker(ticker, interval)
    
    # Cleanup
    if tmp_file and os.path.exists(tmp_file):
        os.remove(tmp_file)
        print(f"✓ Cleaned up: {os.path.basename(tmp_file)}")
    
    if not df_updated.empty:
        print(f"✅ SUCCESS! Latest: {df_updated['timestamp'].max()}")
        print(f"   Total rows: {len(df_updated)}")
        return df_updated
    else:
        print("⚠ Concat returned empty DataFrame")
        return df_existing


# Keep this for standalone testing
if __name__ == "__main__":
    result = fetch_and_update_ticker("BTC", "D")
    print(f"\nFinal DataFrame shape: {result.shape}")


            

            