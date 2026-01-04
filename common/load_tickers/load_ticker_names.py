import json
from pathlib import Path
from common.path_man import BASE_DIR

def load_ticker_normall():
    config_path = BASE_DIR / "common" / "load_tickers" / "altcoins_normal.json"
    with open(config_path, "r") as f:
        data = json.load(f)
    return data["tickers"]

def load_ticker_small():
    config_path = BASE_DIR / "common" / "load_tickers" / "altcoins_small.json"
    with open(config_path, "r") as f:
        data = json.load(f)
    return data["tickers"]