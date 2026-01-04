import json
from pathlib import Path
from common.path_man import BASE_DIR

def load_artikels():
    config_path = BASE_DIR / "common" / "artikels.json"
    with open(config_path, "r") as f:
        data = json.load(f)
    return data["artikels"]

