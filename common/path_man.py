from pathlib import Path
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = Path(__file__).parent.parent.resolve() # on
BASE_PRED_PATH = BASE_DIR / "predictions" 
BASE_PRED_PATH_5MIN = BASE_DIR / "predictions" /  "5min"

BASE_PRED_PATH_EXPORT = BASE_DIR / "predictions" / "export_tmp"

BASE_PRED_PATH_SENT = BASE_DIR / "predictions" / "sent"

BASE_CONCAT_PATH = BASE_DIR / "data" / "concat"
BASE_TMP_PATH = BASE_DIR / "data" / "data_added_tmp"
BASE_RAW_PATH = BASE_DIR / "data" / "raw"

DATA_DAY = BASE_DIR / "data" / "daily_data"
DATA_60MIN = BASE_DIR / "data" / "60m_data"
DATA_30MIN = BASE_DIR / "data" / "30m_data"
DATA_15MIN = BASE_DIR / "data" / "15m_data"
DATA_5MIN = BASE_DIR / "data" / "5m_data"
DATA_1MIN = BASE_DIR / "data" / "1m_data"

TEST_DATA = BASE_DIR / "data" / "test_data"

MERGE_DATA = BASE_DIR / "data" / "merge"