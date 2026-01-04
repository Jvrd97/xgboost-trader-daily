import sys
import os
from datetime import datetime
from dotenv import load_dotenv
from common.path_man import *

load_dotenv()

import time

INTERVAL = "5" 

path_test = TEST_DATA / INTERVAL / "WLD"
files = list(path_test.glob("*.csv"))
range_of_train = len(files)
 
 
 
# 1. Ensure Python finds your modules
# Gets the current folder (MERTE_PRODUCTION)
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 2. Import your functions
try:
    from common.move_one_file import move_one_file
    
    # Note: Ensure you created the __init__.py files if imports fail, 
    # but in Python 3 it usually works without them.
    from models.intervals.xgboost.wld_5min import wld_daily_train
except ImportError as e:
    print(f"Import Error: {e}")
    print("Make sure you exported the notebook to .py and checked the folder paths!")
    sys.exit(1)

def main_func():
    print(f"[{datetime.now()}] Pipeline Started")
    
    try:
        # Step 1: Data Movement
        print(f"[{datetime.now()}] --- Moving Files ---")
        
        time.sleep(0.5)
        
        # Step 2: Retraining
        print(f"[{datetime.now()}] --- Retraining Model ---")
        wld_daily_train()
        time.sleep(0.5)
        
        print(f"[{datetime.now()}] Pipeline Finished Successfully")
        
    except Exception as e:
        print(f"[{datetime.now()}] CRITICAL ERROR: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    for i in range(range_of_train):
        print(f"\n=== RUN {i+1}/150 ===\n")
        main_func()
        time.sleep(1)
   
