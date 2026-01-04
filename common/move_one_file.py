# movers/move_one_file.py
import os
import shutil
from common.path_man import *


def move_one_file(ticker: str, interval: str = "5"):
    """Move one file from TEST_DATA to TMP folder."""
    
    # Source: test data folder
    RAW_FOLDER = TEST_DATA / interval / ticker
    
    # Destination: tmp folder for processing
    TMP_FOLDER = BASE_TMP_PATH / interval / ticker
    
    print(f"\n[{ticker}]")
    print(f"  RAW: {RAW_FOLDER}")
    print(f"  TMP: {TMP_FOLDER}")
    
    # Check source folder
    if not RAW_FOLDER.exists():
        print(f"  ✗ RAW folder not found!")
        return False
    
    # Create TMP if missing
    TMP_FOLDER.mkdir(parents=True, exist_ok=True)
    
    # Find files
    files = sorted([f for f in os.listdir(RAW_FOLDER) if f.endswith('.csv')])
    
    if not files:
        print(f"  ✗ No files to move")
        return False
    
    # Move first file
    file_to_move = files[0]
    src = RAW_FOLDER / file_to_move
    dst = TMP_FOLDER / file_to_move
    
    shutil.move(str(src), str(dst))
    print(f"  ✓ Moved: {file_to_move}")
    print(f"  Remaining: {len(files) - 1}")
    
    return True


if __name__ == "__main__":
    INTERVAL = "5"  # Change as needed: "5", "15", "60", "D"
    
    move_one_file("WLD", interval=INTERVAL)
    move_one_file("BTC", interval=INTERVAL)