import time
from datetime import datetime
from common.path_man import *
from common.move_one_file import move_one_file
from models.intervals.lstm.lstm1 import wld_5min_pipeline

INTERVAL = "5"
MAIN_TICKER = "WLD"
SECONDARY_TICKER = "BTC"

# Count test files
path_test = TEST_DATA / INTERVAL / MAIN_TICKER
files = sorted(path_test.glob("*.csv"))
n_iterations = len(files)

print(f"📁 Found {n_iterations} test files")
print(f"⏱️  Model retrains weekly, predictions are fast!\n")


def run_iteration(i: int):
    print(f"\n{'='*50}")
    print(f"ITERATION {i+1}/{n_iterations}")
    print(f"{'='*50}")
    
    # Move new data files
    move_one_file(MAIN_TICKER, INTERVAL)
    move_one_file(SECONDARY_TICKER, INTERVAL)
    
    # Run pipeline (training happens only when needed!)
    wld_5min_pipeline()


if __name__ == "__main__":
    start = time.time()
    
    for i in range(n_iterations):
        run_iteration(i)
        time.sleep(0.2)
    
    elapsed = time.time() - start
    print(f"\n✓ Completed {n_iterations} iterations in {elapsed:.1f}s")
    print(f"  Average: {elapsed/n_iterations:.2f}s per iteration")