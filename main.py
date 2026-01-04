import sys
import os
from datetime import datetime
from dotenv import load_dotenv
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

load_dotenv()

from common.load_tickers.load_ticker_daily import fetch_and_update_ticker
from models.retrain.daily.BTC.BTC_D import btc_daily_train
from common.trade_pipeline import execute_trade_signal


# ============================================================
# CONFIGURATION
# ============================================================
TICKER = "WLD"           # Trading ticker
INTERVAL = "D"           # Time interval (D = Daily)
TRADE_AMOUNT = 30  # USDT per trade, we have 10x leverage 
USE_TESTNET = False      # True = Testnet, False = Mainnet (BE CAREFUL!)

# ============================================================
# PIPELINE FUNCTIONS
# ============================================================

def data_pipeline(ticker: str, interval: str) -> bool:
    """
    Step 1: Fetch latest data and update concat file
    Returns: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"[1/3] DATA PIPELINE")
    print(f"{'='*60}")
    
    try:
        df = fetch_and_update_ticker(ticker, interval)
        
        if df.empty:
            print("✗ No data fetched")
            return False
        
        print(f"✅ Data updated successfully")
        print(f"   Rows: {len(df)}")
        print(f"   Latest: {df['timestamp'].max()}")
        return True
        
    except Exception as e:
        print(f"✗ Data pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def training_pipeline(ticker: str, interval: str) -> bool:
    """
    Step 2: Train model and generate prediction
    Returns: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"[2/3] TRAINING PIPELINE")
    print(f"{'='*60}")
    
    try:
        # Call the appropriate training function
        btc_daily_train(ticker=ticker, interval=interval)  # This also saves prediction JSON
        
        print(f"✅ Model trained and prediction saved")
        return True
        
    except Exception as e:
        print(f"✗ Training pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def trading_pipeline(ticker: str, interval: str, usdt_amount: float, testnet: bool) -> bool:
    """
    Step 3: Execute trade based on prediction
    Returns: True if successful, False otherwise
    """
    print(f"\n{'='*60}")
    print(f"[3/3] TRADING PIPELINE")
    print(f"{'='*60}")
    
    try:
        execute_trade_signal(
            ticker=ticker,
            interval=interval,
            usdt_amount=usdt_amount,
            testnet=testnet
        )
        
        print(f"✅ Trading pipeline completed")
        return True
        
    except Exception as e:
        print(f"✗ Trading pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


# ============================================================
# MAIN ORCHESTRATOR
# ============================================================

def run_full_pipeline():
    """
    Complete trading pipeline:
    1. Fetch & update data
    2. Train model & generate prediction
    3. Execute trade
    """
    
    print(f"\n{'#'*60}")
    print(f"  AUTOMATED TRADING PIPELINE")
    print(f"  Ticker: {TICKER} | Interval: {INTERVAL}")
    print(f"  Mode: {'TESTNET' if USE_TESTNET else 'MAINNET ⚠️'}")
    print(f"  Started: {datetime.now()}")
    print(f"{'#'*60}")
    
    start_time = time.time()
    
    # Step 1: Data
    success_data = data_pipeline(TICKER, INTERVAL)
    if not success_data:
        print("\n❌ PIPELINE ABORTED: Data fetch failed")
        return False
    
    # Step 2: Training
    success_train = training_pipeline(TICKER, INTERVAL)
    if not success_train:
        print("\n❌ PIPELINE ABORTED: Model training failed")
        return False
    
    # Step 3: Trading
    success_trade = trading_pipeline(TICKER, INTERVAL, TRADE_AMOUNT, USE_TESTNET)
    if not success_trade:
        print("\n⚠️  WARNING: Trading execution failed (but data & model are updated)")
        return False
    
    # Success summary
    elapsed = time.time() - start_time
    
    print(f"\n{'#'*60}")
    print(f"  ✅ PIPELINE COMPLETED SUCCESSFULLY")
    print(f"  Duration: {elapsed:.2f} seconds")
    print(f"  Completed: {datetime.now()}")
    print(f"{'#'*60}\n")
    
    return True


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    
    # Safety check for mainnet
    if not USE_TESTNET:
        print("\n" + "!"*60)
        print("  ⚠️  WARNING: MAINNET MODE ENABLED ⚠️")
        print("  This will trade with REAL MONEY!")
        print("!"*60)
        
        #response = input("\nType 'YES' to continue with mainnet: ")
        #if response != "YES":
            #print("Aborted.")
            #sys.exit(0)
    
    try:
        success = run_full_pipeline()
        sys.exit(0 if success else 1)
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        sys.exit(1)
        
    except Exception as e:
        print(f"\n\n❌ CRITICAL ERROR: {type(e).__name__}")
        print(f"{e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)