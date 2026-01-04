import os
import json
from datetime import datetime
from pathlib import Path
from dotenv import load_dotenv

from common.ccxt.bybit_client import BybitTrader
from common.path_man import *

load_dotenv()

# Map interval to prediction folder



def load_latest_prediction(ticker: str, interval: str) -> dict:
    """Load the most recent prediction JSON file"""
    
    # Get prediction folder from path_man
    pred_folder = BASE_PRED_PATH / interval / ticker
    
    if not pred_folder:
        print(f"✗ No prediction folder configured for interval: {interval}")
        return None
    
    if not pred_folder.exists():
        print(f"✗ Prediction folder not found: {pred_folder}")
        return None
    
    # Find all prediction files for this ticker
    pred_files = sorted(pred_folder.glob(f"pred_*.json"), reverse=True)
    
    if not pred_files:
        print(f"✗ No prediction files found in {pred_folder}")
        return None
    
    # Load the most recent one
    latest_file = pred_files[0]
    
    with open(latest_file, 'r') as f:
        prediction = json.load(f)
    
    # Verify it's for the correct ticker
    if prediction.get('ticker') != ticker:
        print(f"⚠️  Warning: Found prediction for {prediction.get('ticker')}, expected {ticker}")
    
    print(f"📊 Loaded prediction: {latest_file.name}")
    return prediction


def execute_trade_signal(ticker: str, interval: str, usdt_amount: float = 100.0, testnet: bool = True):
    """
    Main trading logic:
    1. Load latest prediction
    2. Check current position
    3. Execute trade based on signal
    """
    
    print(f"\n{'='*50}")
    print(f"TRADE EXECUTION: {ticker} [{interval}]")
    print(f"{'='*50}")
    print(f"Time: {datetime.now()}")
    
    # Load prediction
    prediction = load_latest_prediction(ticker, interval)
    
    if not prediction:
        print("✗ No prediction available, skipping trade")
        return
    
    log_returns = prediction['log_returns']
    pred_date = prediction['prediction_date']
    
    print(f"\n📈 Prediction for {pred_date}")
    print(f"   Log Returns: {log_returns:.6f}")
    
    # Determine signal
    if log_returns > 0:
        signal = 'LONG'
        action = 'buy'
    else:
        signal = 'SHORT'
        action = 'sell'
    
    print(f"   Signal: {signal}")
    
    # Initialize trader
    trader = BybitTrader(testnet=testnet)
    symbol = f"{ticker}/USDT:USDT"  # Perpetual futures format
    
    # Check balance
    balance = trader.get_balance()
    print(f"\n💰 Balance: {balance:.2f} USDT")
    
    if balance < usdt_amount:
        print(f"⚠️  Insufficient balance! Required: {usdt_amount} USDT")
        return
    
    # Get current position
    position = trader.get_position(symbol)
    current_side = position['side']
    current_size = position['size']
    
    print(f"\n📍 Current Position:")
    print(f"   Side: {current_side if current_side else 'None'}")
    print(f"   Size: {current_size}")
    
    if current_side:
        print(f"   Entry: ${position['entry_price']:.2f}")
        print(f"   PnL: ${position['unrealized_pnl']:.2f}")
    
    # Calculate position size
    contracts = trader.calculate_position_size(symbol, usdt_amount)
    current_price = trader.get_current_price(symbol)
    
    print(f"\n📊 Market Info:")
    print(f"   Price: ${current_price:.2f}")
    print(f"   Contracts to trade: {contracts}")
    
    # ============================================================
    # TRADING LOGIC
    # ============================================================
    print(f"\n{'='*50}")
    print("EXECUTING TRADES")
    print(f"{'='*50}")
    
    # Case 1: No position → Open new position
    if not current_side:
        print(f"→ Opening new {signal} position")
        trader.open_position(symbol, action, contracts)
    
    # Case 2: Same direction → Hold (do nothing)
    elif current_side == signal.lower():
        print(f"✓ Already in {signal} position, holding")
    
    # Case 3: Opposite direction → Close old, open new
    else:
        print(f"→ Switching from {current_side.upper()} to {signal}")
        
        # Step 1: Close existing position
        trader.close_position(symbol, current_side, current_size)
        
        # Step 2: Open new position
        trader.open_position(symbol, action, contracts)
    
    # Final position check
    print(f"\n{'='*50}")
    print("FINAL POSITION")
    print(f"{'='*50}")
    
    final_position = trader.get_position(symbol)
    print(f"   Side: {final_position['side']}")
    print(f"   Size: {final_position['size']}")
    print(f"   Entry: ${final_position['entry_price']:.2f}")
    
    print(f"\n✅ Trade execution complete!")


if __name__ == "__main__":
    # Test the trading pipeline
    execute_trade_signal(
        ticker="WLD",
        interval="D",
        usdt_amount=2,
        testnet=True
    )