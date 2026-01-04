from pybit.unified_trading import HTTP
import os
from dotenv import load_dotenv
from typing import Dict

load_dotenv()

class BybitTrader:
    """Bybit trading using official pybit SDK"""
    
    def __init__(self, testnet: bool = True):
        self.testnet = testnet
        
        api_key = os.getenv('BYBIT_API_KEY')
        api_secret = os.getenv('BYBIT_API_SECRET')
        
        if not api_key or not api_secret:
            raise ValueError("Missing BYBIT_API_KEY or BYBIT_API_SECRET in .env")
        
        self.session = HTTP(
            testnet=testnet,
            api_key=api_key,
            api_secret=api_secret
        )
        
        print(f"🔧 Using Bybit {'TESTNET' if testnet else 'MAINNET'}")
    
    def get_balance(self, coin: str = 'USDT') -> float:
        """Get available balance for a specific coin"""
        try:
            result = self.session.get_wallet_balance(accountType="UNIFIED")
            
            if result['retCode'] != 0:
                print(f"Error getting balance: {result['retMsg']}")
                return 0.0
            
            # Parse balance
            for item in result['result']['list']:
                for balance_coin in item.get('coin', []):
                    if balance_coin['coin'] == coin:
                        return float(balance_coin['walletBalance'])
            
            return 0.0
            
        except Exception as e:
            print(f"Error fetching balance: {e}")
            return 0.0
    
    def get_position(self, symbol: str) -> Dict:
        """
        Get current position for a symbol
        Returns: {'side': 'long'/'short'/None, 'size': float, 'entry_price': float}
        """
        try:
            # Convert symbol format: BTC/USDT:USDT -> BTCUSDT
            clean_symbol = symbol.split('/')[0] + 'USDT'
            
            result = self.session.get_positions(
                category="linear",
                symbol=clean_symbol
            )
            
            if result['retCode'] != 0:
                print(f"Error getting position: {result['retMsg']}")
                return {'side': None, 'size': 0.0, 'entry_price': 0.0, 'unrealized_pnl': 0.0}
            
            # Parse positions
            for pos in result['result']['list']:
                size = float(pos['size'])
                if size > 0:
                    side = pos['side'].lower()  # 'Buy' -> 'buy', 'Sell' -> 'sell'
                    return {
                        'side': 'long' if side == 'buy' else 'short',
                        'size': size,
                        'entry_price': float(pos['avgPrice']),
                        'unrealized_pnl': float(pos['unrealisedPnl'])
                    }
            
            return {'side': None, 'size': 0.0, 'entry_price': 0.0, 'unrealized_pnl': 0.0}
            
        except Exception as e:
            print(f"Error fetching position: {e}")
            return {'side': None, 'size': 0.0, 'entry_price': 0.0, 'unrealized_pnl': 0.0}
    
    def get_current_price(self, symbol: str) -> float:
        """Get current market price"""
        try:
            # Convert symbol format
            clean_symbol = symbol.split('/')[0] + 'USDT'
            
            result = self.session.get_tickers(
                category="linear",
                symbol=clean_symbol
            )
            
            if result['retCode'] != 0:
                print(f"Error getting price: {result['retMsg']}")
                return 0.0
            
            return float(result['result']['list'][0]['lastPrice'])
            
        except Exception as e:
            print(f"Error fetching price: {e}")
            return 0.0
    
    def close_position(self, symbol: str, position_side: str, position_size: float):
        """Close existing position"""
        try:
            clean_symbol = symbol.split('/')[0] + 'USDT'
            
            # To close: Sell if long, Buy if short
            order_side = 'Sell' if position_side == 'long' else 'Buy'
            
            result = self.session.place_order(
                category="linear",
                symbol=clean_symbol,
                side=order_side,
                orderType="Market",
                qty=str(position_size),
                reduceOnly=True,
                positionIdx=0  # One-way mode
            )
            
            if result['retCode'] == 0:
                print(f"✓ Closed {position_side.upper()} position: {position_size} contracts")
                return result
            else:
                print(f"✗ Error closing position: {result['retMsg']}")
                return None
            
        except Exception as e:
            print(f"✗ Error closing position: {e}")
            return None
    
    def open_position(self, symbol: str, side: str, amount: float):
        """
        Open new position
        side: 'buy' for LONG, 'sell' for SHORT
        """
        try:
            clean_symbol = symbol.split('/')[0] + 'USDT'
            
            # Capitalize for Bybit API
            order_side = 'Buy' if side == 'buy' else 'Sell'
            
            result = self.session.place_order(
                category="linear",
                symbol=clean_symbol,
                side=order_side,
                orderType="Market",
                qty=str(amount),
                positionIdx=0  # One-way mode
            )
            
            if result['retCode'] == 0:
                print(f"✓ Opened {side.upper()} position: {amount} contracts")
                return result
            else:
                print(f"✗ Error opening position: {result['retMsg']}")
                return None
            
        except Exception as e:
            print(f"✗ Error opening position: {e}")
            return None
    
    def calculate_position_size_small(self, symbol: str, usdt_amount: float) -> float:
        """Calculate how many contracts to trade based on USDT amount"""
        price = self.get_current_price(symbol)
        if price == 0:
            return 0.0
        contracts = (usdt_amount / price)
        return round(contracts, 0)  # Round to 3 decimals
    
    def calculate_position_size(self, symbol: str, usdt_amount: float) -> float:
        """Calculate how many contracts to trade based on USDT amount"""
        price = self.get_current_price(symbol)
        if price == 0:
            return 0.0
        contracts = (usdt_amount / price)
        return round(contracts, 1)  # Round to 3 decimals


if __name__ == "__main__":
    # Test connection
    trader = BybitTrader(testnet=False)
    
    print("\n--- Testing Connection ---")
    balance = trader.get_balance('USDT')
    print(f"USDT Balance: {balance}")
    
    btc_price = trader.get_current_price('BTC/USDT:USDT')
    print(f"BTC Price: ${btc_price}")
    
    position = trader.get_position('BTC/USDT:USDT')
    print(f"BTC Position: {position}")