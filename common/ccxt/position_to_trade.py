from common.ccxt.bybit_client import BybitTrader

def to_trade_fixed_normal():
    trader = BybitTrader(testnet=False)
    balance = trader.get_balance('USDT')
    pos_to_trade = balance * 0.2
    print("\n--- Position to TRADE ---")
    print(f"{round(pos_to_trade, 3)} USDT")
    return round(pos_to_trade, 3)

def to_trade_fixed_small():
    trader = BybitTrader(testnet=False)
    balance = trader.get_balance('USDT')
    pos_to_trade = balance * 0.2
    print("\n--- Position to TRADE ---")
    print(f"{round(pos_to_trade, 3)} USDT")
    return round(pos_to_trade, 3)