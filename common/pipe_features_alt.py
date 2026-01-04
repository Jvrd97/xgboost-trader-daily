import os 
from dotenv import load_dotenv
from common.helper_fin import HelperFin, CustomIndicator
import numpy as np
import pandas as pd 
import logging


logging.basicConfig(level=logging.INFO)

load_dotenv()

helper = HelperFin()
ind = CustomIndicator()

def make_features_clean_BTC_MERGED(df):
    if df is None:
        raise ValueError("There is no DataFrame")
    
    logging.info("DataFrame ist eingeladet")
    df["open_shift"] = df["open"].shift(1)
    df["low_shift"] = df["low"].shift(1)
    df["high_shift"] = df["high"].shift(1)
   
    helper.add_lags(df, "close", 14)
    #df["close_wld_log"] = np.log(df["close"])
    #df["close_btc_log"] = np.log(df["close_BTC"])
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)) 
    # try this for direction strat 
    df['direction'] = (df['log_returns'] > 0).astype(int)

    # Предсказывай "сильные" движения
    #df['strong_up'] = (df['log_returns'].shift(1) > 0.02).astype(int)    # +2%
    #df['strong_down'] = (df['log_returns'].shift(1) < -0.02).astype(int) # -2%
    #df['direction_strong'] = df['strong_up'] - df['strong_down'] # -1, 0, +1

    # empower the direction 
    helper.add_lags(df, "log_returns", 14)
    #helper.add_lags(df, "returns", 14)

    df["ma14"] = df["close"].shift(1).rolling(window=14).mean()
    df["ma24"] = df["close"].shift(1).rolling(window=24).mean()
    df["std14"] = df["close"].shift(1).rolling(window=24).std()
    df["btc_shift"] = np.log(df["close_BTC"].shift(91)) # specific shit actually, with log

    #df["corr_lag90"] = (
    #    df["close"].shift(1)
    #    .rolling(window=90)
    #    .corr(df["btc_shift"])
    #)
    df["volume_shift_wld"] = df["volume"].shift(1)

    df["diff_wld_shift"] = df["close"].shift(1).diff()
    df['garman_klass_vol_shift'] = ((np.log(df['high'].shift(1))-np.log(df['low'].shift(1)))**2)/2-(2*np.log(2)-1)*((np.log(df['close'].shift(1))-np.log(df['open'].shift(1)))**2)

    df = df.pipe(ind.add_rsi,df, close_col='close', length=20) \
       .pipe(ind.add_macd, close_col='close', fast=12, slow=26, signal=9) \
    .pipe(ind.add_momentum_features, close_col='close')

    df = df.dropna()
    return df 

def make_features_clean(df):
    if df is None:
        raise ValueError("There is no DataFrame")
    
    logging.info("DataFrame ist eingeladet")
    df["open_shift"] = df["open"].shift(1)
    df["low_shift"] = df["low"].shift(1)
    df["high_shift"] = df["high"].shift(1)
   
    helper.add_lags(df, "close", 14)
    df["close_wld_log"] = np.log(df["close"])
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)) 
    # try this for direction strat 
    #df['direction'] = (df['log_returns'] > 0).astype(int)

    # Предсказывай "сильные" движения
    #df['strong_up'] = (df['log_returns'].shift(1) > 0.02).astype(int)    # +2%
    #df['strong_down'] = (df['log_returns'].shift(1) < -0.02).astype(int) # -2%
    #df['direction_strong'] = df['strong_up'] - df['strong_down'] # -1, 0, +1

    # empower the direction 
    helper.add_lags(df, "log_returns", 14)
    helper.add_lags(df, "returns", 14)

    df["ma14"] = df["close"].shift(1).rolling(window=14).mean()
    df["ma24"] = df["close"].shift(1).rolling(window=24).mean()
    df["std14"] = df["close"].shift(1).rolling(window=24).std()

    df["volume_shift_wld"] = df["volume"].shift(1)

    df["diff_wld_shift"] = df["close"].shift(1).diff()
    df['garman_klass_vol_shift'] = ((np.log(df['high'].shift(1))-np.log(df['low'].shift(1)))**2)/2-(2*np.log(2)-1)*((np.log(df['close'].shift(1))-np.log(df['open'].shift(1)))**2)

    df = df.pipe(ind.add_rsi,df, close_col='close', length=20) \
       .pipe(ind.add_macd, close_col='close', fast=12, slow=26, signal=9) \
    .pipe(ind.add_momentum_features, close_col='close')

    df = df.dropna()
    return df 





def make_features_btc_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Advanced feature engineering for BTC returns prediction.
    
    Key principles:
    1. All features use shift(1) to avoid data leakage
    2. Heavy focus on returns and volatility
    3. Momentum and trend indicators
    4. Market microstructure features
    
    Target: log_returns (next day's return)
    """
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    logging.info(f"Starting feature engineering on {len(df)} rows")
    
    df = df.copy()
    
    # ============================================================
    # 1. PRICE FEATURES (all shifted)
    # ============================================================
    logging.info("Creating price features...")
    
    df["open_shift"] = df["open"].shift(1)
    df["high_shift"] = df["high"].shift(1)
    df["low_shift"] = df["low"].shift(1)
    df["close_shift"] = df["close"].shift(1)
    df["volume_shift"] = df["volume"].shift(1)
    
    # ============================================================
    # 2. RETURNS - PRIMARY FEATURES
    # ============================================================
    logging.info("Creating return features...")
    
    # Basic returns
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Shifted returns (what we know at prediction time)
    df['returns_lag1'] = df['returns'].shift(1)
    df['log_returns_lag1'] = df['log_returns'].shift(1)
    
    # Return lags (2-14 days back)
    for lag in [2, 3, 5, 7, 10, 14]:
        df[f'log_returns_lag{lag}'] = df['log_returns'].shift(lag)
        df[f'returns_lag{lag}'] = df['returns'].shift(lag)
    
    # ============================================================
    # 3. RETURN STATISTICS (rolling windows)
    # ============================================================
    logging.info("Creating return statistics...")
    
    # Rolling mean returns (momentum)
    for window in [3, 7, 14, 21]:
        df[f'returns_ma{window}'] = df['returns'].shift(1).rolling(window=window).mean()
        df[f'log_returns_ma{window}'] = df['log_returns'].shift(1).rolling(window=window).mean()
    
    # Rolling return std (volatility)
    for window in [7, 14, 21]:
        df[f'returns_std{window}'] = df['returns'].shift(1).rolling(window=window).std()
        df[f'log_returns_std{window}'] = df['log_returns'].shift(1).rolling(window=window).std()
    
    # Return skewness (distribution shape)
    for window in [14, 21]:
        df[f'returns_skew{window}'] = df['returns'].shift(1).rolling(window=window).skew()
        df[f'returns_kurt{window}'] = df['returns'].shift(1).rolling(window=window).kurt()
    
    # ============================================================
    # 4. VOLATILITY FEATURES
    # ============================================================
    logging.info("Creating volatility features...")
    
    # Garman-Klass volatility (uses OHLC)
    df['gk_volatility'] = np.sqrt(
        ((np.log(df['high'].shift(1)) - np.log(df['low'].shift(1)))**2) / 2 -
        (2*np.log(2) - 1) * ((np.log(df['close'].shift(1)) - np.log(df['open'].shift(1)))**2)
    )
    
    # Parkinson volatility (high-low range)
    df['parkinson_vol'] = np.sqrt(
        (1 / (4 * np.log(2))) * 
        ((np.log(df['high'].shift(1)) - np.log(df['low'].shift(1)))**2)
    )
    
    # Rolling volatility
    for window in [7, 14, 21]:
        df[f'gk_volatility_ma{window}'] = df['gk_volatility'].rolling(window=window).mean()
    
    # Realized volatility (sum of squared returns)
    for window in [7, 14]:
        df[f'realized_vol{window}'] = np.sqrt(
            (df['log_returns'].shift(1)**2).rolling(window=window).sum()
        )
    
    # ============================================================
    # 5. PRICE MOMENTUM & TRENDS
    # ============================================================
    logging.info("Creating momentum features...")
    
    # Moving averages
    for window in [7, 14, 21, 50]:
        df[f'ma{window}'] = df['close'].shift(1).rolling(window=window).mean()
    
    # Price relative to MA (momentum indicator)
    for window in [7, 14, 21]:
        df[f'close_to_ma{window}'] = (df['close'].shift(1) - df[f'ma{window}']) / df[f'ma{window}']
    
    # Moving average crossovers
    df['ma7_to_ma21'] = df['ma7'] / df['ma21']
    df['ma14_to_ma50'] = df['ma14'] / df['ma50']
    
    # Price momentum (rate of change)
    for period in [3, 7, 14, 21]:
        df[f'momentum{period}'] = (df['close'].shift(1) - df['close'].shift(period+1)) / df['close'].shift(period+1)
    
    # ============================================================
    # 6. PRICE RANGE FEATURES
    # ============================================================
    logging.info("Creating range features...")
    
    # Daily range
    df['daily_range'] = (df['high'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)
    df['daily_range_ma7'] = df['daily_range'].rolling(window=7).mean()
    
    # High/Low relative to close
    df['high_to_close'] = (df['high'].shift(1) - df['close'].shift(1)) / df['close'].shift(1)
    df['low_to_close'] = (df['close'].shift(1) - df['low'].shift(1)) / df['close'].shift(1)
    
    # Overnight gap
    df['gap'] = (df['open'].shift(1) - df['close'].shift(2)) / df['close'].shift(2)
    
    # ============================================================
    # 7. VOLUME FEATURES
    # ============================================================
    logging.info("Creating volume features...")
    
    # Volume change
    df['volume_change'] = df['volume'].shift(1).pct_change()
    df['volume_ma7'] = df['volume'].shift(1).rolling(window=7).mean()
    df['volume_ma14'] = df['volume'].shift(1).rolling(window=14).mean()
    
    # Volume relative to average
    df['volume_to_ma7'] = df['volume'].shift(1) / df['volume_ma7']
    df['volume_to_ma14'] = df['volume'].shift(1) / df['volume_ma14']
    
    # Price-volume correlation
    for window in [7, 14]:
        df[f'price_volume_corr{window}'] = (
            df['returns'].shift(1).rolling(window=window).corr(df['volume_change'].shift(1))
        )
    
    # ============================================================
    # 8. TECHNICAL INDICATORS
    # ============================================================
    logging.info("Creating technical indicators...")
    
    # RSI
    df = ind.add_rsi(df, close_col='close', length=14)
    df = ind.add_rsi(df, close_col='close', length=7)
    df['rsi_14'] = df['rsi_14'].shift(1)
    df['rsi_7'] = df['rsi_7'].shift(1)
    
    # MACD
    df = ind.add_macd(df, close_col='close', fast=12, slow=26, signal=9)
    df['macd'] = df['macd'].shift(1)
    df['macd_signal'] = df['macd_signal'].shift(1)
    df['macd_histogram'] = df['macd_histogram'].shift(1)
    
    # Bollinger Bands
    for window in [14, 21]:
        ma = df['close'].shift(1).rolling(window=window).mean()
        std = df['close'].shift(1).rolling(window=window).std()
        df[f'bb_upper{window}'] = ma + 2*std
        df[f'bb_lower{window}'] = ma - 2*std
        df[f'bb_position{window}'] = (df['close'].shift(1) - df[f'bb_lower{window}']) / (df[f'bb_upper{window}'] - df[f'bb_lower{window}'])
    
    # ============================================================
    # 9. ADVANCED RETURN FEATURES
    # ============================================================
    logging.info("Creating advanced return features...")
    
    # Return autocorrelation
    for lag in [1, 2, 3, 5]:
        df[f'returns_autocorr_lag{lag}'] = (
            df['returns'].shift(1).rolling(window=14).apply(
                lambda x: x.autocorr(lag=lag) if len(x) > lag else np.nan
            )
        )
    
    # Cumulative returns
    for window in [7, 14, 21]:
        df[f'cumulative_returns{window}'] = (
            df['log_returns'].shift(1).rolling(window=window).sum()
        )
    
    # Return reversals
    df['return_reversal_3d'] = -df['log_returns'].shift(1).rolling(window=3).sum()
    
    # Momentum strength (consecutive positive/negative returns)
    df['positive_momentum'] = (df['returns'].shift(1) > 0).astype(int).rolling(window=5).sum()
    df['negative_momentum'] = (df['returns'].shift(1) < 0).astype(int).rolling(window=5).sum()
    
    # ============================================================
    # 10. MARKET REGIME FEATURES
    # ============================================================
    logging.info("Creating regime features...")
    
    # Volatility regime (high/low vol)
    vol_ma = df['returns_std14'].rolling(window=50).mean()
    df['high_vol_regime'] = (df['returns_std14'] > vol_ma).astype(int)
    
    # Trend regime
    df['uptrend'] = (df['ma7'] > df['ma21']).astype(int)
    df['strong_uptrend'] = ((df['ma7'] > df['ma21']) & (df['ma14'] > df['ma50'])).astype(int)

    df['garman_klass_vol_shift'] = ((np.log(df['high'].shift(1))-np.log(df['low'].shift(1)))**2)/2-(2*np.log(2)-1)*((np.log(df['close'].shift(1))-np.log(df['open'].shift(1)))**2)

    df = df.pipe(ind.add_rsi,df, close_col='close', length=20) \
       .pipe(ind.add_macd, close_col='close', fast=12, slow=26, signal=9) \
    .pipe(ind.add_momentum_features, close_col='close')
    
    # ============================================================
    # 11. TARGET VARIABLE
    # ============================================================
    logging.info("Creating target variable...")
    
    # Target: tomorrow's log return (what we want to predict)
    df['direction'] = df['log_returns']  # This is NOT shifted - it's the target
    
    # Alternative targets (uncomment if needed)
    # df['direction_binary'] = (df['log_returns'] > 0).astype(int)
    # df['direction_strong'] = np.where(df['log_returns'] > 0.02, 1, 
    #                                    np.where(df['log_returns'] < -0.02, -1, 0))
    
    # ============================================================
    # CLEANUP
    # ============================================================
    logging.info("Cleaning up...")
    
    # Drop rows with NaN (from rolling windows and shifts)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    logging.info(f"Feature engineering complete!")
    logging.info(f"Final shape: {df.shape}")
    logging.info(f"Dropped {dropped_rows} rows with NaN")
    logging.info(f"Features created: {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])}")
    
    return df


if __name__ == "__main__":
    # Test the feature engineering
    from common.path_man import BASE_CONCAT_PATH
    
    file_path = BASE_CONCAT_PATH / "D" / "BTC" / "BTC_D_concat.csv"
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    df_features = make_features_btc_returns(df)
    
    print("\n" + "="*60)
    print("FEATURE ENGINEERING TEST")
    print("="*60)
    print(f"Input shape: {df.shape}")
    print(f"Output shape: {df_features.shape}")
    print(f"\nFeatures created: {len(df_features.columns)}")
    print(f"\nLast 3 rows:")
    print(df_features.tail(3))
    print(f"\nTarget (direction/log_returns) stats:")
    print(df_features['direction'].describe())




import numpy as np
import pandas as pd
import logging

from common import ml_helper as helper
ind = CustomIndicator()
logging.basicConfig(level=logging.INFO)

def make_features_btc_enhanced(df: pd.DataFrame, risk_free_rate: float = 0.02) -> pd.DataFrame:
    """
    Enhanced BTC feature engineering inspired by:
    "Stock Price Prediction Using Technical Indicators: A CNN+LSTM+Multi-Head Attention Approach"
    
    Key innovations from the paper:
    1. Focus on advanced technical indicators (ADX, Bollinger Bands, Mass Index, Vortex, etc.)
    2. Volume-based features (PVO - Percentage Volume Oscillator)
    3. Distribution shape features (Kurtosis, Skewness)
    4. Properly shifted features to avoid data leakage
    5. Target: binary classification based on risk-free rate threshold
    
    Args:
        df: OHLCV dataframe with timestamp index
        risk_free_rate: Annual risk-free rate (default 0.02 = 2%)
    
    Returns:
        DataFrame with enhanced features
    """
    
    if df is None or df.empty:
        raise ValueError("DataFrame is empty or None")
    
    logging.info(f"Enhanced feature engineering on {len(df)} rows")
    df = df.copy()
    
    # ============================================================
    # 1. BASIC RETURNS (Paper's primary focus)
    # ============================================================
    logging.info("Creating return features...")
    
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    
    # Daily risk-free rate
    daily_rf_rate = (1 + risk_free_rate) ** (1/365) - 1
    
    # Shifted returns (what we know at prediction time)
    df['returns_lag1'] = df['returns'].shift(1)
    df['log_returns_lag1'] = df['log_returns'].shift(1)
    
    # Return lags
    for lag in [2, 3, 5, 7, 10, 14, 21, 30]:
        df[f'log_returns_lag{lag}'] = df['log_returns'].shift(lag)
    
    # ============================================================
    # 2. VOLUME FEATURES (Critical from paper: PVO)
    # ============================================================
    logging.info("Creating volume features...")
    
    df['volume_shift'] = df['volume'].shift(1)
    df['volume_change'] = df['volume'].shift(1).pct_change()
    
    # Percentage Volume Oscillator (PVO) - From paper Table 2
    df['volume_ema12'] = df['volume'].shift(1).ewm(span=12, adjust=False).mean()
    df['volume_ema26'] = df['volume'].shift(1).ewm(span=26, adjust=False).mean()
    df['PVO_12_26'] = ((df['volume_ema12'] - df['volume_ema26']) / df['volume_ema26']) * 100
    df['PVO_signal_9'] = df['PVO_12_26'].ewm(span=9, adjust=False).mean()
    df['PVO_histogram'] = df['PVO_12_26'] - df['PVO_signal_9']
    
    # Volume moving averages
    for window in [7, 14, 21]:
        df[f'volume_ma{window}'] = df['volume'].shift(1).rolling(window=window).mean()
        df[f'volume_std{window}'] = df['volume'].shift(1).rolling(window=window).std()
    
    # Volume ratio to average
    df['volume_to_ma14'] = df['volume'].shift(1) / df['volume_ma14']
    
    # ============================================================
    # 3. BOLLINGER BANDS (BBB_5_2.0 from paper)
    # ============================================================
    logging.info("Creating Bollinger Band features...")
    
    for window in [5, 14, 20]:
        ma = df['close'].shift(1).rolling(window=window).mean()
        std = df['close'].shift(1).rolling(window=window).std()
        
        df[f'bb_upper_{window}'] = ma + 2*std
        df[f'bb_lower_{window}'] = ma - 2*std
        df[f'bb_middle_{window}'] = ma
        
        # Bollinger Band Bandwidth (BBB) - From paper
        df[f'BBB_{window}'] = (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}']) / df[f'bb_middle_{window}']
        
        # Bollinger Band %B
        df[f'bb_percent_{window}'] = (df['close'].shift(1) - df[f'bb_lower_{window}']) / (df[f'bb_upper_{window}'] - df[f'bb_lower_{window}'])
    
    # ============================================================
    # 4. ADX - Average Directional Index (ADX_14 from paper)
    # ============================================================
    logging.info("Creating ADX features...")
    
    # Calculate True Range
    high_low = df['high'].shift(1) - df['low'].shift(1)
    high_close = np.abs(df['high'].shift(1) - df['close'].shift(2))
    low_close = np.abs(df['low'].shift(1) - df['close'].shift(2))
    df['TR'] = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    df['up_move'] = df['high'].shift(1) - df['high'].shift(2)
    df['down_move'] = df['low'].shift(2) - df['low'].shift(1)
    
    df['plus_dm'] = np.where((df['up_move'] > df['down_move']) & (df['up_move'] > 0), df['up_move'], 0)
    df['minus_dm'] = np.where((df['down_move'] > df['up_move']) & (df['down_move'] > 0), df['down_move'], 0)
    
    # Smoothed indicators (14-period)
    period = 14
    df['ATR_14'] = df['TR'].rolling(window=period).mean()
    df['plus_di_14'] = 100 * (df['plus_dm'].rolling(window=period).mean() / df['ATR_14'])
    df['minus_di_14'] = 100 * (df['minus_dm'].rolling(window=period).mean() / df['ATR_14'])
    
    # ADX calculation
    df['DX'] = 100 * np.abs(df['plus_di_14'] - df['minus_di_14']) / (df['plus_di_14'] + df['minus_di_14'])
    df['ADX_14'] = df['DX'].rolling(window=period).mean()
    
    # ============================================================
    # 5. NATR - Normalized ATR (NATR_14 from paper)
    # ============================================================
    logging.info("Creating volatility features...")
    
    df['NATR_14'] = (df['ATR_14'] / df['close'].shift(1)) * 100
    
    # Garman-Klass volatility
    df['gk_volatility'] = np.sqrt(
        ((np.log(df['high'].shift(1)) - np.log(df['low'].shift(1)))**2) / 2 -
        (2*np.log(2) - 1) * ((np.log(df['close'].shift(1)) - np.log(df['open'].shift(1)))**2)
    )
    
    # ============================================================
    # 6. MASS INDEX (MASSI_9_25 from paper)
    # ============================================================
    logging.info("Creating Mass Index...")
    
    # Single EMA of high-low range
    hl_range = df['high'].shift(1) - df['low'].shift(1)
    ema9 = hl_range.ewm(span=9, adjust=False).mean()
    
    # Double EMA
    ema9_ema9 = ema9.ewm(span=9, adjust=False).mean()
    
    # Mass Index
    df['MASSI_9_25'] = (ema9 / ema9_ema9).rolling(window=25).sum()
    
    # ============================================================
    # 7. VORTEX INDICATOR (VTXM_14 from paper)
    # ============================================================
    logging.info("Creating Vortex Indicator...")
    
    # Vortex Movement
    df['vortex_plus'] = np.abs(df['high'].shift(1) - df['low'].shift(2))
    df['vortex_minus'] = np.abs(df['low'].shift(1) - df['high'].shift(2))
    
    # Vortex Indicators
    period = 14
    df['VI_plus_14'] = df['vortex_plus'].rolling(window=period).sum() / df['TR'].rolling(window=period).sum()
    df['VI_minus_14'] = df['vortex_minus'].rolling(window=period).sum() / df['TR'].rolling(window=period).sum()
    
    # Vortex Trend (VTXM)
    df['VTXM_14'] = df['VI_plus_14'] - df['VI_minus_14']
    
    # ============================================================
    # 8. KURTOSIS & SKEWNESS (KURT_30 from paper)
    # ============================================================
    logging.info("Creating distribution shape features...")
    
    for window in [14, 21, 30]:
        df[f'KURT_{window}'] = df['log_returns'].shift(1).rolling(window=window).kurt()
        df[f'SKEW_{window}'] = df['log_returns'].shift(1).rolling(window=window).skew()
    
    # ============================================================
    # 9. RSI & MACD (Standard technical indicators)
    # ============================================================
    logging.info("Creating standard technical indicators...")
    
    # RSI
    df = ind.add_rsi(df, close_col='close', length=14)
    df['rsi_14'] = df['rsi_14'].shift(1)
    
    # MACD
    df = ind.add_macd(df, close_col='close', fast=12, slow=26, signal=9)
    df['macd'] = df['macd'].shift(1)
    df['macd_signal'] = df['macd_signal'].shift(1)
    df['macd_histogram'] = df['macd_histogram'].shift(1)
    
    # ============================================================
    # 10. MOMENTUM FEATURES
    # ============================================================
    logging.info("Creating momentum features...")
    
    # Rate of Change (ROC)
    for period in [3, 7, 14, 21]:
        df[f'ROC_{period}'] = ((df['close'].shift(1) - df['close'].shift(period+1)) / df['close'].shift(period+1)) * 100
    
    # Moving averages
    for window in [7, 14, 21, 50, 100]:
        df[f'ma{window}'] = df['close'].shift(1).rolling(window=window).mean()
    
    # MA crossovers
    df['ma7_to_ma21'] = df['ma7'] / df['ma21']
    df['ma14_to_ma50'] = df['ma14'] / df['ma50']
    
    # Price deviation from MA
    for window in [7, 14, 21]:
        df[f'price_to_ma{window}'] = (df['close'].shift(1) - df[f'ma{window}']) / df[f'ma{window}']
    
    # ============================================================
    # 11. RETURN STATISTICS
    # ============================================================
    logging.info("Creating return statistics...")
    
    # Rolling return stats
    for window in [7, 14, 21]:
        df[f'returns_ma{window}'] = df['returns'].shift(1).rolling(window=window).mean()
        df[f'returns_std{window}'] = df['returns'].shift(1).rolling(window=window).std()
        df[f'log_returns_ma{window}'] = df['log_returns'].shift(1).rolling(window=window).mean()
        df[f'log_returns_std{window}'] = df['log_returns'].shift(1).rolling(window=window).std()
    
    # Cumulative returns
    for window in [7, 14, 21, 30]:
        df[f'cumulative_returns{window}'] = df['log_returns'].shift(1).rolling(window=window).sum()
    
    # ============================================================
    # 12. TARGET VARIABLE (Paper's approach)
    # ============================================================
    logging.info("Creating target variable...")
    
    # Binary target: 1 if return > risk-free rate, else 0
    df['direction_binary'] = (df['log_returns'] > daily_rf_rate).astype(int)
    
    # Continuous target: actual log returns
    df['direction'] = df['log_returns']
    
    # Alternative: strong moves (optional)
    df['strong_up'] = (df['log_returns'] > 0.02).astype(int)  # +2%
    df['strong_down'] = (df['log_returns'] < -0.02).astype(int)  # -2%
    df['direction_strong'] = df['strong_up'] - df['strong_down']  # -1, 0, +1
    
    # ============================================================
    # CLEANUP
    # ============================================================
    logging.info("Cleaning up...")
    
    # Drop intermediate calculation columns
    cols_to_drop = ['TR', 'up_move', 'down_move', 'plus_dm', 'minus_dm', 'DX',
                    'vortex_plus', 'vortex_minus', 'volume_ema12', 'volume_ema26']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)
    
    logging.info(f"✅ Feature engineering complete!")
    logging.info(f"   Final shape: {df.shape}")
    logging.info(f"   Dropped {dropped_rows} rows with NaN")
    logging.info(f"   Features created: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    # Test
    from common.path_man import BASE_CONCAT_PATH
    
    file_path = BASE_CONCAT_PATH / "D" / "BTC" / "BTC_D_concat.csv"
    df = pd.read_csv(file_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.set_index('timestamp')
    
    df_features = make_features_btc_enhanced(df)
    
    print("\n" + "="*60)
    print("ENHANCED FEATURES TEST")
    print("="*60)
    print(f"Features from paper included:")
    print(f"  ✓ ADX_14: {' ADX_14' in df_features.columns}")
    print(f"  ✓ BBB_5: {'BBB_5' in df_features.columns}")
    print(f"  ✓ KURT_30: {'KURT_30' in df_features.columns}")
    print(f"  ✓ MASSI_9_25: {'MASSI_9_25' in df_features.columns}")
    print(f"  ✓ NATR_14: {'NATR_14' in df_features.columns}")
    print(f"  ✓ PVO_12_26: {'PVO_12_26' in df_features.columns}")
    print(f"  ✓ VTXM_14: {'VTXM_14' in df_features.columns}")
    print(f"\nTotal features: {len(df_features.columns)}")