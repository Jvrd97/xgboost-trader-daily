import os 
from dotenv import load_dotenv
from common.helper_fin import HelperFin, CustomIndicator
import numpy as np
import pandas as pd 
import logging

load_dotenv()

helper = HelperFin()
ind = CustomIndicator()

def make_features_clean(df):
    if df is None:
        raise ValueError("There is no DataFrame")
    
    logging.info("DataFrame ist eingeladet")
    
    helper.add_lags(df, "close", 14)
    df["close_wld_log"] = np.log(df["close"])
    df["close_btc_log"] = np.log(df["close_btc"])
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1)) 

    helper.add_lags(df, "returns", 14)

    df["ma14"] = df["close"].shift(1).rolling(window=14).mean()
    df["ma24"] = df["close"].shift(1).rolling(window=24).mean()
    df["std14"] = df["close"].shift(1).rolling(window=24).std()
    df["btc_shift"] = np.log(df["close_btc"].shift(91)) # specific shit actually, with log

    df["corr_lag90"] = (
        df["close"].shift(1)
        .rolling(window=90)
        .corr(df["btc_shift"])
    )
    df["volume_shift"] = df["volume_wld"].shift(1)

    df["diff_wld_shift"] = df["close"].shift(1).diff()
    df['garman_klass_vol_shift'] = ((np.log(df['high'].shift(1))-np.log(df['low'].shift(1)))**2)/2-(2*np.log(2)-1)*((np.log(df['close'].shift(1))-np.log(df['open'].shift(1)))**2)

    df = df.pipe(ind.add_rsi,df, close_col='close', length=20) \
       .pipe(ind.add_macd, close_col='close', fast=12, slow=26, signal=9) \
    .pipe(ind.add_momentum_features, close_col='close')

    df = df.dropna()
    return df 