from common.pandas_helper import PandasHelper
import pandas as pd
import numpy as np
import logging
import json

with open("common/holidays_de.json", "r") as f:
    holidays_dict = json.load(f)
   
def set_up_data(df):
    df["Bondatum"] = pd.to_datetime(df["Bondatum"])
    df = df[["Bondatum", "VM"]]
    return df

       
def add_feat_sol(df):
    
    if df is None:
        raise ValueError("There is no DataFrame")
    
    logging.info("DataFrame ist eingeladet")
    df["is_holiday"] = df["Bondatum"].dt.strftime("%Y-%m-%d").isin(holidays_dict.keys()).astype(int)
    df["is_holiday_shift3"] = df["is_holiday"].shift(3)

    # wir werden mehrere features hinzufügen
    

    df["VM_log"] = np.log1p(df["VM"])
    df["returns_log"] = df["VM_log"].diff()
    
    df = PandasHelper.add_lags(df, "VM", lags= 14)
    df = PandasHelper.add_lags(df, "VM_log", lags= 14)
    df = PandasHelper.add_lags(df, "returns_log", lags= 14)
    # peaks
    df = PandasHelper.calculate_peaks(df, "VM_log")
    df = PandasHelper.calculate_peaks(df, "returns_log")

    
    df["day_of_week"] = df['Bondatum'].dt.dayofweek
    df["diff"] = df["VM_log"].shift(1).diff() 
    df["isSunday"]  = (df["day_of_week"]==6).astype(int)
    df["month"] =df['Bondatum'].dt.month
    df['day_of_month'] = df['Bondatum'].dt.day
    df['week_of_year'] = df['Bondatum'].dt.isocalendar().week

    for window in [7, 14, 28]:
        df[f'rolling_mean_{window}'] = df['VM_log'].shift(1).rolling(window).mean()
        df[f'rolling_std_{window}'] = df['VM_log'].shift(1).rolling(window).std()
        df[f'rolling_max_{window}'] = df['VM_log'].shift(1).rolling(window).max()
        df[f'rolling_min_{window}'] = df['VM_log'].shift(1).rolling(window).min()

    for window in [7, 14, 28]:
        df[f'rolling_mean_r_{window}'] = df["returns_log"].shift(1).rolling(window).mean()
        df[f'rolling_std_r_{window}'] = df["returns_log"].shift(1).rolling(window).std()
        df[f'rolling_max_r_{window}'] = df["returns_log"].shift(1).rolling(window).max()
        df[f'rolling_min_r_{window}'] = df["returns_log"].shift(1).rolling(window).min()

    df['is_month_start'] = (df['day_of_month'] <= 7).astype(int)
    df['is_month_end'] = (df['day_of_month'] >= 24).astype(int)
    
    df = df.dropna()
    df.set_index('Bondatum', inplace=True)

    return df
