# feat_filter.py

"""
Feature configurations for different targets.
Usage:
    from config_features import CONFIG_VM, CONFIG_VM_LOG, CONFIG_RETURNS_LOG
    X = df.drop(columns=CONFIG_VM["drop"])
    y = df[CONFIG_VM["target"]]
"""

# -------------------------
# VM as target
# -------------------------
CONFIG_VM1 = {
    "Bondatum",
    "VM",
    "VM_log",
    "returns_log",
    # Peaks (leaky - calculated from target)
    "VM_peak_strong",
    "VM_zscore_peak",
    "VM_log_peak_strong",
    "VM_log_zscore_peak",
    "returns_log_peak_strong",
    "returns_log_zscore_peak",
        "rolling_mean_r_7", "rolling_std_r_7", "rolling_max_r_7", "rolling_min_r_7",
        "rolling_mean_r_14", "rolling_std_r_14", "rolling_max_r_14", "rolling_min_r_14",
        "rolling_mean_r_28", "rolling_std_r_28", "rolling_max_r_28", "rolling_min_r_28",
        'returns_log', 'returns_log_lag_1', 'returns_log_lag_2',
       'returns_log_lag_3', 'returns_log_lag_4', 'returns_log_lag_5',
       'returns_log_lag_6', 'returns_log_lag_7', 'returns_log_lag_8',
       'returns_log_lag_9', 'returns_log_lag_10', 'returns_log_lag_11',
       'returns_log_lag_12', 'returns_log_lag_13', 'returns_log_lag_14',
}  

CONFIG_VM_INVERTED = {
    "VM",
    "is_holiday",
}


# -------------------------
# VM_log as target
# -------------------------
CONFIG_VM_LOG1 = {
    "Bondatum",
    "VM",
    "VM_log",
    "returns_log",
    # Peaks (leaky - calculated from target)
    "VM_peak_strong",
    "VM_zscore_peak",
    "VM_log_peak_strong",
    "VM_log_zscore_peak",
    "returns_log_peak_strong",
    "returns_log_zscore_peak",
        "rolling_mean_r_7", "rolling_std_r_7", "rolling_max_r_7", "rolling_min_r_7",
        "rolling_mean_r_14", "rolling_std_r_14", "rolling_max_r_14", "rolling_min_r_14",
        "rolling_mean_r_28", "rolling_std_r_28", "rolling_max_r_28", "rolling_min_r_28",
        'returns_log', 'returns_log_lag_1', 'returns_log_lag_2',
       'returns_log_lag_3', 'returns_log_lag_4', 'returns_log_lag_5',
       'returns_log_lag_6', 'returns_log_lag_7', 'returns_log_lag_8',
       'returns_log_lag_9', 'returns_log_lag_10', 'returns_log_lag_11',
       'returns_log_lag_12', 'returns_log_lag_13', 'returns_log_lag_14',
}  

CONFIG_VM_LOG2 = {
    "Bondatum",
    "VM",
    "VM_log",
    "returns_log",
    # Peaks (leaky - calculated from target)
    "VM_peak_strong",
    "VM_zscore_peak",
    "VM_log_peak_strong",
    "VM_log_zscore_peak",
    "returns_log_peak_strong",
    "returns_log_zscore_peak",
        "rolling_mean_r_7", "rolling_std_r_7", "rolling_max_r_7", "rolling_min_r_7",
        "rolling_mean_r_14", "rolling_std_r_14", "rolling_max_r_14", "rolling_min_r_14",
        "rolling_mean_r_28", "rolling_std_r_28", "rolling_max_r_28", "rolling_min_r_28",
        'returns_log'
}  

CONFIG_VM_LOG_LESS_LAGS = {
    "Bondatum",
    "VM",
    "VM_log",
    "returns_log",
    # Peaks (leaky - calculated from target)
    "VM_peak_strong",
    "VM_zscore_peak",
    "VM_log_peak_strong",
    "VM_log_zscore_peak",
    "returns_log_peak_strong",
    "returns_log_zscore_peak",
        "rolling_mean_r_7", "rolling_std_r_7", "rolling_max_r_7", "rolling_min_r_7",
        "rolling_mean_r_14", "rolling_std_r_14", "rolling_max_r_14", "rolling_min_r_14",
        "rolling_mean_r_28", "rolling_std_r_28", "rolling_max_r_28", "rolling_min_r_28",
        'returns_log', 'returns_log_lag_1', 'returns_log_lag_2',
       'returns_log_lag_3', 'returns_log_lag_4', 'returns_log_lag_5',
       'returns_log_lag_6', 'returns_log_lag_8',
       'returns_log_lag_9', 'returns_log_lag_10', 'returns_log_lag_11',
       'returns_log_lag_12', 'returns_log_lag_13'
       "VM_lag_1", "VM_lag_2", "VM_lag_3", "VM_lag_4", "VM_lag_5",
        "VM_lag_6", "VM_lag_7", "VM_lag_8", "VM_lag_9", "VM_lag_10",
        "VM_lag_11", "VM_lag_12", "VM_lag_13", "VM_lag_14",
    
        "VM_log_lag_1", "VM_log_lag_2", "VM_log_lag_3", "VM_log_lag_4",
        "VM_log_lag_5", "VM_log_lag_6", "VM_log_lag_8",
        "VM_log_lag_9", "VM_log_lag_10", "VM_log_lag_11", "VM_log_lag_12",
        "VM_log_lag_13"
}  

# -------------------------
# returns_log as target
# -------------------------
CONFIG_RETURNS_LOG_WLD = {
    "low_btc", "close_BTC", "volume_BTC", "open_BTC", "high_BTC", "open", "high", "low","open", "close", "volume",'close_wld_log', 'close_btc_log', "returns",'log_returns',
'log_returns'
 "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "close_lag_4",
    "close_lag_5",
    "close_lag_6",
    "close_lag_7",
    "close_lag_8",
    "close_lag_9",
    "close_lag_10",
    "close_lag_11",
    "close_lag_12",
    "close_lag_13",
    "close_lag_14",
    "high",
    "low",
    "close",
    "volume",
    "open_BTC",
    "high_BTC",
    "low_BTC",
    "close_BTC",
    "volume_BTC",
    "open_shift",
    "low_shift",
    "high_shift",
    "close_lag_1",
    "direction",
    "returns_lag_1",
    "returns_lag_2",
    "returns_lag_3",
    "returns_lag_4",
    "returns_lag_5",
    "returns_lag_6",
    "returns_lag_7",
    "returns_lag_8",
    "returns_lag_9",
    "returns_lag_10",
    "returns_lag_11",
    "returns_lag_12",
    "returns_lag_13",
    "returns_lag_14",             
}

CONFIG_RETURNS_LOG_BTC = {
    "low_btc", "close_BTC", "volume_BTC", "open_BTC", "high_BTC", "open", "high", "low","open", "close", "volume",'close_wld_log', 'close_btc_log', "returns",'log_returns',
'log_returns'
 "close_lag_1",
    "close_lag_2",
    "close_lag_3",
    "close_lag_4",
    "close_lag_5",
    "close_lag_6",
    "close_lag_7",
    "close_lag_8",
    "close_lag_9",
    "close_lag_10",
    "close_lag_11",
    "close_lag_12",
    "close_lag_13",
    "close_lag_14",
    "high",
    "low",
    "close",
    "volume",
    "open_BTC",
    "high_BTC",
    "low_BTC",
    "close_BTC",
    "volume_BTC",
    "open_shift",
    "low_shift",
    "high_shift",
    "close_lag_1",
    "direction",
    "returns_lag_1",
    "returns_lag_2",
    "returns_lag_3",
    "returns_lag_4",
    "returns_lag_5",
    "returns_lag_6",
    "returns_lag_7",
    "returns_lag_8",
    "returns_lag_9",
    "returns_lag_10",
    "returns_lag_11",
    "returns_lag_12",
    "returns_lag_13",
    "returns_lag_14",             
}



CONFIG_RETURNS_ENHANCED_BTC = {
    # Target variables
    "direction", "direction_binary", "direction_strong",
    "strong_up", "strong_down",
    
    # Raw OHLCV
    "open", "high", "low", "close", "volume",
    
    # Non-shifted returns
    "returns", "log_returns",
    
    # Intermediate BB calculations
    "bb_upper_5", "bb_lower_5", "bb_middle_5",
    "bb_upper_14", "bb_lower_14", "bb_middle_14",
    "bb_upper_20", "bb_lower_20", "bb_middle_20",
    
    # Intermediate DI calculations
    "plus_di_14", "minus_di_14",
    
    # Intermediate VI calculations
    "VI_plus_14", "VI_minus_14",
}

# -------------------------
# returns_log minimal (only returns lags + calendar)
# -------------------------
CONFIG_RETURNS_LOG1 = {
    "Bondatum",
    "VM",
    "VM_log",
    "returns_log",
    # Rolling all
    "rolling_mean_r_7", "rolling_std_r_7", "rolling_max_r_7", "rolling_min_r_7",
    "rolling_mean_r_14", "rolling_std_r_14", "rolling_max_r_14", "rolling_min_r_14",
    "rolling_mean_r_28", "rolling_std_r_28", "rolling_max_r_28", "rolling_min_r_28",
    # Peaks
    "VM_peak_strong", "VM_zscore_peak",
    "VM_log_peak_strong", "VM_log_zscore_peak",
    "returns_log_peak_strong", "returns_log_zscore_peak",
}

### HAHA TEST IS NOT OVER!!!!!
