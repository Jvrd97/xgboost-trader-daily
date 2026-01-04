# models/intervals/wld_5min.py

import os
import json
import pandas as pd
from datetime import datetime
import numpy as np

from common.path_man import *
from common.model_manager import ModelManager
from common.load_concat_merge import load_concat_merge, delete_temp_files
from common.pipe_features_alt import make_features_clean
from common.feat_filter import CONFIG_RETURNS_LOG


# Config
MAIN_TICKER = "WLD"
SECONDARY_TICKER = "BTC"
INTERVAL = "5"
LAG_DAYS = 90
RETRAIN_EVERY_DAYS = 7  # Retrain weekly


def to_native(obj):
    """Convert numpy/pandas types to Python types."""
    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj]
    if isinstance(obj, list):
        return [to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}
    return obj


def wld_5min_pipeline(force_retrain: bool = False):
    """
    Main pipeline:
    - Loads data
    - Checks if model needs retraining (weekly)
    - If yes: retrain and save
    - If no: load existing model
    - Make prediction
    - Save prediction
    """
    
    PREDICTION_FOLDER = BASE_PRED_PATH_5MIN / MAIN_TICKER
    
    # ============================================================
    # STEP 1: LOAD DATA
    # ============================================================
    df, files_to_delete = load_concat_merge(
        main_ticker=MAIN_TICKER,
        secondary_ticker=SECONDARY_TICKER,
        interval=INTERVAL,
        lag_days=LAG_DAYS
    )
    
    if df.empty:
        print("✗ No data to process!")
        return
    
    # ============================================================
    # STEP 2: FEATURES
    # ============================================================
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    
    df_with_features = make_features_clean(df)
    
    target = 'log_returns'
    EXCLUDE = CONFIG_RETURNS_LOG
    FEATURES = [col for col in df_with_features.columns if col not in EXCLUDE]
    
    X = df_with_features[FEATURES]
    y = df_with_features[target]
    
    print(f"  Features: {len(FEATURES)} | Samples: {len(X)}")
    
    # ============================================================
    # STEP 3: MODEL MANAGER (Train only if needed!)
    # ============================================================
    manager = ModelManager(
        ticker=MAIN_TICKER,
        interval=INTERVAL,
        model_type="xgb",
        retrain_frequency_days=RETRAIN_EVERY_DAYS
    )
    
    # This only trains if model is expired or doesn't exist
    manager.train(X, y, force=force_retrain)
    
    # ============================================================
    # STEP 4: PREDICT (Fast - uses saved model)
    # ============================================================
    current_features = X.iloc[[-1]]
    log_return_pred = manager.predict(current_features)
    
    # Prediction timestamp
    last_ts = X.index[-1]
    prediction_ts = last_ts + pd.Timedelta(minutes=int(INTERVAL))
    
    print(f"\n{'=' * 50}")
    print("FORECAST")
    print(f"{'=' * 50}")
    print(f"  Features from: {last_ts}")
    print(f"  Predicting:    {prediction_ts}")
    print(f"  log_returns:   {log_return_pred:.6f}")
    
  
    # ============================================================
    # STEP 5: SAVE PREDICTION
    # ============================================================

    # Get the last timestamp from features
    last_timestamp = X.index[-1]

    # Calculate prediction timestamp based on interval
    if INTERVAL == "D":
        prediction_timestamp = last_timestamp + pd.Timedelta(days=1)
        date_format = '%Y-%m-%d'
    elif INTERVAL == "W":
        prediction_timestamp = last_timestamp + pd.Timedelta(weeks=1)
        date_format = '%Y-%m-%d'
    else:
        # Numeric intervals (5, 15, 60, 240) are in minutes
        prediction_timestamp = last_timestamp + pd.Timedelta(minutes=int(INTERVAL))
        date_format = '%Y-%m-%d_%H-%M'

    prediction_data = {
        "ticker": MAIN_TICKER,
        "interval": INTERVAL,
        "feature_timestamp": last_timestamp.strftime('%Y-%m-%d %H:%M:%S'),  # When features are from
        "prediction_timestamp": prediction_timestamp.strftime('%Y-%m-%d %H:%M:%S'),  # What we're predicting
        "prediction_date": prediction_timestamp.strftime(date_format),  # For filename compatibility
        "log_returns": round(float(log_return_pred), 6),
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "features": FEATURES  # Fixed typo: was "featues"
    }

    prediction_data = to_native(prediction_data)

    os.makedirs(PREDICTION_FOLDER, exist_ok=True)

    # Filename with correct timestamp
    json_filename = f"{PREDICTION_FOLDER}/pred_{prediction_timestamp.strftime(date_format)}.json"

    with open(json_filename, 'w') as f:
        json.dump(prediction_data, f, indent=2)

    print(f"\n✓ Saved: {json_filename}")
    
    # ============================================================
    # STEP 6: CLEANUP
    # ============================================================
    delete_temp_files(files_to_delete)
    
    return prediction_data


if __name__ == "__main__":
    wld_5min_pipeline()