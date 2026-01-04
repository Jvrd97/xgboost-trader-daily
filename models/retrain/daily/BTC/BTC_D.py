import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import requests

from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from catboost import CatBoostRegressor
from xgboost import XGBRegressor

from common.ml_helper import CustomForecaster
from common.pipe_features_alt import make_features_btc_enhanced
from common.load_concat_merge import load_concat_merge, delete_temp_files  # ← раскомментируй!
from common.feat_filter import CONFIG_RETURNS_ENHANCED_BTC

from common.path_man import *

from dotenv import load_dotenv
load_dotenv()

ticker = "BTC"
interval = "D"


def to_native(obj):
    """
    Recursively convert numpy/pandas types to pure Python types.
    """
    import numpy as np
    import pandas as pd

    if isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.integer, np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, (np.bool_)):
        return bool(obj)
    if isinstance(obj, (pd.Timestamp)):
        return obj.isoformat()
    if isinstance(obj, np.ndarray):
        return [to_native(x) for x in obj]
    if isinstance(obj, list):
        return [to_native(x) for x in obj]
    if isinstance(obj, dict):
        return {k: to_native(v) for k, v in obj.items()}

    return obj




def btc_daily_train(ticker: str = "BTC", interval: str = "D"):
    concat_path = BASE_CONCAT_PATH / interval / ticker
    file_name= f"{ticker}_{interval}_concat.csv"
    file_to_read = concat_path / file_name
    PREDICTION_FOLDER = BASE_PRED_PATH / interval / ticker

    # ============================================================
    # STEP 1: LOAD, CONCAT, MERGE
    # ============================================================
    df = pd.read_csv(file_to_read)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    
    if df.empty:
        print("✗ No data to process!")
        return

    # ============================================================
    # STEP 2: FEATURES
    # ============================================================
    print(f"\n{'=' * 50}")
    print("FEATURES")
    print(f"{'=' * 50}")
    
    df = df.sort_values('timestamp').reset_index(drop=True)
    df.set_index('timestamp', inplace=True)
    
    df_with_features = make_features_btc_enhanced(df)
    
    target = 'direction'
    EXCLUDE = CONFIG_RETURNS_ENHANCED_BTC  # используй конфиг вместо хардкода
    print(df_with_features)
    FEATURES = [col for col in df_with_features.columns if col not in EXCLUDE]
    
    X = df_with_features[FEATURES]
    y = df_with_features[target]
    
    print(f"  Features: {len(FEATURES)}")
    print(f"  Target: {target}")
    print(f"  Samples: {len(X)}")

    # ============================================================
    # STEP 3: TRAIN
    # ============================================================
    best_params_cat = {
        "iterations": 552,
        "learning_rate": 0.05,
        "depth": 4,
        "l2_leaf_reg": 0.088,
        "min_data_in_leaf": 26,
        "random_strength": 0.275,
        "bagging_temperature": 0.967,
        "border_count": 82,
        "loss_function": "RMSE",
        "random_state": 42,
        "verbose": False
    }
    model_from_json_params = {
    "objective": "reg:squarederror",
    "colsample_bytree": 0.8,
    "learning_rate": 0.05,
    "max_depth": 8,
    "min_child_weight": 5,
    "n_estimators": 1000,
    "random_state": 42,
    "reg_lambda": 0.5,
    "subsample": 0.8,
    "tree_method": "hist",
}

    base_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('model', XGBRegressor(**model_from_json_params))
    ])

    FORECAST_HORIZON = 1
    forecaster = CustomForecaster(
        base_pipeline=base_pipeline,
        param_grid=None,
        horizon=FORECAST_HORIZON,
        cv_splits=3
    )

    forecaster.train(X, y)

    # ============================================================
    # STEP 4: PREDICT
    # ============================================================
    current_features = X.iloc[[-1]]
    future_predictions = forecaster.predict_future(current_features)

    print(f"\n{'=' * 50}")
    print("FORECAST")
    print(f"{'=' * 50}")
    print(f"Using features from: {X.index[-1].date()}")
    print(f"Predicting for:      {X.index[-1].date() + pd.Timedelta(days=1)}")
    
    log_return_pred = future_predictions['prediction'].values[0]
    print(f"\nPrediction (log_returns): {log_return_pred:.6f}")

    # ============================================================
    # STEP 5: SAVE PREDICTION
    # ============================================================
    prediction_date = X.index[-1] + pd.Timedelta(days=1)

    prediction_data = {
        "ticker": ticker,
        "interval": interval,
        "prediction_date": prediction_date.strftime('%Y-%m-%d'),
        "log_returns": round(log_return_pred, 6),
        "created_at": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        "featues" : FEATURES
    }

    prediction_data = to_native(prediction_data)

    os.makedirs(PREDICTION_FOLDER, exist_ok=True)
    json_filename = f"{PREDICTION_FOLDER}/pred_{prediction_date.strftime('%Y-%m-%d')}.json"

    with open(json_filename, 'w') as f:
        json.dump(prediction_data, f, indent=2)

    print(f"\n✓ Saved: {json_filename}")
    print(json.dumps(prediction_data, indent=2))

if __name__ == "__main__":
    btc_daily_train()