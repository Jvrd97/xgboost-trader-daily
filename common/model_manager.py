# common/model_manager.py

import os
import json
import pickle
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from common.path_man import *


class ModelManager:
    """
    Manages model lifecycle: train, save, load, predict.
    Avoids retraining on every prediction.
    """
    
    def __init__(
        self,
        ticker: str,
        interval: str,
        model_type: str = "xgb",  # "xgb", "catboost", "lstm"
        retrain_frequency_days: int = 7
    ):
        self.ticker = ticker
        self.interval = interval
        self.model_type = model_type
        self.retrain_frequency_days = retrain_frequency_days
        
        # Paths
        self.model_dir = BASE_DIR / "models" / "saved" / interval / ticker
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_path = self.model_dir / f"{ticker}_{interval}_{model_type}.pkl"
        self.metadata_path = self.model_dir / f"{ticker}_{interval}_{model_type}_meta.json"
        
        # State
        self.model = None
        self.features = None
        self.metadata = None
    
    # =========================================================
    # SAVE / LOAD
    # =========================================================
    
    def save_model(self, model, features: list, extra_meta: Dict = None):
        """Save trained model + metadata."""
        
        # Save model
        with open(self.model_path, 'wb') as f:
            pickle.dump(model, f)
        
        # Save metadata
        metadata = {
            "ticker": self.ticker,
            "interval": self.interval,
            "model_type": self.model_type,
            "features": features,
            "trained_at": datetime.now().isoformat(),
            "retrain_after": (datetime.now() + timedelta(days=self.retrain_frequency_days)).isoformat(),
            "n_features": len(features),
            **(extra_meta or {})
        }
        
        with open(self.metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✓ Model saved: {self.model_path}")
        print(f"  Retrain after: {metadata['retrain_after']}")
        
        self.model = model
        self.features = features
        self.metadata = metadata
    
    def load_model(self) -> bool:
        """Load model from disk. Returns True if successful."""
        
        if not self.model_path.exists():
            print(f"✗ No saved model found: {self.model_path}")
            return False
        
        # Load model
        with open(self.model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        # Load metadata
        with open(self.metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        self.features = self.metadata.get("features", [])
        
        print(f"✓ Model loaded: {self.model_path}")
        print(f"  Trained: {self.metadata['trained_at']}")
        
        return True
    
    # =========================================================
    # RETRAIN CHECK
    # =========================================================
    
    def needs_retrain(self) -> bool:
        """Check if model needs retraining based on schedule."""
        
        # No model exists
        if not self.model_path.exists():
            print("→ No model found, training required")
            return True
        
        # Load metadata to check date
        if not self.metadata:
            self.load_model()
        
        retrain_after = datetime.fromisoformat(self.metadata['retrain_after'])
        
        if datetime.now() > retrain_after:
            print(f"→ Model expired ({retrain_after}), retraining required")
            return True
        
        print(f"→ Model valid until {retrain_after}")
        return False
    
    # =========================================================
    # PREDICT (FAST - No Training)
    # =========================================================
    
    def predict(self, X: pd.DataFrame) -> float:
        """Fast prediction using loaded model."""
        
        if self.model is None:
            if not self.load_model():
                raise RuntimeError("No model available! Train first.")
        
        # Ensure correct features in correct order
        X_aligned = X[self.features]
        
        prediction = self.model.predict(X_aligned)
        
        return prediction[0] if hasattr(prediction, '__len__') else prediction
    
    # =========================================================
    # TRAIN (SLOW - Only When Needed)
    # =========================================================
    
    def train(self, X: pd.DataFrame, y: pd.Series, force: bool = False):
        """Train model (only if needed or forced)."""
        
        if not force and not self.needs_retrain():
            print("→ Skipping training, using existing model")
            return
        
        print(f"\n{'=' * 50}")
        print(f"TRAINING {self.model_type.upper()} MODEL")
        print(f"{'=' * 50}")
        
        features = list(X.columns)
        
        # Build model based on type
        if self.model_type == "xgb":
            model = self._train_xgb(X, y)
        elif self.model_type == "catboost":
            model = self._train_catboost(X, y)
        elif self.model_type == "lstm":
            model = self._train_lstm(X, y)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Save
        self.save_model(model, features, extra_meta={
            "n_samples": len(X),
            "date_range": f"{X.index.min()} to {X.index.max()}"
        })
    
    def _train_xgb(self, X, y):
        from sklearn.preprocessing import StandardScaler
        from sklearn.pipeline import Pipeline
        from xgboost import XGBRegressor
        
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('model', XGBRegressor(
                n_estimators=500,
                max_depth=8,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                tree_method="hist"
            ))
        ])
        pipeline.fit(X, y)
        return pipeline
    
    def _train_catboost(self, X, y):
        from catboost import CatBoostRegressor
        
        model = CatBoostRegressor(
            iterations=500,
            learning_rate=0.05,
            depth=6,
            random_state=42,
            verbose=False
        )
        model.fit(X, y)
        return model
    
    def _train_lstm(self, X, y):
        # Placeholder - implement your LSTM-MHA here
        raise NotImplementedError("LSTM training not yet implemented")