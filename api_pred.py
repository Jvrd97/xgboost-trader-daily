"""
5-Minute Prediction Monitoring API
==================================
FastAPI server for real-time 5-min prediction evaluation.

Run:
    uvicorn prediction_api_5min:app --reload --port 8000

Dashboard:
    http://localhost:8000
"""

import json
import glob
import os
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional

import pandas as pd
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Local imports
from common.path_man import *


# =============================================================================
# Configuration
# =============================================================================

TICKERS = ["WLD", "BTC", "SOL", "ETH"]  # Add your tickers here
DEFAULT_INTERVAL = "D"


# =============================================================================
# FastAPI App
# =============================================================================

app = FastAPI(
    title="Daily Prediction Monitor",
    description="Real-time ML prediction evaluation for crypto forecasting",
    version="2.0.0"
)


# =============================================================================
# Pydantic Models
# =============================================================================

class PredictionResult(BaseModel):
    timestamp: str
    predicted_log_return: float
    predicted_price: Optional[float] = None
    actual_log_return: Optional[float] = None
    actual_price: Optional[float] = None
    error_log_return: Optional[float] = None
    error_price: Optional[float] = None
    error_pct: Optional[float] = None
    direction_correct: Optional[bool] = None


class MetricsSummary(BaseModel):
    ticker: str
    interval: str
    total_predictions: int
    evaluated: int
    pending: int
    mae_log_return: Optional[float] = None
    mae_price: Optional[float] = None
    rmse_log_return: Optional[float] = None
    mape: Optional[float] = None
    mpe: Optional[float] = None
    direction_accuracy: Optional[float] = None
    last_prediction: Optional[str] = None
    last_updated: str


class TickerStatus(BaseModel):
    ticker: str
    interval: str
    prediction_count: int
    latest_prediction: Optional[str] = None
    latest_actual: Optional[str] = None
    status: str


class DashboardData(BaseModel):
    tickers: List[TickerStatus]
    total_tickers: int
    total_predictions: int
    last_refresh: str


# =============================================================================
# Helper: Convert log_returns to price
# =============================================================================

def log_return_to_price(log_return: float, prev_price: float) -> float:
    """
    Convert log return to actual price.
    
    Formula: price_t = price_{t-1} * exp(log_return)
    """
    return prev_price * np.exp(log_return)


def price_to_log_return(price: float, prev_price: float) -> float:
    """
    Convert prices to log return.
    
    Formula: log_return = log(price_t / price_{t-1})
    """
    if prev_price <= 0:
        return 0.0
    return np.log(price / prev_price)


# =============================================================================
# Core Evaluator
# =============================================================================

class CryptoEvaluator:
    """Evaluator for 5-minute crypto predictions."""
    
    def __init__(self, ticker: str, interval: str = "D"):
        self.ticker = ticker
        self.interval = interval
        
        # Paths
        self.pred_folder = BASE_PRED_PATH / interval / ticker
        self.concat_file = BASE_CONCAT_PATH / interval / ticker / f"{ticker}_{interval}_concat.csv"
    
    def get_predictions(self) -> List[Dict]:
        """Load all prediction JSON files."""
        if not self.pred_folder.exists():
            return []
        
        pattern = self.pred_folder / "pred_*.json"
        pred_files = glob.glob(str(pattern))
        
        predictions = []
        for pred_file in sorted(pred_files):
            try:
                with open(pred_file, 'r') as f:
                    pred = json.load(f)
                    predictions.append({
                        "prediction_timestamp": pred.get("prediction_timestamp"),
                        "feature_timestamp": pred.get("feature_timestamp"),
                        "log_returns": pred.get("log_returns"),
                        "ticker": pred.get("ticker"),
                        "interval": pred.get("interval"),
                        "created_at": pred.get("created_at"),
                        "model_trained_at": pred.get("model_trained_at"),
                        "file": os.path.basename(pred_file)
                    })
            except Exception as e:
                print(f"Error loading {pred_file}: {e}")
        
        return predictions
    
    def get_actuals(self) -> pd.DataFrame:
        """Load actual values from concat CSV."""
        if not self.concat_file.exists():
            print(f"⚠ Concat file not found: {self.concat_file}")
            return pd.DataFrame()
        
        try:
            df = pd.read_csv(self.concat_file, parse_dates=['timestamp'])
            df = df.sort_values('timestamp').set_index('timestamp')
            
            # Calculate actual log returns if not present
            if 'log_returns' not in df.columns and 'close' in df.columns:
                df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            
            return df
        except Exception as e:
            print(f"Error loading actuals: {e}")
            return pd.DataFrame()
    
    def evaluate(self) -> Dict:
        """Run evaluation and return results."""
        predictions = self.get_predictions()
        df_actual = self.get_actuals()
        
        results = []
        evaluated_count = 0
        pending_count = 0
        
        for pred in predictions:
            pred_ts = pd.to_datetime(pred["prediction_timestamp"])
            pred_log_return = pred["log_returns"]
            
            result = {
                "timestamp": pred["prediction_timestamp"],
                "predicted_log_return": pred_log_return,
                "predicted_price": None,
                "actual_log_return": None,
                "actual_price": None,
                "error_log_return": None,
                "error_price": None,
                "error_pct": None,
                "direction_correct": None
            }
            
            if not df_actual.empty:
                # Find matching actual data
                # Use nearest timestamp within tolerance (e.g., 1 minute)
                time_diff = abs(df_actual.index - pred_ts)
                min_diff_idx = time_diff.argmin()
                min_diff = time_diff[min_diff_idx]
                
                # Only match if within 5 minutes
                if min_diff <= pd.Timedelta(minutes=5):
                    actual_row = df_actual.iloc[min_diff_idx]
                    actual_log_return = actual_row.get('log_returns', None)
                    actual_price = actual_row.get('close', None)
                    
                    # Get previous price for conversion
                    if min_diff_idx > 0:
                        prev_price = df_actual.iloc[min_diff_idx - 1].get('close', None)
                    else:
                        prev_price = None
                    
                    if actual_log_return is not None and not pd.isna(actual_log_return):
                        result["actual_log_return"] = float(actual_log_return)
                        result["actual_price"] = float(actual_price) if actual_price else None
                        
                        # Calculate predicted price
                        if prev_price and not pd.isna(prev_price):
                            result["predicted_price"] = log_return_to_price(pred_log_return, prev_price)
                        
                        # Errors
                        error_log = pred_log_return - actual_log_return
                        result["error_log_return"] = float(error_log)
                        
                        if result["predicted_price"] and actual_price:
                            result["error_price"] = float(result["predicted_price"] - actual_price)
                            result["error_pct"] = float((result["error_price"] / actual_price) * 100)
                        
                        # Direction accuracy (did we predict the right direction?)
                        result["direction_correct"] = bool(
                            (pred_log_return > 0) == (actual_log_return > 0)
                        )
                        
                        evaluated_count += 1
                    else:
                        pending_count += 1
                else:
                    pending_count += 1
            else:
                pending_count += 1
            
            results.append(result)
        
        # Calculate metrics
        metrics = self._calculate_metrics(results)
        metrics["ticker"] = self.ticker
        metrics["interval"] = self.interval
        metrics["total_predictions"] = len(predictions)
        metrics["evaluated"] = evaluated_count
        metrics["pending"] = pending_count
        metrics["last_prediction"] = predictions[-1]["prediction_timestamp"] if predictions else None
        metrics["last_updated"] = datetime.now().isoformat()
        
        return {
            "metrics": metrics,
            "results": results
        }
    
    def _calculate_metrics(self, results: List[Dict]) -> Dict:
        """Calculate summary metrics."""
        evaluated = [r for r in results if r["actual_log_return"] is not None]
        
        if not evaluated:
            return {
                "mae_log_return": None,
                "mae_price": None,
                "rmse_log_return": None,
                "mape": None,
                "mpe": None,
                "direction_accuracy": None
            }
        
        # Log return errors
        log_errors = [r["error_log_return"] for r in evaluated if r["error_log_return"] is not None]
        abs_log_errors = [abs(e) for e in log_errors]
        
        # Price errors
        price_errors = [r["error_price"] for r in evaluated if r["error_price"] is not None]
        abs_price_errors = [abs(e) for e in price_errors]
        
        # Percentage errors
        pct_errors = [r["error_pct"] for r in evaluated if r["error_pct"] is not None]
        
        # Direction accuracy
        direction_results = [r for r in evaluated if r["direction_correct"] is not None]
        direction_acc = None
        if direction_results:
            correct = sum(1 for r in direction_results if r["direction_correct"])
            direction_acc = (correct / len(direction_results)) * 100
        
        # MPE (Mean Percentage Error) - signed
        mpe = np.mean(pct_errors) if pct_errors else None
        
        # MAPE (Mean Absolute Percentage Error)
        mape = np.mean([abs(e) for e in pct_errors]) if pct_errors else None
        
        return {
            "mae_log_return": float(np.mean(abs_log_errors)) if abs_log_errors else None,
            "mae_price": float(np.mean(abs_price_errors)) if abs_price_errors else None,
            "rmse_log_return": float(np.sqrt(np.mean([e**2 for e in log_errors]))) if log_errors else None,
            "mape": float(mape) if mape else None,
            "mpe": float(mpe) if mpe else None,
            "direction_accuracy": float(direction_acc) if direction_acc else None
        }


# =============================================================================
# API Endpoints
# =============================================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Main dashboard page."""
    return get_dashboard_html()


@app.get("/api/tickers", response_model=List[str])
async def list_tickers():
    """Get list of all tickers with predictions."""
    try:
        # Find tickers that have prediction folders
        available = []
        for ticker in TICKERS:
            pred_folder = BASE_PRED_PATH / DEFAULT_INTERVAL / ticker
            if pred_folder.exists() and list(pred_folder.glob("pred_*.json")):
                available.append(ticker)
        return available if available else TICKERS
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/status", response_model=DashboardData)
async def get_status():
    """Get overall status of all tickers."""
    try:
        tickers = await list_tickers()
        statuses = []
        total_predictions = 0
        
        for ticker in tickers:
            evaluator = CryptoEvaluator(ticker, DEFAULT_INTERVAL)
            predictions = evaluator.get_predictions()
            df_actual = evaluator.get_actuals()
            
            status = TickerStatus(
                ticker=ticker,
                interval=DEFAULT_INTERVAL,
                prediction_count=len(predictions),
                latest_prediction=predictions[-1]["prediction_timestamp"] if predictions else None,
                latest_actual=df_actual.index.max().isoformat() if not df_actual.empty else None,
                status="ok" if predictions else "no_data"
            )
            statuses.append(status)
            total_predictions += len(predictions)
        
        return DashboardData(
            tickers=statuses,
            total_tickers=len(tickers),
            total_predictions=total_predictions,
            last_refresh=datetime.now().isoformat()
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ticker/{ticker}")
async def get_ticker_detail(ticker: str, interval: str = "D"):
    """Get detailed evaluation for a specific ticker."""
    try:
        evaluator = CryptoEvaluator(ticker, interval)
        return evaluator.evaluate()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ticker/{ticker}/metrics", response_model=MetricsSummary)
async def get_ticker_metrics(ticker: str, interval: str = "5"):
    """Get metrics summary for a specific ticker."""
    try:
        evaluator = CryptoEvaluator(ticker, interval)
        data = evaluator.evaluate()
        return MetricsSummary(**data["metrics"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ticker/{ticker}/predictions", response_model=List[PredictionResult])
async def get_ticker_predictions(ticker: str, interval: str = "5", limit: int = 50):
    """Get predictions for a specific ticker."""
    try:
        evaluator = CryptoEvaluator(ticker, interval)
        data = evaluator.evaluate()
        results = data["results"][-limit:]
        return [PredictionResult(**r) for r in results]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ticker/{ticker}/chart-data")
async def get_chart_data(ticker: str, interval: str = "5", points: int = 100):
    """Get data formatted for charts."""
    try:
        evaluator = CryptoEvaluator(ticker, interval)
        data = evaluator.evaluate()
        results = data["results"][-points:]
        
        return {
            "labels": [r["timestamp"] for r in results],
            "predicted_price": [r["predicted_price"] for r in results],
            "actual_price": [r["actual_price"] for r in results],
            "predicted_log_return": [r["predicted_log_return"] for r in results],
            "actual_log_return": [r["actual_log_return"] for r in results],
            "error_pct": [r["error_pct"] for r in results],
            "direction_correct": [r["direction_correct"] for r in results]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ticker/{ticker}/latest")
async def get_latest_prediction(ticker: str, interval: str = "D"):
    """Get the most recent prediction."""
    try:
        evaluator = CryptoEvaluator(ticker, interval)
        predictions = evaluator.get_predictions()
        
        if not predictions:
            raise HTTPException(status_code=404, detail="No predictions found")
        
        latest = predictions[-1]
        
        # Get actual data for context
        df_actual = evaluator.get_actuals()
        current_price = None
        if not df_actual.empty:
            current_price = float(df_actual['close'].iloc[-1])
        
        return {
            "ticker": ticker,
            "interval": interval,
            "prediction_timestamp": latest["prediction_timestamp"],
            "predicted_log_return": latest["log_returns"],
            "predicted_direction": "UP 📈" if latest["log_returns"] > 0 else "DOWN 📉",
            "current_price": current_price,
            "predicted_price": log_return_to_price(latest["log_returns"], current_price) if current_price else None,
            "model_trained_at": latest.get("model_trained_at"),
            "created_at": latest["created_at"]
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Dashboard HTML
# =============================================================================

def get_dashboard_html() -> str:
    """Generate dashboard HTML."""
    return """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔮 5-Min Crypto Prediction Monitor</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <style>
        * { box-sizing: border-box; margin: 0; padding: 0; }
        body { 
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #0f0f1a 0%, #1a1a2e 100%);
            color: #eee;
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1600px; margin: 0 auto; }
        
        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 30px;
            padding-bottom: 20px;
            border-bottom: 1px solid #333;
        }
        h1 { font-size: 2em; }
        h1 span { color: #f7931a; }
        .refresh-btn {
            background: linear-gradient(135deg, #f7931a, #ff6b00);
            color: white;
            border: none;
            padding: 12px 24px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 1em;
            font-weight: bold;
        }
        .refresh-btn:hover { opacity: 0.9; }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin-bottom: 30px;
        }
        .stat-card {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            text-align: center;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .stat-value { font-size: 2.2em; font-weight: bold; }
        .stat-value.green { color: #00d395; }
        .stat-value.orange { color: #f7931a; }
        .stat-value.red { color: #ff6b6b; }
        .stat-label { color: #888; margin-top: 5px; font-size: 0.9em; }
        
        .ticker-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 20px;
            margin-bottom: 30px;
        }
        .ticker-card {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 20px;
            cursor: pointer;
            transition: all 0.3s;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .ticker-card:hover {
            transform: translateY(-3px);
            border-color: #f7931a;
            box-shadow: 0 10px 40px rgba(247, 147, 26, 0.15);
        }
        .ticker-card.selected {
            border-color: #f7931a;
            background: rgba(247, 147, 26, 0.1);
        }
        .ticker-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 15px;
        }
        .ticker-name { font-size: 1.6em; font-weight: bold; }
        .ticker-name .symbol { color: #f7931a; }
        .ticker-status {
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.75em;
            font-weight: bold;
        }
        .status-ok { background: rgba(0, 211, 149, 0.2); color: #00d395; }
        .status-no_data { background: rgba(255, 107, 107, 0.2); color: #ff6b6b; }
        
        .metric-row {
            display: flex;
            justify-content: space-between;
            padding: 10px 0;
            border-bottom: 1px solid rgba(255,255,255,0.05);
        }
        .metric-label { color: #888; }
        .metric-value { font-weight: bold; }
        .metric-good { color: #00d395; }
        .metric-bad { color: #ff6b6b; }
        .metric-neutral { color: #f7931a; }
        
        .detail-section {
            background: rgba(255,255,255,0.03);
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 20px;
            border: 1px solid rgba(255,255,255,0.08);
        }
        .detail-section h2 {
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 1px solid #333;
            color: #f7931a;
        }
        
        .chart-container {
            position: relative;
            height: 350px;
            margin-bottom: 20px;
        }
        
        .latest-prediction {
            background: linear-gradient(135deg, rgba(247, 147, 26, 0.1), rgba(255, 107, 0, 0.1));
            border: 1px solid #f7931a;
            border-radius: 12px;
            padding: 25px;
            margin-bottom: 30px;
            text-align: center;
        }
        .latest-prediction h3 { color: #f7931a; margin-bottom: 15px; }
        .prediction-direction {
            font-size: 3em;
            font-weight: bold;
            margin: 15px 0;
        }
        .prediction-direction.up { color: #00d395; }
        .prediction-direction.down { color: #ff6b6b; }
        
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid rgba(255,255,255,0.08);
        }
        th { color: #888; font-weight: normal; font-size: 0.85em; }
        
        .direction-icon { font-size: 1.2em; }
        
        #lastUpdate { color: #666; font-size: 0.85em; }
        
        .auto-refresh {
            display: flex;
            align-items: center;
            gap: 8px;
            color: #888;
            font-size: 0.9em;
        }
        .auto-refresh input { accent-color: #f7931a; }
    </style>
</head>
<body>
    <div class="container">
        <header>
            <div>
                <h1>🔮 <span>5-Min</span> Crypto Prediction Monitor</h1>
                <span id="lastUpdate"></span>
            </div>
            <div style="display: flex; gap: 20px; align-items: center;">
                <div class="auto-refresh">
                    <input type="checkbox" id="autoRefresh" checked>
                    <label for="autoRefresh">Auto-refresh (30s)</label>
                </div>
                <button class="refresh-btn" onclick="refreshAll()">🔄 Refresh</button>
            </div>
        </header>
        
        <div class="stats-grid" id="statsGrid">
            <div class="stat-card">
                <div class="stat-value orange" id="totalTickers">-</div>
                <div class="stat-label">Tickers</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="totalPredictions">-</div>
                <div class="stat-label">Total Predictions</div>
            </div>
            <div class="stat-card">
                <div class="stat-value green" id="avgAccuracy">-</div>
                <div class="stat-label">Avg Direction Accuracy</div>
            </div>
            <div class="stat-card">
                <div class="stat-value" id="avgMPE">-</div>
                <div class="stat-label">Avg MPE</div>
            </div>
        </div>
        
        <div class="ticker-grid" id="tickerGrid">
            <div style="color: #888; padding: 40px; text-align: center;">Loading tickers...</div>
        </div>
        
        <div id="latestSection" style="display: none;">
            <div class="latest-prediction">
                <h3>📡 Latest Prediction - <span id="latestTicker"></span></h3>
                <div>Predicting for: <strong id="latestTimestamp"></strong></div>
                <div class="prediction-direction" id="latestDirection">-</div>
                <div>
                    Log Return: <strong id="latestLogReturn"></strong> | 
                    Predicted Price: <strong id="latestPrice"></strong>
                </div>
                <div style="margin-top: 10px; color: #888; font-size: 0.85em;">
                    Model trained: <span id="latestModelDate"></span>
                </div>
            </div>
        </div>
        
        <div id="detailSection" style="display: none;">
            <div class="detail-section">
                <h2>📈 Price: Predicted vs Actual</h2>
                <div class="chart-container">
                    <canvas id="priceChart"></canvas>
                </div>
            </div>
            
            <div class="detail-section">
                <h2>📊 Log Returns: Predicted vs Actual</h2>
                <div class="chart-container">
                    <canvas id="returnChart"></canvas>
                </div>
            </div>
            
            <div class="detail-section">
                <h2>📉 Error % Over Time</h2>
                <div class="chart-container">
                    <canvas id="errorChart"></canvas>
                </div>
            </div>
            
            <div class="detail-section">
                <h2>📋 Recent Predictions</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Timestamp</th>
                            <th>Pred Log Return</th>
                            <th>Actual Log Return</th>
                            <th>Pred Price</th>
                            <th>Actual Price</th>
                            <th>Error %</th>
                            <th>Direction</th>
                        </tr>
                    </thead>
                    <tbody id="predictionsTable">
                    </tbody>
                </table>
            </div>
        </div>
    </div>
    
    <script>
        let priceChart = null;
        let returnChart = null;
        let errorChart = null;
        let selectedTicker = null;
        let autoRefreshInterval = null;
        
        document.addEventListener('DOMContentLoaded', () => {
            refreshAll();
            setupAutoRefresh();
        });
        
        function setupAutoRefresh() {
            const checkbox = document.getElementById('autoRefresh');
            checkbox.addEventListener('change', () => {
                if (checkbox.checked) {
                    autoRefreshInterval = setInterval(refreshAll, 30000);
                } else {
                    clearInterval(autoRefreshInterval);
                }
            });
            autoRefreshInterval = setInterval(refreshAll, 30000);
        }
        
        async function refreshAll() {
            await loadStatus();
            if (selectedTicker) {
                await loadTickerDetail(selectedTicker);
            }
            document.getElementById('lastUpdate').textContent = 
                'Last update: ' + new Date().toLocaleTimeString();
        }
        
        async function loadStatus() {
            try {
                const response = await fetch('/api/status');
                const data = await response.json();
                
                document.getElementById('totalTickers').textContent = data.total_tickers;
                document.getElementById('totalPredictions').textContent = data.total_predictions;
                
                let accuracies = [];
                let mpes = [];
                
                const grid = document.getElementById('tickerGrid');
                grid.innerHTML = '';
                
                for (const ticker of data.tickers) {
                    const metrics = await fetch(`/api/ticker/${ticker.ticker}/metrics`).then(r => r.json());
                    
                    if (metrics.direction_accuracy) accuracies.push(metrics.direction_accuracy);
                    if (metrics.mpe) mpes.push(metrics.mpe);
                    
                    const card = createTickerCard(ticker, metrics);
                    grid.appendChild(card);
                }
                
                document.getElementById('avgAccuracy').textContent = 
                    accuracies.length > 0 ? (accuracies.reduce((a,b) => a+b, 0) / accuracies.length).toFixed(1) + '%' : '-';
                document.getElementById('avgMPE').textContent = 
                    mpes.length > 0 ? (mpes.reduce((a,b) => a+b, 0) / mpes.length).toFixed(2) + '%' : '-';
                    
            } catch (error) {
                console.error('Error loading status:', error);
            }
        }
        
        function createTickerCard(ticker, metrics) {
            const card = document.createElement('div');
            card.className = 'ticker-card' + (selectedTicker === ticker.ticker ? ' selected' : '');
            card.onclick = () => selectTicker(ticker.ticker, card);
            
            const dirAcc = metrics.direction_accuracy;
            const dirClass = dirAcc > 55 ? 'metric-good' : dirAcc < 45 ? 'metric-bad' : 'metric-neutral';
            
            const mpe = metrics.mpe;
            const mpeClass = mpe ? (Math.abs(mpe) < 1 ? 'metric-good' : Math.abs(mpe) > 5 ? 'metric-bad' : 'metric-neutral') : '';
            
            card.innerHTML = `
                <div class="ticker-header">
                    <span class="ticker-name">🪙 <span class="symbol">${ticker.ticker}</span>/USDT</span>
                    <span class="ticker-status status-${ticker.status}">${ticker.status}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Predictions</span>
                    <span class="metric-value">${metrics.total_predictions}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Evaluated</span>
                    <span class="metric-value">${metrics.evaluated} / ${metrics.total_predictions}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">Direction Accuracy</span>
                    <span class="metric-value ${dirClass}">${dirAcc ? dirAcc.toFixed(1) + '%' : '-'}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">MPE</span>
                    <span class="metric-value ${mpeClass}">${mpe ? (mpe > 0 ? '+' : '') + mpe.toFixed(4) + '%' : '-'}</span>
                </div>
                <div class="metric-row">
                    <span class="metric-label">MAE (log return)</span>
                    <span class="metric-value">${metrics.mae_log_return ? metrics.mae_log_return.toFixed(6) : '-'}</span>
                </div>
            `;
            
            return card;
        }
        
        async function selectTicker(ticker, cardElement) {
            selectedTicker = ticker;
            
            document.querySelectorAll('.ticker-card').forEach(card => {
                card.classList.remove('selected');
            });
            cardElement.classList.add('selected');
            
            await loadTickerDetail(ticker);
            document.getElementById('latestSection').style.display = 'block';
            document.getElementById('detailSection').style.display = 'block';
        }
        
        async function loadTickerDetail(ticker) {
            try {
                // Load latest prediction
                const latest = await fetch(`/api/ticker/${ticker}/latest`).then(r => r.json());
                document.getElementById('latestTicker').textContent = ticker;
                document.getElementById('latestTimestamp').textContent = latest.prediction_timestamp;
                document.getElementById('latestLogReturn').textContent = latest.predicted_log_return.toFixed(6);
                document.getElementById('latestPrice').textContent = latest.predicted_price ? '$' + latest.predicted_price.toFixed(4) : '-';
                document.getElementById('latestModelDate').textContent = latest.model_trained_at || '-';
                
                const dirEl = document.getElementById('latestDirection');
                if (latest.predicted_log_return > 0) {
                    dirEl.textContent = '📈 UP';
                    dirEl.className = 'prediction-direction up';
                } else {
                    dirEl.textContent = '📉 DOWN';
                    dirEl.className = 'prediction-direction down';
                }
                
                // Load chart data
                const chartData = await fetch(`/api/ticker/${ticker}/chart-data?points=50`).then(r => r.json());
                
                updatePriceChart(chartData);
                updateReturnChart(chartData);
                updateErrorChart(chartData);
                
                // Load predictions table
                const predictions = await fetch(`/api/ticker/${ticker}/predictions?limit=30`).then(r => r.json());
                updatePredictionsTable(predictions);
                
            } catch (error) {
                console.error('Error loading ticker detail:', error);
            }
        }
        
        function updatePriceChart(data) {
            const ctx = document.getElementById('priceChart').getContext('2d');
            if (priceChart) priceChart.destroy();
            
            priceChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels.map(l => l.split('T')[1]?.substring(0,5) || l.substring(11,16)),
                    datasets: [
                        {
                            label: 'Actual Price',
                            data: data.actual_price,
                            borderColor: '#00d395',
                            backgroundColor: 'rgba(0, 211, 149, 0.1)',
                            fill: true,
                            tension: 0.3,
                            pointRadius: 2
                        },
                        {
                            label: 'Predicted Price',
                            data: data.predicted_price,
                            borderColor: '#f7931a',
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.3,
                            pointRadius: 2
                        }
                    ]
                },
                options: getChartOptions()
            });
        }
        
        function updateReturnChart(data) {
            const ctx = document.getElementById('returnChart').getContext('2d');
            if (returnChart) returnChart.destroy();
            
            returnChart = new Chart(ctx, {
                type: 'line',
                data: {
                    labels: data.labels.map(l => l.split('T')[1]?.substring(0,5) || l.substring(11,16)),
                    datasets: [
                        {
                            label: 'Actual Log Return',
                            data: data.actual_log_return,
                            borderColor: '#00d395',
                            fill: false,
                            tension: 0.3,
                            pointRadius: 2
                        },
                        {
                            label: 'Predicted Log Return',
                            data: data.predicted_log_return,
                            borderColor: '#f7931a',
                            borderDash: [5, 5],
                            fill: false,
                            tension: 0.3,
                            pointRadius: 2
                        }
                    ]
                },
                options: getChartOptions()
            });
        }
        
        function updateErrorChart(data) {
            const ctx = document.getElementById('errorChart').getContext('2d');
            if (errorChart) errorChart.destroy();
            
            const colors = data.error_pct.map(e => 
                e === null ? 'rgba(128,128,128,0.5)' : 
                e > 0 ? 'rgba(255, 107, 107, 0.7)' : 'rgba(0, 211, 149, 0.7)'
            );
            
            errorChart = new Chart(ctx, {
                type: 'bar',
                data: {
                    labels: data.labels.map(l => l.split('T')[1]?.substring(0,5) || l.substring(11,16)),
                    datasets: [{
                        label: 'Error %',
                        data: data.error_pct,
                        backgroundColor: colors,
                        borderWidth: 0
                    }]
                },
                options: getChartOptions()
            });
        }
        
        function getChartOptions() {
            return {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { labels: { color: '#eee' } }
                },
                scales: {
                    x: { ticks: { color: '#888', maxTicksLimit: 15 }, grid: { color: 'rgba(255,255,255,0.05)' } },
                    y: { ticks: { color: '#888' }, grid: { color: 'rgba(255,255,255,0.05)' } }
                }
            };
        }
        
        function updatePredictionsTable(predictions) {
            const tbody = document.getElementById('predictionsTable');
            tbody.innerHTML = '';
            
            predictions.reverse().forEach(pred => {
                const row = document.createElement('tr');
                
                const dirIcon = pred.direction_correct === true ? '✅' : 
                               pred.direction_correct === false ? '❌' : '⏳';
                
                row.innerHTML = `
                    <td>${pred.timestamp}</td>
                    <td>${pred.predicted_log_return.toFixed(6)}</td>
                    <td>${pred.actual_log_return !== null ? pred.actual_log_return.toFixed(6) : '⏳'}</td>
                    <td>${pred.predicted_price !== null ? '$' + pred.predicted_price.toFixed(4) : '-'}</td>
                    <td>${pred.actual_price !== null ? '$' + pred.actual_price.toFixed(4) : '⏳'}</td>
                    <td>${pred.error_pct !== null ? (pred.error_pct > 0 ? '+' : '') + pred.error_pct.toFixed(4) + '%' : '-'}</td>
                    <td class="direction-icon">${dirIcon}</td>
                `;
                
                tbody.appendChild(row);
            });
        }
    </script>
</body>
</html>
"""


# =============================================================================
# Run Server
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting 5-Min Prediction Monitor...")
    print(f"📂 Predictions: {BASE_PRED_PATH_5MIN}")
    print(f"📂 Concat data: {BASE_CONCAT_PATH}")
    print("📊 Dashboard: http://localhost:8000")
    print("📖 API Docs:  http://localhost:8000/docs")
    uvicorn.run(app, host="0.0.0.0", port=8000)