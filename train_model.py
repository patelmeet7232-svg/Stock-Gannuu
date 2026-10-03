"""
Automated ML Training & Probability Calibration Pipeline for Excessive Shootup Prediction.
Trains LightGBM / Random Forest ensemble on historical multi-day swing anomalies.
"""

import os
import joblib
import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.model_selection import TimeSeriesSplit
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from rich.console import Console
from rich.table import Table

from core.config import config

console = Console()

def extract_features_and_labels(ticker: str, period: str = "2y") -> tuple[pd.DataFrame, pd.Series]:
    """
    Downloads historical data for a ticker, extracts quantitative agent features,
    and labels whether an excessive shootup occurred within the forward 5-day swing window.
    """
    console.print(f"Ingesting historical data for [bold]{ticker}[/bold] ({period})...")
    df = yf.download(ticker, period=period, interval="1d", progress=False, auto_adjust=True)
    
    if df.empty or len(df) < 60:
        return pd.DataFrame(), pd.Series()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    data = df.copy()

    # 1. Technical & Volatility Indicators
    data['Vol_MA20'] = data['Volume'].rolling(window=20).mean()
    data['RVol'] = (data['Volume'] / data['Vol_MA20'].replace(0, np.nan)).fillna(1.0)

    high_low = data['High'] - data['Low']
    high_close_prev = (data['High'] - data['Close'].shift(1)).abs()
    low_close_prev = (data['Low'] - data['Close'].shift(1)).abs()
    data['TR'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    data['ATR14'] = data['TR'].rolling(window=14).mean().bfill()
    data['ATR_Pct'] = (data['ATR14'] / data['Close']) * 100.0

    ma20 = data['Close'].rolling(window=20).mean()
    std20 = data['Close'].rolling(window=20).std()
    bb_upper = ma20 + (2.0 * std20)
    bb_lower = ma20 - (2.0 * std20)
    kc_upper = ma20 + (1.5 * data['ATR14'])
    kc_lower = ma20 - (1.5 * data['ATR14'])

    # Squeeze Indicator
    data['Squeeze_Active'] = ((bb_upper < kc_upper) & (bb_lower > kc_lower)).astype(float)

    # Momentum features
    data['Mom_3D'] = data['Close'].pct_change(3) * 100.0
    data['Mom_5D'] = data['Close'].pct_change(5) * 100.0
    data['Vol_Velocity'] = (data['RVol'] / data['RVol'].shift(1).replace(0, np.nan)).fillna(1.0)

    # Break of structure (20-day high breakout)
    data['Rolling_High_20'] = data['High'].shift(1).rolling(window=20).max()
    data['Break_of_Structure'] = (data['Close'] > data['Rolling_High_20']).astype(float)

    # 52-Week High Distance
    data['Rolling_52W_High'] = data['High'].rolling(window=252, min_periods=30).max()
    data['Dist_52W_High'] = ((data['Rolling_52W_High'] - data['Close']) / data['Rolling_52W_High']).fillna(0.0)

    # Smart Money Concept: Fair Value Gap (Bullish)
    fvg = (data['Low'] > data['High'].shift(2)) & (abs(data['Close'].shift(1) - data['Open'].shift(1)) > 0.8 * data['ATR14'].shift(1))
    data['FVG_Detected'] = fvg.astype(float)

    # Heuristic Technical Score
    data['Technical_Score'] = (
        (data['Squeeze_Active'] * 25.0) +
        (data['Break_of_Structure'] * 25.0) +
        (data['FVG_Detected'] * 20.0) +
        (np.clip(data['RVol'], 0, 3) * 10.0)
    )

    # 2. Forward Target Labeling (Multi-Day Swing Shootup)
    horizon = config.SWING_HORIZON_DAYS
    threshold = config.INDEX_SHOOTUP_THRESHOLD if (ticker.startswith("^") or ticker in ["SPY", "QQQ", "IWM"]) else config.STOCK_SHOOTUP_THRESHOLD

    # Max future price in next 1 to horizon days
    future_max = data['High'].shift(-1).rolling(window=horizon, min_periods=1).max().shift(-(horizon-1))
    future_max_return = (future_max - data['Close']) / data['Close']

    # Max future drawdown before reaching peak
    future_min = data['Low'].shift(-1).rolling(window=horizon, min_periods=1).min().shift(-(horizon-1))
    future_max_dip = (data['Close'] - future_min) / data['Close']

    # Target is 1 if price shot up >= threshold AND dip did not violate max risk
    target = ((future_max_return >= threshold) & (future_max_dip <= config.MAX_ALLOWABLE_DRAWDOWN)).astype(int)

    feature_cols = [
        'RVol', 'Dist_52W_High', 'ATR_Pct', 'Squeeze_Active',
        'FVG_Detected', 'Break_of_Structure', 'Technical_Score',
        'Vol_Velocity', 'Mom_3D', 'Mom_5D'
    ]

    cleaned_data = data[feature_cols].copy()
    cleaned_data['Target'] = target

    # Drop NaN rows produced by rolling windows
    cleaned_data = cleaned_data.dropna()

    X = cleaned_data[feature_cols]
    y = cleaned_data['Target']

    return X, y

def train_ensemble_model():
    """
    Trains and calibrates a high-precision ML ensemble across historical market data.
    """
    console.print("\n[bold cyan]=== Initiating Quantitative ML Architect Training Pipeline ===[/bold cyan]\n")

    training_universe = [
        "NVDA", "TSLA", "AMD", "PLTR", "MARA", "COIN", "CVNA", "MSTR", "SOFI", "ARM",
        "SPY", "QQQ", "IWM"
    ]

    all_X = []
    all_y = []

    for ticker in training_universe:
        try:
            X_t, y_t = extract_features_and_labels(ticker, period="3y")
            if not X_t.empty and len(X_t) > 100:
                all_X.append(X_t)
                all_y.append(y_t)
        except Exception as e:
            console.print(f"[red]Skipping {ticker} due to fetch error: {e}[/red]")

    if not all_X:
        raise ValueError("Could not extract historical dataset.")

    X = pd.concat(all_X, ignore_index=True)
    y = pd.concat(all_y, ignore_index=True)

    console.print(f"\n[green]Dataset Assembled successfully:[/green] {len(X)} samples.")
    console.print(f"Positive Shootup Events: [bold]{y.sum()}[/bold] ({y.mean()*100:.2f}%)")

    # Time series split for evaluation
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    # Base models: Gradient Boosting + Random Forest with class rebalancing
    hgb = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.05,
        max_depth=5,
        class_weight='balanced',
        random_state=config.RANDOM_SEED
    )
    
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight='balanced',
        random_state=config.RANDOM_SEED,
        n_jobs=-1
    )

    ensemble = VotingClassifier(
        estimators=[('hgb', hgb), ('rf', rf)],
        voting='soft'
    )

    # Probability calibration (Platt scaling / Sigmoid)
    calibrated_model = CalibratedClassifierCV(estimator=ensemble, method='sigmoid', cv=3)
    calibrated_model.fit(X_train, y_train)

    # Evaluate on test set
    y_pred_proba = calibrated_model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= config.CONFIDENCE_ALERT_THRESHOLD).astype(int)

    roc_auc = roc_auc_score(y_test, y_pred_proba)
    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    pr_auc = auc(recall, precision)

    eval_table = Table(title="Model Validation Metrics on Out-of-Sample Test Set", header_style="bold yellow")
    eval_table.add_column("Metric", style="cyan")
    eval_table.add_column("Score", style="bold green")

    eval_table.add_row("ROC-AUC", f"{roc_auc:.4f}")
    eval_table.add_row("PR-AUC (Precision-Recall)", f"{pr_auc:.4f}")
    eval_table.add_row("Precision @ Threshold >= 0.65", f"{(y_test[y_pred == 1].mean() if y_pred.sum() > 0 else 0):.2%}")
    eval_table.add_row("Signals Generated", f"{y_pred.sum()} of {len(y_test)} bars")

    console.print(eval_table)

    # Save trained model to disk
    os.makedirs(os.path.dirname(config.MODEL_SAVE_PATH), exist_ok=True)
    joblib.dump(calibrated_model, config.MODEL_SAVE_PATH)
    console.print(f"[bold green]Ensemble Model saved to: {config.MODEL_SAVE_PATH}[/bold green]\n")

if __name__ == "__main__":
    train_ensemble_model()
