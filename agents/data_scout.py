"""
Agent 1: Market Data Scout (@data-scout)
Responsible for market data ingestion, OHLCV normalization, volume anomaly detection, and universe screening.
"""

from typing import Dict, Any, Optional
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
from agents.base_agent import BaseAgent
from core.schema import MarketSnapshot
from core.config import config

class MarketDataScout(BaseAgent):
    def __init__(self):
        super().__init__(
            name="MarketDataScout",
            role="Market Data & Universe Specialist",
            criteria="Ingests multi-timeframe OHLCV, calculates RVol, ATR, 52W High distance, and filters liquid candidates."
        )

    def execute(self, ticker: str, period: str = "6mo", interval: str = "1d") -> tuple[pd.DataFrame, MarketSnapshot]:
        """
        Fetches historical data, computes baseline metrics, and returns the DataFrame alongside a MarketSnapshot.
        """
        # Fetch OHLCV data from yfinance
        stock = yf.Ticker(ticker)
        df = stock.history(period=period, interval=interval)
        
        if df.empty or len(df) < 30:
            raise ValueError(f"Insufficient historical data available for ticker: {ticker}")

        # Ensure consistent column naming
        df = df.copy()
        
        # Calculate 20-period Moving Average of Volume
        df['Vol_MA20'] = df['Volume'].rolling(window=config.BB_PERIOD).mean()
        df['RVol'] = df['Volume'] / df['Vol_MA20'].replace(0, np.nan)
        df['RVol'] = df['RVol'].fillna(1.0)

        # Calculate True Range and 14-period ATR
        high_low = df['High'] - df['Low']
        high_close_prev = (df['High'] - df['Close'].shift(1)).abs()
        low_close_prev = (df['Low'] - df['Close'].shift(1)).abs()
        df['TR'] = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
        df['ATR14'] = df['TR'].rolling(window=14).mean().bfill()

        # 52-week (approx 252 bars) high
        lookback_bars = min(len(df), 252)
        high_52w = df['High'].iloc[-lookback_bars:].max()

        latest = df.iloc[-1]
        current_price = float(latest['Close'])
        dist_52w = float((high_52w - current_price) / high_52w)

        is_index = ticker.startswith("^") or ticker in ["SPY", "QQQ", "IWM"]

        snapshot = MarketSnapshot(
            ticker=ticker,
            timestamp=datetime.now(),
            current_price=round(current_price, 2),
            open=round(float(latest['Open']), 2),
            high=round(float(latest['High']), 2),
            low=round(float(latest['Low']), 2),
            close=round(float(latest['Close']), 2),
            volume=float(latest['Volume']),
            avg_volume_20d=float(latest['Vol_MA20']),
            rvol=round(float(latest['RVol']), 2),
            atr_14=round(float(latest['ATR14']), 2),
            distance_from_52w_high=round(dist_52w, 4),
            is_index=is_index
        )

        return df, snapshot
