"""
Agent 4: Catalyst & Sentiment Radar (@catalyst-radar)
Tracks sudden volume acceleration, momentum velocity, and news headline catalysts.
"""

from typing import Dict, Any, List
import pandas as pd
import yfinance as yf
from agents.base_agent import BaseAgent
from core.schema import CatalystSignal, MarketSnapshot
from core.config import config

class CatalystSentimentRadar(BaseAgent):
    def __init__(self):
        super().__init__(
            name="CatalystSentimentRadar",
            role="Catalyst & Sentiment Velocity Specialist",
            criteria="Monitors sudden volume velocity spikes, multi-day momentum expansion, and material news triggers."
        )

    def execute(self, df: pd.DataFrame, snapshot: MarketSnapshot) -> CatalystSignal:
        """
        Extracts momentum velocity and news catalysts for the asset.
        """
        ticker = snapshot.ticker
        data = df.copy()

        # Multi-day momentum calculations
        p_current = data['Close'].iloc[-1]
        p_3d_ago = data['Close'].iloc[-4] if len(data) >= 4 else p_current
        p_5d_ago = data['Close'].iloc[-6] if len(data) >= 6 else p_current

        mom_3d_pct = round(float((p_current - p_3d_ago) / p_3d_ago) * 100.0, 2)
        mom_5d_pct = round(float((p_current - p_5d_ago) / p_5d_ago) * 100.0, 2)

        # Volume velocity (acceleration in volume over the past 2 sessions)
        rvol_today = data['RVol'].iloc[-1]
        rvol_yesterday = data['RVol'].iloc[-2] if len(data) >= 2 else 1.0
        vol_accel = (rvol_today / max(rvol_yesterday, 0.1))

        # Check news headlines from yfinance
        news_drivers = []
        try:
            stock = yf.Ticker(ticker)
            news_items = stock.news or []
            for item in news_items[:3]:
                title = item.get("title", "")
                if title:
                    news_drivers.append(title)
        except Exception:
            pass

        # Compute volume velocity score (0 to 100)
        vol_velocity_score = min(float(rvol_today * 25.0), 100.0)

        # Compute overall catalyst score (0 to 100)
        catalyst_score = 0.0
        # Positive short-term momentum
        if mom_3d_pct > 3.0: catalyst_score += 25.0
        if mom_5d_pct > 6.0: catalyst_score += 25.0
        # Volume burst
        if rvol_today >= config.RVOL_THRESHOLD: catalyst_score += 30.0
        if vol_accel >= 1.5: catalyst_score += 10.0
        # Presence of news
        if news_drivers: catalyst_score += 10.0

        catalyst_score = min(max(catalyst_score, 0.0), 100.0)

        drivers_summary = news_drivers if news_drivers else [
            f"3D Momentum: {mom_3d_pct}%",
            f"5D Momentum: {mom_5d_pct}%",
            f"Relative Volume: {rvol_today:.1f}x"
        ]

        return CatalystSignal(
            ticker=ticker,
            volume_velocity_score=round(vol_velocity_score, 1),
            momentum_3d_pct=mom_3d_pct,
            momentum_5d_pct=mom_5d_pct,
            catalyst_score=round(catalyst_score, 1),
            key_drivers=drivers_summary
        )
