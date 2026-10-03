"""
Pydantic Data Schemas defining the communication contracts between all agents.
"""

from typing import List, Dict, Optional, Any
from pydantic import BaseModel, Field
from datetime import datetime

class MarketSnapshot(BaseModel):
    ticker: str
    timestamp: datetime
    current_price: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    avg_volume_20d: float
    rvol: float
    atr_14: float
    distance_from_52w_high: float  # Percentage from 52-week high
    is_index: bool = False

class PriceActionSignal(BaseModel):
    ticker: str
    squeeze_active: bool = Field(description="Bollinger Bands inside Keltner Channel (Volatility compression)")
    fvg_detected: bool = Field(description="Bullish Fair Value Gap identified")
    order_block_detected: bool = Field(description="Institutional Bullish Order Block retested")
    liquidity_sweep: bool = Field(description="Previous lows swept followed by strong rejection")
    break_of_structure: bool = Field(description="Recent swing high broken with momentum")
    candlestick_pattern: str = Field(description="Dominant bullish candlestick formation (e.g., Bullish Marubozu, Hammer)")
    technical_score: float = Field(ge=0.0, le=100.0, description="Normalized score 0-100 from Technical & SMC Analyst")
    support_invalidation_level: float = Field(description="Price level where the breakout setup is invalidated")

class DerivativesSignal(BaseModel):
    ticker: str
    short_float_pct: Optional[float] = Field(default=None, description="Percentage of float shorted")
    days_to_cover: Optional[float] = Field(default=None, description="Short interest ratio / days to cover")
    put_call_oi_ratio: Optional[float] = Field(default=None, description="Put/Call Open Interest ratio")
    squeeze_potential_score: float = Field(ge=0.0, le=100.0, description="Score 0-100 of short/gamma squeeze likelihood")
    squeeze_regime: str = Field(description="LOW, MODERATE, ELEVATED, or EXTREME")
    commentary: str

class CatalystSignal(BaseModel):
    ticker: str
    volume_velocity_score: float = Field(ge=0.0, le=100.0, description="Sudden acceleration in trading activity")
    momentum_3d_pct: float
    momentum_5d_pct: float
    catalyst_score: float = Field(ge=0.0, le=100.0)
    key_drivers: List[str]

class MLPrediction(BaseModel):
    ticker: str
    shootup_probability: float = Field(ge=0.0, le=1.0, description="Calibrated probability of >=8% surge within 5 days")
    predicted_magnitude_tier: str = Field(description="MODERATE (8-12%), HIGH (12-20%), EXPLOSIVE (>20%)")
    confidence_tier: str = Field(description="LOW, MEDIUM, HIGH")
    top_contributing_factors: List[str]
    model_version: str = "Ensemble-v1.0"

class TradeSignal(BaseModel):
    ticker: str
    timestamp: datetime
    verdict: str  # STRONG_BUY_SHOOTUP, SPECULATIVE_SQUEEZE, WATCHLIST, NEUTRAL
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    risk_reward_ratio: float
    shootup_probability: float
    synthesis_summary: str
    agent_scores: Dict[str, float]

class AgentExecutionReport(BaseModel):
    agent_name: str
    role: str
    success: bool
    execution_time_seconds: float
    message: str
    data: Optional[Dict[str, Any]] = None
