"""
Agent 6: Risk & Backtest Validator (@risk-validator)
Specializes in False Breakout Filtering, Stop-Loss / Take-Profit Sizing, and Trade Signal Validation.
"""

from typing import Dict, Any, Optional
from datetime import datetime
from agents.base_agent import BaseAgent
from core.schema import (
    MarketSnapshot,
    PriceActionSignal,
    DerivativesSignal,
    CatalystSignal,
    MLPrediction,
    TradeSignal
)
from core.config import config

class RiskBacktestValidator(BaseAgent):
    def __init__(self):
        super().__init__(
            name="RiskBacktestValidator",
            role="Risk & False Breakout Validator",
            criteria="Vets prediction quality against false breakout criteria, calculates dynamic Stop-Loss and multi-target Take-Profits."
        )

    def execute(
        self,
        snapshot: MarketSnapshot,
        pa_signal: PriceActionSignal,
        deriv_signal: DerivativesSignal,
        cat_signal: CatalystSignal,
        ml_pred: MLPrediction
    ) -> TradeSignal:
        """
        Synthesizes all agent intelligence, filters traps, and produces the finalized TradeSignal.
        """
        entry_price = snapshot.current_price
        invalidation = pa_signal.support_invalidation_level

        # Ensure stop loss is strictly below current price and within prudent risk limits
        raw_risk = entry_price - invalidation
        max_acceptable_risk = entry_price * config.MAX_ALLOWABLE_DRAWDOWN

        if raw_risk <= 0:
            # Fallback stop loss based on 1.5 ATR
            stop_loss = round(entry_price - (1.5 * snapshot.atr_14), 2)
        elif raw_risk > max_acceptable_risk:
            # Tighten stop loss to max allowable risk
            stop_loss = round(entry_price - max_acceptable_risk, 2)
        else:
            stop_loss = round(invalidation, 2)

        risk_amount = max(entry_price - stop_loss, 0.01)

        # Dynamic Targets based on Instrument type and Magnitude Tier
        if snapshot.is_index:
            t1 = round(entry_price + (1.5 * risk_amount), 2)
            t2 = round(entry_price + (2.5 * risk_amount), 2)
        else:
            # Equities: Aim for minimum 2R and 3.5R
            t1 = round(entry_price + (2.0 * risk_amount), 2)
            t2 = round(entry_price + (3.5 * risk_amount), 2)

        rr_ratio = round((t1 - entry_price) / risk_amount, 2)

        # Trap & Overextension checks
        is_overextended = (entry_price - (entry_price - snapshot.atr_14)) > (4.0 * snapshot.atr_14)
        is_low_volume_trap = (snapshot.rvol < 0.8) and pa_signal.break_of_structure

        # Final verdict determination
        prob = ml_pred.shootup_probability
        agent_scores = {
            "technical_score": pa_signal.technical_score,
            "squeeze_potential_score": deriv_signal.squeeze_potential_score,
            "volume_velocity_score": cat_signal.volume_velocity_score,
            "catalyst_score": cat_signal.catalyst_score,
            "ml_probability_pct": round(prob * 100.0, 1)
        }

        if is_low_volume_trap:
            verdict = "NEUTRAL (Low Volume Trap Detected)"
            summary = "Price broke resistance on below-average volume; high probability of a bull trap."
        elif prob >= config.CONFIDENCE_ALERT_THRESHOLD and deriv_signal.squeeze_potential_score >= 60.0:
            verdict = "SPECULATIVE_SQUEEZE"
            summary = f"High-probability short/gamma squeeze ({deriv_signal.squeeze_regime}). Heavy upside fuel with {deriv_signal.short_float_pct or 0}% short float."
        elif prob >= config.CONFIDENCE_ALERT_THRESHOLD:
            verdict = "STRONG_BUY_SHOOTUP"
            summary = f"Confluence of technical compression, SMC structure, and volume velocity. Expected {ml_pred.predicted_magnitude_tier}."
        elif pa_signal.squeeze_active:
            verdict = "WATCHLIST (Squeeze Coiling)"
            summary = "Volatility is heavily compressed. Watch for high-volume breakout trigger above resistance."
        else:
            verdict = "NEUTRAL / PASS"
            summary = "Insufficient confluence of upside shootup catalysts at current price."

        return TradeSignal(
            ticker=snapshot.ticker,
            timestamp=datetime.now(),
            verdict=verdict,
            entry_price=entry_price,
            stop_loss=stop_loss,
            target_1=t1,
            target_2=t2,
            risk_reward_ratio=rr_ratio,
            shootup_probability=prob,
            synthesis_summary=summary,
            agent_scores=agent_scores
        )
