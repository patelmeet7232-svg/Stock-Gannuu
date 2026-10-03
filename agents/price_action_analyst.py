"""
Agent 2: Price Action & Smart Money Concepts (SMC) Analyst (@price-action-analyst)
Specializes in Volatility Compression Squeezes, Fair Value Gaps, Order Blocks, Liquidity Sweeps, and Candlestick Formations.
Incorporates principles from Smart Money Concepts and Classic Technical Mastery.
"""

from typing import Dict, Any, List
import pandas as pd
import numpy as np
from agents.base_agent import BaseAgent
from core.schema import PriceActionSignal, MarketSnapshot
from core.config import config

class PriceActionSMCAnalyst(BaseAgent):
    def __init__(self):
        super().__init__(
            name="PriceActionSMCAnalyst",
            role="Price Action & Smart Money Specialist",
            criteria="Detects Bollinger Band Squeezes inside Keltner Channels, Bullish FVGs, Order Blocks, Liquidity Sweeps, and Candlestick momentum."
        )

    def execute(self, df: pd.DataFrame, snapshot: MarketSnapshot) -> PriceActionSignal:
        """
        Analyzes the historical OHLCV series for compression, SMC imbalances, and momentum patterns.
        """
        data = df.copy()

        # 1. Bollinger Bands vs Keltner Channels (TTM Squeeze Mechanism)
        ma20 = data['Close'].rolling(window=config.BB_PERIOD).mean()
        std20 = data['Close'].rolling(window=config.BB_PERIOD).std()
        bb_upper = ma20 + (config.BB_STD * std20)
        bb_lower = ma20 - (config.BB_STD * std20)

        # Keltner Channels (using ATR14)
        kc_upper = ma20 + (config.KC_MULT * data['ATR14'])
        kc_lower = ma20 - (config.KC_MULT * data['ATR14'])

        # Squeeze is active if Bollinger Bands are compressed entirely inside Keltner Channels
        latest_squeeze = bool(
            (bb_upper.iloc[-1] < kc_upper.iloc[-1]) and 
            (bb_lower.iloc[-1] > kc_lower.iloc[-1])
        )

        # Also check if squeeze was recently released (fired) in last 3 bars
        squeeze_recent = any([
            (bb_upper.iloc[-i] < kc_upper.iloc[-i]) and (bb_lower.iloc[-i] > kc_lower.iloc[-i])
            for i in range(1, min(5, len(data)))
        ])
        is_squeeze_setup = latest_squeeze or (squeeze_recent and data['Close'].iloc[-1] > data['Close'].iloc[-2])

        # 2. Bullish Fair Value Gap (FVG) Detection (3-candle sequence: Low of bar 0 > High of bar -2)
        fvg_detected = False
        if len(data) >= 3:
            for i in range(-1, -4, -1):
                c3_low = data['Low'].iloc[i]
                c1_high = data['High'].iloc[i - 2]
                c2_body = abs(data['Close'].iloc[i - 1] - data['Open'].iloc[i - 1])
                c2_atr = data['ATR14'].iloc[i - 1]
                
                # Gap exists and middle candle is strong displacement
                if (c3_low > c1_high) and (c2_body > 0.8 * c2_atr):
                    fvg_detected = True
                    break

        # 3. Institutional Order Block (OB) Detection
        # A down candle preceding an aggressive breakout impulse that retests the zone
        order_block_detected = False
        if len(data) >= 10:
            swing_high = data['High'].iloc[-15:-3].max()
            current_close = data['Close'].iloc[-1]
            if current_close > swing_high:
                # Find the last bearish candle before this impulse
                for j in range(-2, -8, -1):
                    if data['Close'].iloc[j] < data['Open'].iloc[j]:
                        order_block_detected = True
                        break

        # 4. Liquidity Sweep Detection
        # Price broke below the 10-bar low but closed back above it with a strong wick
        liquidity_sweep = False
        if len(data) >= 15:
            prev_10_low = data['Low'].iloc[-12:-2].min()
            recent_low = data['Low'].iloc[-2:].min()
            recent_close = data['Close'].iloc[-1]
            if (recent_low < prev_10_low) and (recent_close > prev_10_low):
                liquidity_sweep = True

        # 5. Break of Structure (BOS)
        # Price closes above the 20-period swing high
        lookback_swing = min(len(data) - 2, 20)
        recent_swing_high = data['High'].iloc[-lookback_swing:-1].max()
        break_of_structure = bool(data['Close'].iloc[-1] > recent_swing_high)

        # 6. Candlestick Pattern Recognition
        candlestick_pattern = self._identify_candlestick_pattern(data)

        # 7. Invalidation Level (Support floor)
        # Lowest price of the last 5 bars or lower Keltner channel
        invalidation_level = float(min(data['Low'].iloc[-5:].min(), kc_lower.iloc[-1]))

        # 8. Composite Technical Score (0 - 100)
        score = 0.0
        if is_squeeze_setup: score += 25.0
        if fvg_detected: score += 20.0
        if break_of_structure: score += 25.0
        if liquidity_sweep: score += 15.0
        if order_block_detected: score += 10.0
        if "Bullish" in candlestick_pattern or "Hammer" in candlestick_pattern: score += 5.0

        # Cap score between 0 and 100
        score = min(max(score, 0.0), 100.0)

        return PriceActionSignal(
            ticker=snapshot.ticker,
            squeeze_active=latest_squeeze,
            fvg_detected=fvg_detected,
            order_block_detected=order_block_detected,
            liquidity_sweep=liquidity_sweep,
            break_of_structure=break_of_structure,
            candlestick_pattern=candlestick_pattern,
            technical_score=round(score, 1),
            support_invalidation_level=round(invalidation_level, 2)
        )

    def _identify_candlestick_pattern(self, df: pd.DataFrame) -> str:
        """Evaluates latest candlestick formation."""
        c = df.iloc[-1]
        prev = df.iloc[-2]
        
        body = c['Close'] - c['Open']
        abs_body = abs(body)
        total_range = c['High'] - c['Low']
        upper_wick = c['High'] - max(c['Close'], c['Open'])
        lower_wick = min(c['Close'], c['Open']) - c['Low']

        if total_range == 0:
            return "Doji / Indecision"

        # Bullish Marubozu (Huge green body, almost no wicks)
        if body > 0 and (abs_body / total_range) >= 0.85:
            return "Bullish Marubozu (Strong Conviction)"

        # Bullish Engulfing
        prev_body = prev['Close'] - prev['Open']
        if prev_body < 0 and body > 0 and c['Close'] > prev['Open'] and c['Open'] < prev['Close']:
            return "Bullish Engulfing"

        # Hammer / Pin Bar (Long lower wick >= 2x body, small upper wick)
        if lower_wick >= 2.0 * abs_body and upper_wick <= 0.3 * abs_body:
            return "Bullish Hammer / Pin Bar"

        # Morning Star check
        if len(df) >= 3:
            p2 = df.iloc[-3]
            if (p2['Close'] < p2['Open']) and (abs(prev['Close'] - prev['Open']) < 0.3 * (p2['High'] - p2['Low'])) and (body > 0 and c['Close'] > (p2['Open'] + p2['Close']) / 2):
                return "Morning Star"

        if body > 0:
            return "Bullish Expansion Candle"
        else:
            return "Consolidation / Pullback Candle"
