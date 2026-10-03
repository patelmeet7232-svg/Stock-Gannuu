"""
Agent 3: Derivatives & Squeeze Specialist (@derivatives-analyst)
Specializes in Short Squeeze Mechanics, Options Open Interest Skew, and Gamma Risk Analysis.
"""

from typing import Dict, Any, Optional
import yfinance as yf
from agents.base_agent import BaseAgent
from core.schema import DerivativesSignal, MarketSnapshot
from core.config import config

class DerivativesSqueezeAnalyst(BaseAgent):
    def __init__(self):
        super().__init__(
            name="DerivativesSqueezeAnalyst",
            role="Derivatives & Squeeze Specialist",
            criteria="Analyzes Short Float %, Days-to-Cover, Options Call/Put Open Interest ratio, and Gamma Squeeze risk."
        )

    def execute(self, snapshot: MarketSnapshot) -> DerivativesSignal:
        """
        Inspects short float metrics and options chain structure for squeeze potential.
        """
        ticker = snapshot.ticker
        stock = yf.Ticker(ticker)

        short_float_pct = None
        days_to_cover = None
        put_call_oi_ratio = None
        commentary_parts = []
        squeeze_score = 0.0

        # Retrieve short interest info for equities
        try:
            info = stock.info or {}
            short_float_pct = info.get("shortPercentOfFloat")
            if short_float_pct is not None:
                short_float_pct = float(short_float_pct) * 100.0  # Convert to %
            
            days_to_cover = info.get("shortRatio")
            if days_to_cover is not None:
                days_to_cover = float(days_to_cover)
        except Exception:
            pass

        # Retrieve options chain metrics if available
        try:
            expirations = stock.options
            if expirations and len(expirations) > 0:
                # Inspect front-month options chain
                front_exp = expirations[0]
                opt = stock.option_chain(front_exp)
                calls = opt.calls
                puts = opt.puts

                total_call_oi = calls['openInterest'].fillna(0).sum()
                total_put_oi = puts['openInterest'].fillna(0).sum()

                if total_call_oi > 0:
                    put_call_oi_ratio = round(float(total_put_oi / total_call_oi), 2)
                    
                    # Heavy call skew (more calls than puts) often indicates gamma squeeze positioning
                    if put_call_oi_ratio < 0.6:
                        squeeze_score += 25.0
                        commentary_parts.append(f"Heavy Call Open Interest skew (PCR: {put_call_oi_ratio}), high gamma squeeze potential.")
                    elif put_call_oi_ratio > 1.4:
                        commentary_parts.append(f"High Put Open Interest (PCR: {put_call_oi_ratio}), indicating hedging or bearish bias.")
        except Exception:
            pass

        # Evaluate Short Float
        if short_float_pct is not None:
            if short_float_pct >= 25.0:
                squeeze_score += 45.0
                commentary_parts.append(f"Extreme short interest ({short_float_pct:.1f}% float short). Prime candidate for short squeeze.")
            elif short_float_pct >= 15.0:
                squeeze_score += 30.0
                commentary_parts.append(f"High short interest ({short_float_pct:.1f}% float short).")
            elif short_float_pct >= 8.0:
                squeeze_score += 15.0

        # Evaluate Days to Cover
        if days_to_cover is not None:
            if days_to_cover >= 6.0:
                squeeze_score += 30.0
                commentary_parts.append(f"High days to cover ({days_to_cover:.1f} days), shorts will struggle to exit.")
            elif days_to_cover >= 3.5:
                squeeze_score += 20.0
                commentary_parts.append(f"Moderate-to-high days to cover ({days_to_cover:.1f} days).")

        # For Indexes (SPY, QQQ, ^NDX, ^GSPC), short float is not the main driver; options flow is.
        if snapshot.is_index:
            # Index squeeze is typically a short-covering rally or dealer vanna/charm rally
            if put_call_oi_ratio is not None and put_call_oi_ratio > 1.2:
                # Oversold puts being closed often triggers massive index rallies
                squeeze_score = max(squeeze_score, 40.0)
                commentary_parts.append("Elevated index put hedging creates fertile ground for short-covering upside squeeze.")
            else:
                squeeze_score = max(squeeze_score, 20.0)
                commentary_parts.append("Standard index options distribution.")

        squeeze_score = min(max(squeeze_score, 0.0), 100.0)

        # Classify regime
        if squeeze_score >= 70.0:
            regime = "EXTREME"
        elif squeeze_score >= 50.0:
            regime = "ELEVATED"
        elif squeeze_score >= 25.0:
            regime = "MODERATE"
        else:
            regime = "LOW"

        if not commentary_parts:
            commentary_parts.append("Normal derivatives and short positioning.")

        return DerivativesSignal(
            ticker=ticker,
            short_float_pct=round(short_float_pct, 2) if short_float_pct else None,
            days_to_cover=round(days_to_cover, 2) if days_to_cover else None,
            put_call_oi_ratio=put_call_oi_ratio,
            squeeze_potential_score=round(squeeze_score, 1),
            squeeze_regime=regime,
            commentary=" ".join(commentary_parts)
        )
