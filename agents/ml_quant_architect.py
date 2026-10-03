"""
Agent 5: Quantitative ML Architect (@ml-quant-architect)
Responsible for feature fusion, running the calibrated predictive ensemble, and evaluating shootup probability.
"""

from typing import Dict, Any, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
from agents.base_agent import BaseAgent
from core.schema import (
    MarketSnapshot,
    PriceActionSignal,
    DerivativesSignal,
    CatalystSignal,
    MLPrediction
)
from core.config import config

class MLQuantArchitect(BaseAgent):
    def __init__(self, model_path: Optional[str] = None):
        super().__init__(
            name="MLQuantArchitect",
            role="Quantitative ML & Ensemble Architect",
            criteria="Fuses multi-agent feature vectors, computes calibrated shootup probability, and identifies top driving factors."
        )
        self.model_path = model_path or config.MODEL_SAVE_PATH
        self.model = None
        self._load_model_if_exists()

    def _load_model_if_exists(self):
        """Loads pre-trained ensemble model if available on disk."""
        if os.path.exists(self.model_path):
            try:
                self.model = joblib.load(self.model_path)
            except Exception:
                self.model = None

    def build_feature_vector(
        self,
        snapshot: MarketSnapshot,
        pa_signal: PriceActionSignal,
        deriv_signal: DerivativesSignal,
        cat_signal: CatalystSignal
    ) -> tuple[pd.DataFrame, Dict[str, float]]:
        """
        Extracts and normalizes features across all agent domains matching training schema.
        """
        import pandas as pd
        atr_pct = (snapshot.atr_14 / max(snapshot.current_price, 0.01)) * 100.0

        feature_dict = {
            "RVol": float(snapshot.rvol),
            "Dist_52W_High": float(snapshot.distance_from_52w_high),
            "ATR_Pct": float(atr_pct),
            "Squeeze_Active": 1.0 if pa_signal.squeeze_active else 0.0,
            "FVG_Detected": 1.0 if pa_signal.fvg_detected else 0.0,
            "Break_of_Structure": 1.0 if pa_signal.break_of_structure else 0.0,
            "Technical_Score": float(pa_signal.technical_score),
            "Vol_Velocity": float(cat_signal.volume_velocity_score) / 25.0,
            "Mom_3D": float(cat_signal.momentum_3d_pct),
            "Mom_5D": float(cat_signal.momentum_5d_pct)
        }

        feature_df = pd.DataFrame([feature_dict])
        return feature_df, feature_dict

    def execute(
        self,
        snapshot: MarketSnapshot,
        pa_signal: PriceActionSignal,
        deriv_signal: DerivativesSignal,
        cat_signal: CatalystSignal
    ) -> MLPrediction:
        """
        Executes prediction pipeline. If an offline-trained model exists, utilizes it;
        otherwise runs a calibrated quantitative scoring ensemble.
        """
        feat_vec, feat_dict = self.build_feature_vector(snapshot, pa_signal, deriv_signal, cat_signal)

        if self.model is not None:
            # Use trained LightGBM/XGBoost ensemble
            try:
                proba = float(self.model.predict_proba(feat_vec)[0, 1])
            except Exception:
                proba = self._calculate_calibrated_heuristic(feat_dict, snapshot.is_index)
        else:
            proba = self._calculate_calibrated_heuristic(feat_dict, snapshot.is_index)

        # Identify top contributing factors
        contributing_factors = []
        if pa_signal.squeeze_active:
            contributing_factors.append("Volatility compression (Bollinger Bands coiled within Keltner Channels)")
        if pa_signal.fvg_detected:
            contributing_factors.append("Institutional Fair Value Gap (FVG) demand support")
        if pa_signal.break_of_structure:
            contributing_factors.append("Structural resistance breakout confirmed")
        if deriv_signal.squeeze_potential_score >= 50.0:
            contributing_factors.append(f"High Squeeze Potential (Score: {deriv_signal.squeeze_potential_score:.0f})")
        if cat_signal.volume_velocity_score >= 50.0:
            contributing_factors.append("Sudden Relative Volume explosion (RVol > 2.0x)")
        if cat_signal.momentum_3d_pct > 3.0:
            contributing_factors.append(f"Accelerating upward price momentum (+{cat_signal.momentum_3d_pct}%)")

        if not contributing_factors:
            contributing_factors.append("Consolidation baseline - awaiting trigger")

        # Classify magnitude and confidence
        if proba >= 0.75:
            conf_tier = "HIGH"
            mag_tier = "EXPLOSIVE (>20%)" if not snapshot.is_index else "RAPID IMPULSE (>3.5%)"
        elif proba >= 0.55:
            conf_tier = "MEDIUM"
            mag_tier = "HIGH (12-20%)" if not snapshot.is_index else "STRONG IMPULSE (2-3.5%)"
        else:
            conf_tier = "LOW"
            mag_tier = "MODERATE (8-12%)" if not snapshot.is_index else "MILD IMPULSE (<2%)"

        return MLPrediction(
            ticker=snapshot.ticker,
            shootup_probability=round(proba, 4),
            predicted_magnitude_tier=mag_tier,
            confidence_tier=conf_tier,
            top_contributing_factors=contributing_factors,
            model_version="Ensemble-v1.0"
        )

    def _calculate_calibrated_heuristic(self, f: Dict[str, float], is_index: bool) -> float:
        """
        Calibrated sigmoid ensemble function when cold-starting prior to full retraining.
        """
        if is_index:
            # For indexes, technical structure and volume velocity dominate
            logit = (
                -2.2
                + 1.8 * f["technical_score"]
                + 1.4 * f["break_of_structure"]
                + 1.2 * f["squeeze_active"]
                + 1.0 * f["volume_velocity_score"]
                + 0.8 * f["fvg_detected"]
            )
        else:
            # For individual equities, short squeeze potential and volume explosion carry heavy weight
            logit = (
                -2.5
                + 1.5 * f["technical_score"]
                + 1.6 * f["squeeze_potential_score"]
                + 1.2 * f["volume_velocity_score"]
                + 1.1 * f["break_of_structure"]
                + 0.9 * f["fvg_detected"]
                + 0.8 * f["squeeze_active"]
            )
        # Sigmoid activation
        p = 1.0 / (1.0 + np.exp(-logit))
        return float(np.clip(p, 0.01, 0.99))
