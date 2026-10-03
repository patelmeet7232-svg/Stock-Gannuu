"""
Configuration settings for the Multi-Agent Excessive Shootup Prediction Engine.
Configured for US Equities & Indexes with Multi-Day Swing Horizon (1-5 trading days).
"""

from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class SystemConfig:
    # Target Market Universe: US Equities & Indexes
    INDEX_WATCHLIST: List[str] = field(default_factory=lambda: [
        "^GSPC",   # S&P 500 Index
        "^NDX",    # Nasdaq 100 Index
        "QQQ",     # Invesco QQQ Trust
        "SPY",     # SPDR S&P 500 ETF Trust
        "IWM",     # iShares Russell 2000 ETF
    ])

    STOCK_WATCHLIST: List[str] = field(default_factory=lambda: [
        "NVDA", "TSLA", "AMD", "PLTR", "SMCI", 
        "COIN", "MARA", "MSTR", "UPST", "AFRM", 
        "CVNA", "SOFI", "ARM", "RIVN", "LCID",
        "GME", "AMC", "IONQ", "DKNG", "AI"
    ])

    # Target Market Universe: Indian Equities & Indexes (NSE / BSE)
    INDIAN_INDEX_WATCHLIST: List[str] = field(default_factory=lambda: [
        "^NSEI",     # NIFTY 50
        "^NSEBANK",  # BANK NIFTY
    ])

    INDIAN_STOCK_WATCHLIST: List[str] = field(default_factory=lambda: [
        "RELIANCE.NS",   # Reliance Industries (Energy & Telecom)
        "TRENT.NS",      # Trent Ltd (Tata Retail momentum leader)
        "HAL.NS",        # Hindustan Aeronautics (Defense)
        "BEL.NS",        # Bharat Electronics (Defense)
        "IRFC.NS",       # Indian Railway Finance Corp (Railway momentum)
        "RVNL.NS",       # Rail Vikas Nigam Ltd (Railway infrastructure)
        "SUZLON.NS",     # Suzlon Energy (Renewable energy & high-beta)
        "BSE.NS",        # BSE Ltd (Capital market momentum)
        "ADANIENT.NS",   # Adani Enterprises (High-beta flagship)
        "HDFCBANK.NS",   # HDFC Bank (Private Banking heavyweight)
        "SBIN.NS",       # State Bank of India (PSU Banking leader)
        "TATASTEEL.NS",  # Tata Steel
        "VEDL.NS",       # Vedanta Ltd (Metals & high-dividend)
        "JIOFIN.NS",     # Jio Financial Services
        "KALYANKJIL.NS", # Kalyan Jewellers (Consumption breakout)
        "COCHINSHIP.NS", # Cochin Shipyard (Defense shipbuilder)
        "MAZDOCK.NS",    # Mazagon Dock Shipbuilders
        "IREDA.NS",      # Indian Renewable Energy Dev Agency
        "DIXON.NS",      # Dixon Technologies (Electronics manufacturing)
    ])

    # Prediction Target Definition (Multi-Day Swing)
    SWING_HORIZON_DAYS: int = 5          # Forecast window: next 1 to 5 trading days
    STOCK_SHOOTUP_THRESHOLD: float = 0.08 # +8.0% minimum gain for single stock to qualify as shootup
    STOCK_EXPLOSIVE_THRESHOLD: float = 0.15 # +15.0% explosive surge
    INDEX_SHOOTUP_THRESHOLD: float = 0.025 # +2.5% move for index (since indices are less volatile)
    MAX_ALLOWABLE_DRAWDOWN: float = 0.035 # Invalidation threshold: max initial dip before the surge

    # Technical Compression & SMC Criteria
    BB_PERIOD: int = 20
    BB_STD: float = 2.0
    KC_PERIOD: int = 20
    KC_MULT: float = 1.5
    RVOL_THRESHOLD: float = 2.0           # Relative Volume >= 2x 20-day average
    FVG_THRESHOLD_PERCENT: float = 0.008  # 0.8% Fair Value Gap minimum size

    # Squeeze & Derivatives Criteria
    HIGH_SHORT_INTEREST_THRESHOLD: float = 0.15  # Short float >= 15%
    DAYS_TO_COVER_THRESHOLD: float = 3.5         # Days to cover >= 3.5 days
    CALL_PUT_OI_RATIO_BULLISH: float = 1.3       # Heavy call open interest

    # Model Parameters
    CONFIDENCE_ALERT_THRESHOLD: float = 0.65     # Calibrated probability >= 65% triggers high-priority alert
    RANDOM_SEED: int = 42
    MODEL_SAVE_PATH: str = "models/shootup_ensemble.pkl"

config = SystemConfig()
