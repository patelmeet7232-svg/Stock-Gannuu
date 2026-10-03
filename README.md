# Multi-Agent Excessive Shootup Prediction Engine (Stocks & Indexes)

An autonomous multi-agent quantitative framework designed to detect and predict **excessive shootups** (explosive breakouts, short squeezes, gamma squeezes, and volatility expansions) across **US Equities and Major Indexes** (S&P 500, Nasdaq 100, Russell 2000, and High Short-Interest Growth Equities) over a **multi-day swing horizon (1–5 trading days)**.

---

## 1. The Agent Squad & Task Criteria

Each agent operates as an independent domain specialist with strict evaluation criteria:

| Agent Name | Specialist Role | Primary Criteria & Tasks Assigned |
| :--- | :--- | :--- |
| **Market Data Scout**<br>`@data-scout` | Universe Screening & OHLCV Ingestion | • Ingests clean daily OHLCV from real-time market feeds.<br>• Calculates 20-period Moving Average Volume, Relative Volume ($RVol \ge 2.0x$), and 14-period True Range ($ATR_{14}$).<br>• Screens candidates by baseline liquidity and proximity to 52-week highs. |
| **Price Action & SMC Analyst**<br>`@price-action-analyst` | Technical Structure & Liquidity Voids | • **Volatility Compression**: Detects Bollinger Bands contracting completely inside Keltner Channels.<br>• **Smart Money Concepts (SMC)**: Bullish Fair Value Gaps (FVG), Bullish Order Blocks (OB), Liquidity Sweeps of previous swing lows, and Break of Structure (BOS).<br>• Candlestick momentum patterns (Bullish Marubozu, Bullish Engulfing, Hammer Pin Bars). |
| **Derivatives & Squeeze Specialist**<br>`@derivatives-analyst` | Short Squeeze & Gamma Dynamics | • Tracks Short Interest % of Float and Days-to-Cover (Short Ratio).<br>• Evaluates front-month Options Chains: Call vs. Put Open Interest skew ($PCR < 0.6$ indicating gamma squeeze fuel).<br>• Computes Squeeze Potential Index ($0-100$) and classifies regime (LOW, MODERATE, ELEVATED, EXTREME). |
| **Catalyst & Sentiment Radar**<br>`@catalyst-radar` | Momentum Velocity & News Signals | • Evaluates sudden Relative Volume acceleration ($Vol_{velocity}$).<br>• Measures 3-day and 5-day price rate-of-change momentum.<br>• Scans material news headlines and catalyst triggers. |
| **Quantitative ML Architect**<br>`@ml-quant-architect` | Predictive Ensemble & Calibration | • Fuses a 16-dimensional multi-domain feature vector.<br>• Executes a calibrated Gradient Boosting + Random Forest ensemble trained on historical multi-day upside anomalies.<br>• Generates calibrated probability $P(\text{Shootup} \ge 8\% \text{ within 5 days})$ and identifies top driving factors. |
| **Risk & Backtest Validator**<br>`@risk-validator` | False Breakout Defense & Trade Planning | • Filters out low-volume traps and overextended bull traps.<br>• Computes dynamic Stop-Loss at the invalidation level ($StopLoss \le entry - \text{drawdown floor}$).<br>• Formulates multi-target Take-Profits ($Target_1$ at 2.0R, $Target_2$ at 3.5R) and final trade verdict. |

---

## 2. Directory Structure

```
Antigravity/
├── core/
│   ├── __init__.py
│   ├── config.py           # Universe lists, shootup thresholds, model hyperparams
│   └── schema.py           # Pydantic data contracts between agents
├── agents/
│   ├── __init__.py
│   ├── base_agent.py       # Base agent class with timing, isolation, and logging
│   ├── data_scout.py       # @data-scout agent
│   ├── price_action_analyst.py # @price-action-analyst agent (SMC + Squeeze)
│   ├── derivatives_analyst.py  # @derivatives-analyst agent (Shorts + Options)
│   ├── catalyst_radar.py   # @catalyst-radar agent (Volume velocity + News)
│   ├── ml_quant_architect.py   # @ml-quant-architect agent (Predictive ML)
│   └── risk_validator.py   # @risk-validator agent (Risk & Trade Plan)
├── pipeline/
│   ├── __init__.py
│   └── orchestrator.py     # ShootupPredictionTeam orchestrator
├── train_model.py          # Historical data download, labeling, and training pipeline
├── main.py                 # CLI interface for market scanning and analysis
└── requirements.txt        # Python dependency manifest
```

---

## 3. How to Run the System

### 1. In-Depth Multi-Agent Audit for a Specific Stock or Index
Inspect each agent's individual audit, metrics, and actionable trade plan:
```bash
python main.py analyze --ticker NVDA
python main.py analyze --ticker SPY
python main.py analyze --ticker TSLA
python main.py analyze --ticker MARA
```

### 2. Market-Wide Shootup Scan
Scan the predefined universe (S&P 500, Nasdaq, Russell 2000, high short-interest momentum stocks) to rank assets by shootup probability:
```bash
python main.py scan
```
Or scan custom tickers:
```bash
python main.py scan --tickers AAPL MSFT AMZN PLTR GME
```

### 3. Retrain and Calibrate the ML Ensemble Model
Pulls 3 years of historical market data, extracts features, labels shootup events, trains an ensemble with class balancing, and calibrates probabilities:
```bash
python main.py train
```
