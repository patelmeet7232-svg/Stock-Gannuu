"""
Lead Orchestrator: Multi-Agent Team Coordinator (ShootupPredictionTeam)
Coordinates the 6 specialist agents, aggregates multi-domain intelligence, and produces actionable trade signals.
"""

from typing import List, Dict, Any, Optional
import time
from datetime import datetime
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

from agents.data_scout import MarketDataScout
from agents.price_action_analyst import PriceActionSMCAnalyst
from agents.derivatives_analyst import DerivativesSqueezeAnalyst
from agents.catalyst_radar import CatalystSentimentRadar
from agents.ml_quant_architect import MLQuantArchitect
from agents.risk_validator import RiskBacktestValidator

from core.schema import (
    MarketSnapshot,
    PriceActionSignal,
    DerivativesSignal,
    CatalystSignal,
    MLPrediction,
    TradeSignal,
    AgentExecutionReport
)
from core.config import config

console = Console()

class ShootupPredictionTeam:
    def __init__(self, model_path: Optional[str] = None):
        console.print("[bold cyan]Initializing Multi-Agent Team for Excessive Shootup Prediction...[/bold cyan]")
        self.data_scout = MarketDataScout()
        self.pa_analyst = PriceActionSMCAnalyst()
        self.deriv_analyst = DerivativesSqueezeAnalyst()
        self.catalyst_radar = CatalystSentimentRadar()
        self.ml_architect = MLQuantArchitect(model_path=model_path)
        self.risk_validator = RiskBacktestValidator()

        self.agents = [
            self.data_scout,
            self.pa_analyst,
            self.deriv_analyst,
            self.catalyst_radar,
            self.ml_architect,
            self.risk_validator
        ]
        self._print_team_roster()

    def _print_team_roster(self):
        table = Table(title="Autonomous Agent Squad Roster", header_style="bold magenta")
        table.add_column("Agent Name", style="cyan", no_wrap=True)
        table.add_column("Specialist Role", style="green")
        table.add_column("Assigned Criteria", style="white")

        for agent in self.agents:
            table.add_row(agent.name, agent.role, agent.criteria)
        
        console.print(table)

    def analyze_asset(self, ticker: str) -> tuple[Optional[TradeSignal], Dict[str, Any]]:
        """
        Executes a synchronized analysis pipeline across all agents for a single ticker.
        """
        execution_reports: List[AgentExecutionReport] = []
        artifacts: Dict[str, Any] = {}

        try:
            # 1. Market Data Scout
            (df, snapshot), scout_rep = self.data_scout.run(ticker=ticker)
            execution_reports.append(scout_rep)
            artifacts["snapshot"] = snapshot

            # 2. Price Action & SMC Analyst
            pa_signal, pa_rep = self.pa_analyst.run(df=df, snapshot=snapshot)
            execution_reports.append(pa_rep)
            artifacts["price_action"] = pa_signal

            # 3. Derivatives & Squeeze Specialist
            deriv_signal, deriv_rep = self.deriv_analyst.run(snapshot=snapshot)
            execution_reports.append(deriv_rep)
            artifacts["derivatives"] = deriv_signal

            # 4. Catalyst & Sentiment Radar
            cat_signal, cat_rep = self.catalyst_radar.run(df=df, snapshot=snapshot)
            execution_reports.append(cat_rep)
            artifacts["catalyst"] = cat_signal

            # 5. Quantitative ML Architect
            ml_pred, ml_rep = self.ml_architect.run(
                snapshot=snapshot,
                pa_signal=pa_signal,
                deriv_signal=deriv_signal,
                cat_signal=cat_signal
            )
            execution_reports.append(ml_rep)
            artifacts["ml_prediction"] = ml_pred

            # 6. Risk & False Breakout Validator
            trade_signal, risk_rep = self.risk_validator.run(
                snapshot=snapshot,
                pa_signal=pa_signal,
                deriv_signal=deriv_signal,
                cat_signal=cat_signal,
                ml_pred=ml_pred
            )
            execution_reports.append(risk_rep)
            artifacts["trade_signal"] = trade_signal
            artifacts["execution_reports"] = execution_reports

            return trade_signal, artifacts

        except Exception as e:
            console.print(f"[bold red]Pipeline failure analyzing {ticker}:[/bold red] {str(e)}")
            return None, {"error": str(e), "execution_reports": execution_reports}

    def scan_universe(self, tickers: Optional[List[str]] = None) -> List[TradeSignal]:
        """
        Scans a candidate universe, sorts assets by shootup probability, and presents a visual dashboard.
        """
        target_tickers = tickers or (config.INDEX_WATCHLIST + config.STOCK_WATCHLIST)
        console.print(f"\n[bold yellow]Initiating multi-agent scan across {len(target_tickers)} assets...[/bold yellow]\n")

        results: List[TradeSignal] = []

        for ticker in target_tickers:
            console.print(f"Scanning [bold]{ticker}[/bold]...")
            signal, _ = self.analyze_asset(ticker)
            if signal is not None:
                results.append(signal)

        # Sort by Shootup Probability descending
        results.sort(key=lambda s: s.shootup_probability, reverse=True)

        self._display_scan_dashboard(results)
        return results

    def _display_scan_dashboard(self, signals: List[TradeSignal]):
        table = Table(title="Excessive Shootup Prediction Dashboard", header_style="bold green")
        table.add_column("Ticker", style="bold white")
        table.add_column("Shootup Prob", justify="center")
        table.add_column("Verdict", style="bold")
        table.add_column("Current Price", justify="right")
        table.add_column("Stop Loss", justify="right", style="red")
        table.add_column("Target 1 (2R)", justify="right", style="green")
        table.add_column("Target 2 (3.5R)", justify="right", style="bold green")
        table.add_column("SMC / Tech", justify="center")
        table.add_column("Squeeze Risk", justify="center")

        for s in signals:
            prob_pct = s.shootup_probability * 100.0
            prob_str = f"{prob_pct:.1f}%"
            if prob_pct >= 65.0:
                prob_styled = f"[bold green]{prob_str}[/bold green]"
                verdict_styled = f"[bold green]{s.verdict}[/bold green]"
            elif prob_pct >= 45.0:
                prob_styled = f"[yellow]{prob_str}[/yellow]"
                verdict_styled = f"[yellow]{s.verdict}[/yellow]"
            else:
                prob_styled = f"[dim]{prob_str}[/dim]"
                verdict_styled = f"[dim]{s.verdict}[/dim]"

            tech_score = f"{s.agent_scores.get('technical_score', 0):.0f}"
            squeeze_score = f"{s.agent_scores.get('squeeze_potential_score', 0):.0f}"
            curr_sym = "Rs." if (s.ticker.endswith(".NS") or s.ticker.endswith(".BO") or s.ticker.startswith("^NSE")) else "$"

            table.add_row(
                s.ticker,
                prob_styled,
                verdict_styled,
                f"{curr_sym}{s.entry_price:.2f}",
                f"{curr_sym}{s.stop_loss:.2f}",
                f"{curr_sym}{s.target_1:.2f}",
                f"{curr_sym}{s.target_2:.2f}",
                tech_score,
                squeeze_score
            )

        console.print(table)
