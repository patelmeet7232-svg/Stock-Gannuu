"""
Main Command Line Interface for the Multi-Agent Excessive Shootup Prediction Engine.
Usage:
  python main.py scan                  # Scans predefined US Equities & Indexes Watchlist
  python main.py analyze --ticker NVDA # Deep dive analysis for a single ticker
  python main.py train                 # Retrains and calibrates the ML Ensemble
"""

import sys
import argparse
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pipeline.orchestrator import ShootupPredictionTeam
from core.config import config
import train_model

console = Console()

def run_scan(team: ShootupPredictionTeam, custom_tickers: list[str] = None):
    tickers = custom_tickers if custom_tickers else (config.INDEX_WATCHLIST + config.STOCK_WATCHLIST)
    team.scan_universe(tickers)

def run_analyze(team: ShootupPredictionTeam, ticker: str):
    console.print(f"\n[bold magenta]Running In-Depth Multi-Agent Audit for: {ticker}[/bold magenta]\n")
    trade_signal, artifacts = team.analyze_asset(ticker)

    if trade_signal is None:
        console.print(f"[red]Failed to analyze ticker {ticker}.[/red]")
        return

    # Print Agent Reports Table
    exec_table = Table(title="Agent Team Execution Audit", header_style="bold cyan")
    exec_table.add_column("Agent", style="bold white")
    exec_table.add_column("Role", style="cyan")
    exec_table.add_column("Execution Time", justify="right")
    exec_table.add_column("Status", justify="center")

    for rep in artifacts.get("execution_reports", []):
        status = "[green]SUCCESS[/green]" if rep.success else "[red]FAILED[/red]"
        exec_table.add_row(rep.agent_name, rep.role, f"{rep.execution_time_seconds:.3f}s", status)

    console.print(exec_table)

    # Detailed Breakdown Cards
    pa = artifacts.get("price_action")
    deriv = artifacts.get("derivatives")
    cat = artifacts.get("catalyst")
    ml = artifacts.get("ml_prediction")

    curr = "Rs." if (ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^NSE")) else "$"

    breakdown = f"""
[bold yellow]1. Technical & SMC Analysis:[/bold yellow]
  • Technical Score: [bold]{pa.technical_score}/100[/bold]
  • Squeeze Active: {'[bold green]YES[/bold green]' if pa.squeeze_active else 'No'}
  • Bullish FVG: {'[bold green]DETECTED[/bold green]' if pa.fvg_detected else 'None'}
  • Order Block: {'[bold green]DETECTED[/bold green]' if pa.order_block_detected else 'None'}
  • Break of Structure: {'[bold green]CONFIRMED[/bold green]' if pa.break_of_structure else 'No'}
  • Candlestick Formation: [italic]{pa.candlestick_pattern}[/italic]
  • Invalidation Floor: {curr}{pa.support_invalidation_level:.2f}

[bold yellow]2. Derivatives & Squeeze Dynamics:[/bold yellow]
  • Squeeze Potential Score: [bold]{deriv.squeeze_potential_score}/100[/bold] ({deriv.squeeze_regime})
  • Short Float %: {f"{deriv.short_float_pct}%" if deriv.short_float_pct else "N/A"}
  • Days to Cover: {f"{deriv.days_to_cover} days" if deriv.days_to_cover else "N/A"}
  • Put/Call OI Ratio: {deriv.put_call_oi_ratio or "N/A"}
  • Commentary: [italic]{deriv.commentary}[/italic]

[bold yellow]3. Catalyst & Momentum Radar:[/bold yellow]
  • Catalyst Score: [bold]{cat.catalyst_score}/100[/bold]
  • 3-Day Momentum: {cat.momentum_3d_pct}% | 5-Day Momentum: {cat.momentum_5d_pct}%
  • Drivers: {", ".join(cat.key_drivers[:2])}

[bold yellow]4. Quantitative ML Ensemble:[/bold yellow]
  • Calibrated Shootup Probability: [bold green]{ml.shootup_probability * 100:.1f}%[/bold green]
  • Confidence Tier: [bold]{ml.confidence_tier}[/bold]
  • Expected Surge Magnitude: [bold]{ml.predicted_magnitude_tier}[/bold]
  • Key Factors: {"; ".join(ml.top_contributing_factors)}
"""
    console.print(Panel(breakdown, title=f"Intelligence Breakdown: {ticker}", border_style="cyan"))

    # Final Verdict Card
    trade_summary = f"""
[bold]VERDICT:[/bold] {trade_signal.verdict}
[bold]Current Entry:[/bold] {curr}{trade_signal.entry_price:.2f}
[bold]Stop-Loss:[/bold] {curr}{trade_signal.stop_loss:.2f} (Max Risk: {curr}{(trade_signal.entry_price - trade_signal.stop_loss):.2f})
[bold]Target 1 (2R):[/bold] {curr}{trade_signal.target_1:.2f}
[bold]Target 2 (3.5R):[/bold] {curr}{trade_signal.target_2:.2f}
[bold]Risk/Reward Ratio:[/bold] {trade_signal.risk_reward_ratio}:1
[bold]Strategic Rationale:[/bold] {trade_signal.synthesis_summary}
"""
    console.print(Panel(trade_summary, title=f"Actionable Trade Plan: {ticker}", border_style="green" if "BUY" in trade_signal.verdict or "SQUEEZE" in trade_signal.verdict else "yellow"))

def main():
    parser = argparse.ArgumentParser(description="Multi-Agent Excessive Shootup Prediction Engine")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # Scan command
    scan_parser = subparsers.add_parser("scan", help="Scan the universe of stocks and indexes")
    scan_parser.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    scan_parser.add_argument("--market", choices=["us", "india", "all"], default="us", help="Market to scan (us, india, all)")

    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Detailed multi-agent audit for a single ticker")
    analyze_parser.add_argument("--ticker", type=str, required=True, help="Ticker symbol (e.g., RELIANCE.NS, ^NSEI, NVDA)")

    # Train command
    subparsers.add_parser("train", help="Train and calibrate the ML ensemble on historical data")

    args = parser.parse_args()

    if args.command == "train":
        train_model.train_ensemble_model()
        return

    # Initialize agent team
    team = ShootupPredictionTeam()

    if args.command == "analyze":
        run_analyze(team, args.ticker.upper())
    elif args.command == "scan":
        if args.tickers:
            target_list = [t.upper() for t in args.tickers]
        elif args.market == "india":
            target_list = config.INDIAN_INDEX_WATCHLIST + config.INDIAN_STOCK_WATCHLIST
        elif args.market == "all":
            target_list = config.INDEX_WATCHLIST + config.STOCK_WATCHLIST + config.INDIAN_INDEX_WATCHLIST + config.INDIAN_STOCK_WATCHLIST
        else:
            target_list = config.INDEX_WATCHLIST + config.STOCK_WATCHLIST
        run_scan(team, target_list)
    else:
        # Default behavior: run scan
        run_scan(team)

if __name__ == "__main__":
    main()
