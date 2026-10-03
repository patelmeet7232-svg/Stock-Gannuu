"""
Autonomous Continuous Market Monitor Daemon.
Continuously scans the coiled watchlist and entire market universe for breakout shootup triggers.
Alerts when an asset breaches its invalidation breakout trigger or an agent signal upgrades to actionable buy.
"""

import sys
import time
import os
from datetime import datetime
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from pipeline.orchestrator import ShootupPredictionTeam
from core.config import config
from core.schema import TradeSignal

console = Console()

# Defined Trigger Levels from Agent Analysis
TRIGGER_LEVELS = {
    # Indian Coiled Setups
    "TATASTEEL.NS": {"trigger_price": 180.50, "stop_loss": 175.75, "target": 185.88, "condition": "above"},
    "IREDA.NS": {"trigger_price": 113.50, "stop_loss": 107.11, "target": 124.62, "condition": "above"},
    "ADANIENT.NS": {"trigger_price": 2860.00, "stop_loss": 2772.00, "target": 2973.60, "condition": "above"},
    # US Coiled & High Squeeze Setups
    "RIVN": {"trigger_price": 14.75, "stop_loss": 13.80, "target": 15.49, "condition": "above"},
    "AI": {"trigger_price": 11.45, "stop_loss": 10.62, "target": 12.38, "condition": "above"},
    "AFRM": {"trigger_price": 72.50, "stop_loss": 68.29, "target": 79.45, "condition": "above"},
    "TSLA": {"trigger_price": 382.00, "stop_loss": 357.62, "target": 415.98, "condition": "above"},
    "MARA": {"trigger_price": 11.85, "stop_loss": 10.84, "target": 12.60, "condition": "above"},
}

def run_monitor_check(team: ShootupPredictionTeam) -> list[dict]:
    """
    Scans the priority watchlist and checks if any trigger conditions or actionable signals are met.
    """
    console.print(f"[bold cyan][{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running automated trigger check across coiled watchlists...[/bold cyan]")
    
    actionable_alerts = []
    
    # Priority watchlists from both markets
    priority_tickers = list(TRIGGER_LEVELS.keys()) + ["^NSEI", "QQQ", "SPY"]
    
    for ticker in priority_tickers:
        try:
            signal, artifacts = team.analyze_asset(ticker)
            if signal is None:
                continue

            curr_price = signal.entry_price
            rvol = artifacts.get("snapshot").rvol if artifacts.get("snapshot") else 1.0
            
            curr_sym = "Rs." if (ticker.endswith(".NS") or ticker.endswith(".BO") or ticker.startswith("^NSE")) else "$"

            # Check 1: Did price cross the predefined breakout trigger?
            if ticker in TRIGGER_LEVELS:
                spec = TRIGGER_LEVELS[ticker]
                if curr_price >= spec["trigger_price"]:
                    alert_info = {
                        "ticker": ticker,
                        "type": "BREAKOUT_TRIGGER_HIT",
                        "current_price": f"{curr_sym}{curr_price:.2f}",
                        "trigger_price": f"{curr_sym}{spec['trigger_price']:.2f}",
                        "stop_loss": f"{curr_sym}{spec['stop_loss']:.2f}",
                        "target": f"{curr_sym}{spec['target']:.2f}",
                        "rvol": rvol,
                        "verdict": signal.verdict,
                        "summary": f"Price crossed above breakout level of {curr_sym}{spec['trigger_price']:.2f} on {rvol:.1f}x volume!"
                    }
                    actionable_alerts.append(alert_info)
                    _emit_alert(alert_info)
                    continue

            # Check 2: Did the agent team upgrade verdict to actionable BUY or SQUEEZE?
            if "BUY" in signal.verdict or "SQUEEZE" in signal.verdict:
                alert_info = {
                    "ticker": ticker,
                    "type": "AGENT_SIGNAL_ALERT",
                    "current_price": f"{curr_sym}{curr_price:.2f}",
                    "stop_loss": f"{curr_sym}{signal.stop_loss:.2f}",
                    "target": f"{curr_sym}{signal.target_2:.2f}",
                    "rvol": rvol,
                    "verdict": signal.verdict,
                    "summary": signal.synthesis_summary
                }
                actionable_alerts.append(alert_info)
                _emit_alert(alert_info)

        except Exception as e:
            console.print(f"[dim red]Error checking {ticker}: {e}[/dim red]")

    if not actionable_alerts:
        console.print("[dim green]All monitored assets remain within safe consolidation coiled boundaries. No premature entries.[/dim green]\n")
        
    return actionable_alerts

def _emit_alert(alert: dict):
    msg = f"""
[bold yellow]TRIGGER ALERT FIRED: {alert['ticker']}[/bold yellow]
• Type: [bold]{alert['type']}[/bold]
• Current Price: [bold green]{alert['current_price']}[/bold green]
• Stop-Loss: [bold red]{alert['stop_loss']}[/bold red]
• Target: [bold green]{alert['target']}[/bold green]
• Volume (RVol): {alert['rvol']:.2f}x
• Verdict: {alert['verdict']}
• Rationale: {alert['summary']}
"""
    console.print(Panel(msg, title=f"🚨 TRADE ALERT: {alert['ticker']} 🚨", border_style="bold red"))

    # Dispatch to Telegram if configured
    try:
        from core.telegram_notifier import send_telegram_message, format_telegram_alert
        tele_msg = format_telegram_alert(alert)
        sent = send_telegram_message(tele_msg)
        if sent:
            console.print(f"[bold green]✓ Telegram alert successfully sent to your phone for {alert['ticker']}[/bold green]")
        else:
            console.print("[dim yellow]Telegram not configured yet (set TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID in .env)[/dim yellow]")
    except Exception as e:
        console.print(f"[dim red]Telegram dispatch error: {e}[/dim red]")

if __name__ == "__main__":
    team = ShootupPredictionTeam()
    run_monitor_check(team)
