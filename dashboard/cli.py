"""Rich CLI dashboard — top bets, bankroll stats, live refresh."""
from __future__ import annotations

import time
from datetime import datetime
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich import box

from bets.edge_detector import EdgeBet

# legacy_windows=False forces VT100 on Windows 11, avoiding cp1252 encoding errors
console = Console(legacy_windows=False)

# ── Colour palette ─────────────────────────────────────────────────────────────
EV_HIGH = "bright_green"
EV_MED = "yellow"
EV_LOW = "white"
SHARP_COLOUR = "cyan"
SOFT_COLOUR = "magenta"


def _ev_colour(ev_pct: float) -> str:
    if ev_pct >= 8:
        return EV_HIGH
    if ev_pct >= 4:
        return EV_MED
    return EV_LOW


def _odds_str(american: float) -> str:
    return f"+{american:.0f}" if american >= 0 else f"{american:.0f}"


# ── Tables ─────────────────────────────────────────────────────────────────────

def build_bets_table(edges: list[EdgeBet], title: str = "Top Bets by EV") -> Table:
    t = Table(
        title=title,
        box=box.ROUNDED,
        show_lines=True,
        header_style="bold white on dark_blue",
        expand=True,
    )
    t.add_column("Sport", style="bold", width=6)
    t.add_column("Game", min_width=28)
    t.add_column("Commence", width=14)
    t.add_column("Type", width=12)
    t.add_column("Side", width=10)
    t.add_column("Line", width=6, justify="right")
    t.add_column("Book", width=14)
    t.add_column("Odds", width=7, justify="right")
    t.add_column("Model%", width=8, justify="right")
    t.add_column("Imp%", width=7, justify="right")
    t.add_column("EV%", width=8, justify="right")
    t.add_column("Stake", width=9, justify="right")

    for edge in edges:
        ev_col = _ev_colour(edge.ev_pct)
        book_col = SHARP_COLOUR if edge.is_sharp_book else SOFT_COLOUR
        t.add_row(
            edge.sport,
            f"{edge.away_team} @\n{edge.home_team}",
            edge.commence_time.strftime("%m/%d %H:%M"),
            edge.bet_type.replace("_", " "),
            edge.side.upper(),
            f"{edge.line:+g}" if edge.line is not None else "—",
            Text(edge.best_book, style=book_col),
            _odds_str(edge.best_odds),
            f"{edge.model_prob:.1%}",
            f"{edge.implied_prob:.1%}",
            Text(f"{edge.ev_pct:+.2f}%", style=ev_col),
            f"${edge.recommended_stake:.2f}",
        )
    return t


def build_stats_panel(stats: dict) -> Panel:
    if "message" in stats:
        return Panel(stats["message"], title="Bankroll Stats", border_style="yellow")

    lines = [
        f"[bold]Bankroll (Kelly)[/bold]:  ${stats.get('current_bankroll_kelly', 0):>10,.2f}",
        f"[bold]Bankroll (Flat)[/bold]:   ${stats.get('current_bankroll_flat', 0):>10,.2f}",
        "",
        f"Total Bets:  {stats.get('total_bets', 0):>6}",
        f"Wins:        {stats.get('wins', 0):>6}",
        f"Losses:      {stats.get('losses', 0):>6}",
        f"Hit Rate:    {stats.get('hit_rate', 'N/A'):>6}",
        "",
        f"ROI (Kelly): {stats.get('roi_kelly', 'N/A'):>8}",
        f"ROI (Flat):  {stats.get('roi_flat', 'N/A'):>8}",
        f"Avg CLV:     {stats.get('avg_clv', 'N/A'):>8}",
    ]
    return Panel("\n".join(lines), title="Bankroll Stats", border_style="green")


def build_header(requests_remaining: int | None = None) -> Panel:
    ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    rem = f"  |  API Requests Remaining: [cyan]{requests_remaining}[/cyan]" if requests_remaining is not None else ""
    return Panel(
        f"[bold bright_white]Sports Betting Model[/bold bright_white]  —  {ts}{rem}",
        style="bold on dark_blue",
        box=box.HORIZONTALS,
    )


def build_legend() -> str:
    return (
        "[cyan]* Sharp Book[/cyan]   "
        "[magenta]* Soft Book[/magenta]   "
        "[bright_green]* EV >= 8%[/bright_green]   "
        "[yellow]* EV 4-8%[/yellow]   "
        "[white]* EV 3-4%[/white]"
    )


# ── Full dashboard render ──────────────────────────────────────────────────────

def render_dashboard(
    edges: list[EdgeBet],
    stats: dict,
    requests_remaining: int | None = None,
) -> None:
    console.clear()
    console.print(build_header(requests_remaining))
    console.print(build_stats_panel(stats))
    console.print(build_legend())
    console.print()

    if not edges:
        console.print("[yellow]No qualifying edges found above EV threshold.[/yellow]")
    else:
        console.print(build_bets_table(edges, title=f"Top {len(edges)} Edges by EV%"))


def live_dashboard(
    edge_fn,    # callable() → list[EdgeBet]
    stats_fn,   # callable() → dict
    refresh_seconds: int = 60,
    requests_remaining_fn=None,
) -> None:
    """Run a continuously refreshing dashboard. Ctrl-C to exit."""
    console.print("[bold]Starting live dashboard[/bold] — press Ctrl-C to exit.\n")
    try:
        while True:
            edges = edge_fn()
            stats = stats_fn()
            rem = requests_remaining_fn() if requests_remaining_fn else None
            render_dashboard(edges, stats, rem)
            time.sleep(refresh_seconds)
    except KeyboardInterrupt:
        console.print("\n[yellow]Dashboard stopped.[/yellow]")
