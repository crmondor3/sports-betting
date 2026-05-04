"""Expected value calculations and odds conversion utilities."""
from __future__ import annotations

from dataclasses import dataclass


# ── Odds conversion ────────────────────────────────────────────────────────────

def american_to_decimal(american: float) -> float:
    """Convert American odds to decimal odds."""
    if american >= 100:
        return (american / 100) + 1
    return (100 / abs(american)) + 1


def decimal_to_american(decimal: float) -> float:
    """Convert decimal odds to American odds."""
    if decimal >= 2.0:
        return (decimal - 1) * 100
    return -100 / (decimal - 1)


def implied_probability(american: float) -> float:
    """
    True implied probability stripped of vig.
    Raw conversion only — use no_vig_probability() for vig removal.
    """
    if american >= 0:
        return 100 / (american + 100)
    return abs(american) / (abs(american) + 100)


def no_vig_probability(american_home: float, american_away: float) -> tuple[float, float]:
    """
    Remove the bookmaker's vig from a two-outcome market.
    Returns (p_home, p_away) that sum to 1.0.
    """
    raw_home = implied_probability(american_home)
    raw_away = implied_probability(american_away)
    total = raw_home + raw_away
    return raw_home / total, raw_away / total


def vig_percentage(american_home: float, american_away: float) -> float:
    """Bookmaker's overround as a percentage."""
    raw_home = implied_probability(american_home)
    raw_away = implied_probability(american_away)
    return (raw_home + raw_away - 1) * 100


# ── Expected Value ─────────────────────────────────────────────────────────────

def calculate_ev(model_prob: float, american_odds: float) -> float:
    """
    EV = (model_prob × net_payout) − (1 − model_prob)
    Expressed as a fraction of stake (e.g. 0.05 = +5% EV).
    """
    if not 0 < model_prob < 1:
        raise ValueError(f"model_prob must be in (0, 1), got {model_prob}")
    decimal = american_to_decimal(american_odds)
    net_payout = decimal - 1  # profit per unit staked
    return (model_prob * net_payout) - (1 - model_prob)


def ev_percentage(model_prob: float, american_odds: float) -> float:
    """EV as a percentage of stake."""
    return calculate_ev(model_prob, american_odds) * 100


# ── Best line finder ──────────────────────────────────────────────────────────

@dataclass
class BestLine:
    bookmaker: str
    american_odds: float
    decimal_odds: float
    implied_prob: float
    is_sharp: bool


def find_best_line(
    outcomes_by_book: dict[str, float],
    sharp_books: set[str],
    side: str,
) -> BestLine | None:
    """
    Given {bookmaker: american_odds} for one side, return the best available line.
    Prefers the highest decimal odds (most value to bettor).
    """
    best: BestLine | None = None
    for book, american in outcomes_by_book.items():
        decimal = american_to_decimal(american)
        candidate = BestLine(
            bookmaker=book,
            american_odds=american,
            decimal_odds=decimal,
            implied_prob=implied_probability(american),
            is_sharp=book.lower() in sharp_books,
        )
        if best is None or decimal > best.decimal_odds:
            best = candidate
    return best


# ── CLV calculation ────────────────────────────────────────────────────────────

def closing_line_value(open_american: float, close_american: float) -> float:
    """
    CLV = closing implied prob − opening implied prob.
    Positive CLV means you beat the closing line (good).
    """
    return implied_probability(close_american) - implied_probability(open_american)
