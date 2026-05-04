"""Kelly Criterion bet sizing utilities."""
from __future__ import annotations

import config
from models.ev_calculator import american_to_decimal


def kelly_fraction(
    model_prob: float,
    american_odds: float,
    fractional: float = config.DEFAULT_KELLY_FRACTION,
) -> float:
    """
    Full Kelly: f* = (bp - q) / b
      b = decimal odds - 1 (net odds)
      p = model probability of winning
      q = 1 - p

    Returns fractional Kelly (default 25%) capped at config.KELLY_CAP.
    Negative values (no edge) are clamped to 0.
    """
    if not 0 < model_prob < 1:
        raise ValueError(f"model_prob must be in (0, 1), got {model_prob}")

    b = american_to_decimal(american_odds) - 1
    if b <= 0:
        return 0.0

    p = model_prob
    q = 1 - p

    full_kelly = (b * p - q) / b
    if full_kelly <= 0:
        return 0.0

    sized = full_kelly * fractional
    return min(sized, config.KELLY_CAP)


def stake_amount(
    bankroll: float,
    model_prob: float,
    american_odds: float,
    fractional: float = config.DEFAULT_KELLY_FRACTION,
) -> float:
    """Dollar stake = Kelly fraction × current bankroll, rounded to 2 dp."""
    fraction = kelly_fraction(model_prob, american_odds, fractional)
    return round(bankroll * fraction, 2)


def flat_stake(bankroll: float, pct: float = 0.02) -> float:
    """Simple flat-stake sizing: default 2% of bankroll."""
    return round(bankroll * pct, 2)


def expected_growth(
    model_prob: float,
    american_odds: float,
    fraction: float,
) -> float:
    """
    Log-growth rate per bet: p·ln(1 + b·f) + q·ln(1 - f)
    Useful for comparing sizing strategies.
    """
    import math
    b = american_to_decimal(american_odds) - 1
    p = model_prob
    q = 1 - p
    try:
        return p * math.log(1 + b * fraction) + q * math.log(1 - fraction)
    except ValueError:
        return float("-inf")
