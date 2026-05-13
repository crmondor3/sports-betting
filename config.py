"""
Central configuration.

Priority order for every value:
  1. Streamlit secrets  (st.secrets) — used on Streamlit Cloud
  2. Environment variable / .env     — used locally or in CLI scripts
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")


def _secret(key: str, default: str = "") -> str:
    """Return value from st.secrets (cloud) or os.getenv (local), in that order."""
    try:
        import streamlit as st
        val = st.secrets.get(key)
        if val is not None:
            return str(val)
    except Exception:
        pass
    return os.getenv(key, default)


# ── API Keys ──────────────────────────────────────────────────────────────────
ODDS_API_KEY: str = _secret("ODDS_API_KEY")
TELEGRAM_BOT_TOKEN: str = _secret("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID: str = _secret("TELEGRAM_CHAT_ID")
EMAIL_SENDER: str = _secret("EMAIL_SENDER")
EMAIL_PASSWORD: str = _secret("EMAIL_PASSWORD")
EMAIL_RECIPIENT: str = _secret("EMAIL_RECIPIENT")

# ── Bankroll ──────────────────────────────────────────────────────────────────
STARTING_BANKROLL: float = float(_secret("STARTING_BANKROLL", "1000.00"))
DEFAULT_KELLY_FRACTION: float = float(_secret("DEFAULT_KELLY_FRACTION", "0.25"))

# ── Model Thresholds ─────────────────────────────────────────────────────────
# Raised from 3% → 5%: fewer bets, only high-conviction edges
EV_THRESHOLD: float = float(_secret("EV_THRESHOLD", "0.05"))
MAX_KELLY: float = float(_secret("MAX_KELLY", "0.08"))
MIN_CONFIDENCE: int = int(_secret("MIN_CONFIDENCE", "55"))  # 0-100
MAX_PICKS_PER_DAY: int = int(_secret("MAX_PICKS_PER_DAY", "10"))

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = _secret("DATABASE_URL", "sqlite:///sports_betting.db")

# ── The Odds API ──────────────────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Primary sportsbook for bet placement
TARGET_BOOK = "draftkings"

# All US books fetched for consensus probability (same API credit cost as single book)
CONSENSUS_BOOKS = "draftkings,fanduel,betmgm,caesars,betrivers,williamhill_us"

# Minimum books needed to trust consensus over ESPN model
MIN_BOOKS_FOR_CONSENSUS = 3

SUPPORTED_SPORTS = {
    # ── American Football ────────────────────────────────────────────────────
    "NFL":        "americanfootball_nfl",
    "NCAAF":      "americanfootball_ncaaf",
    # ── Basketball ──────────────────────────────────────────────────────────
    "NBA":        "basketball_nba",
    "WNBA":       "basketball_wnba",
    "NCAAB":      "basketball_ncaab",
    # ── Baseball ────────────────────────────────────────────────────────────
    "MLB":        "baseball_mlb",
    # ── Ice Hockey ──────────────────────────────────────────────────────────
    "NHL":        "icehockey_nhl",
    # ── Soccer ──────────────────────────────────────────────────────────────
    "MLS":        "soccer_usa_mls",
    "EPL":        "soccer_epl",
    "LALIGA":     "soccer_spain_la_liga",
    "BUNDESLIGA": "soccer_germany_bundesliga",
    "SERIEA":     "soccer_italy_serie_a",
    "LIGUE1":     "soccer_france_ligue_one",
    "UCL":        "soccer_uefa_champs_league",
    "UEL":        "soccer_uefa_europa_league",
    "LIGAMX":     "soccer_mexico_ligamx",
    # ── Combat Sports ───────────────────────────────────────────────────────
    "MMA":        "mma_mixed_martial_arts",
    "BOXING":     "boxing_boxing",
}

# Sports with ESPN data available for full ML model training.
# All other sports use consensus-only mode (multi-book no-vig edge detection).
ESPN_SPORTS: frozenset[str] = frozenset({
    "NFL", "NCAAF", "NBA", "WNBA", "NCAAB", "MLB", "NHL",
})

# UI grouping for app.py and CLI --list-sports
SPORT_CATEGORIES: dict[str, list[str]] = {
    "American Football": ["NFL", "NCAAF"],
    "Basketball":        ["NBA", "WNBA", "NCAAB"],
    "Baseball":          ["MLB"],
    "Ice Hockey":        ["NHL"],
    "Soccer":            ["MLS", "EPL", "LALIGA", "BUNDESLIGA", "SERIEA", "LIGUE1", "UCL", "UEL", "LIGAMX"],
    "Combat Sports":     ["MMA", "BOXING"],
}

MARKETS = ["h2h", "spreads", "totals"]

# ── Kelly sizing ──────────────────────────────────────────────────────────────
KELLY_CAP = MAX_KELLY
