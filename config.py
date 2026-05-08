"""Central configuration loaded from .env."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent / ".env")

# ── API Keys ──────────────────────────────────────────────────────────────────
ODDS_API_KEY: str = os.getenv("ODDS_API_KEY", "")
TELEGRAM_BOT_TOKEN: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID: str = os.getenv("TELEGRAM_CHAT_ID", "")
EMAIL_SENDER: str = os.getenv("EMAIL_SENDER", "")
EMAIL_PASSWORD: str = os.getenv("EMAIL_PASSWORD", "")
EMAIL_RECIPIENT: str = os.getenv("EMAIL_RECIPIENT", "")

# ── Bankroll ──────────────────────────────────────────────────────────────────
STARTING_BANKROLL: float = float(os.getenv("STARTING_BANKROLL", "1000.00"))
DEFAULT_KELLY_FRACTION: float = float(os.getenv("DEFAULT_KELLY_FRACTION", "0.25"))

# ── Model Thresholds ─────────────────────────────────────────────────────────
# Raised from 3% → 5%: fewer bets, only high-conviction edges
EV_THRESHOLD: float = float(os.getenv("EV_THRESHOLD", "0.05"))
MAX_KELLY: float = float(os.getenv("MAX_KELLY", "0.08"))
MIN_CONFIDENCE: int = int(os.getenv("MIN_CONFIDENCE", "55"))  # 0-100
MAX_PICKS_PER_DAY: int = int(os.getenv("MAX_PICKS_PER_DAY", "10"))

# ── Database ──────────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv("DATABASE_URL", "sqlite:///sports_betting.db")

# ── The Odds API ──────────────────────────────────────────────────────────────
ODDS_API_BASE = "https://api.the-odds-api.com/v4"

# Primary sportsbook for bet placement
TARGET_BOOK = "draftkings"

# All US books fetched for consensus probability (same API credit cost as single book)
CONSENSUS_BOOKS = "draftkings,fanduel,betmgm,caesars,betrivers,williamhill_us"

# Minimum books needed to trust consensus over ESPN model
MIN_BOOKS_FOR_CONSENSUS = 3

SUPPORTED_SPORTS = {
    "NFL": "americanfootball_nfl",
    "NBA": "basketball_nba",
    "MLB": "baseball_mlb",
    "NHL": "icehockey_nhl",
}

MARKETS = ["h2h", "spreads", "totals"]

# ── Kelly sizing ──────────────────────────────────────────────────────────────
KELLY_CAP = MAX_KELLY
