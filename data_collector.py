"""
data_collector.py — fetch, persist, and retrieve game data + DraftKings odds snapshots.

Builds on the existing data/odds_api.py and data/season_fetcher.py infrastructure.
Adds a dedicated odds_snapshots table keyed by (game_id, bookmaker, market, timestamp)
for line-movement tracking.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Callable, Optional

from dotenv import load_dotenv
from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
    UniqueConstraint, create_engine, select,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

import config

load_dotenv()
logger = logging.getLogger(__name__)

_DB_URL = config.DATABASE_URL
_engine = create_engine(_DB_URL, connect_args={"check_same_thread": False})
_Session = sessionmaker(bind=_engine)

# ── ORM models ────────────────────────────────────────────────────────────────


class _Base(DeclarativeBase):
    pass


class MLGame(_Base):
    """Historical game with final score — one row per game."""
    __tablename__ = "ml_games"
    __table_args__ = (UniqueConstraint("date_str", "home_team", "away_team", name="uq_ml_game"),)

    id         = Column(Integer, primary_key=True, autoincrement=True)
    game_id    = Column(String(64), nullable=True, index=True)  # Odds API ID if available
    sport      = Column(String(16), nullable=False)
    home_team  = Column(String(64), nullable=False)
    away_team  = Column(String(64), nullable=False)
    date_str   = Column(String(10), nullable=False)             # YYYY-MM-DD
    game_date  = Column(DateTime,   nullable=False)
    home_score = Column(Integer,    nullable=True)
    away_score = Column(Integer,    nullable=True)
    home_winner = Column(Boolean,   nullable=True)
    season     = Column(Integer,    nullable=True)
    created_at = Column(DateTime,   default=datetime.utcnow)


class MLOddsSnapshot(_Base):
    """
    One odds snapshot per (game_id, bookmaker, market, timestamp).
    Stores both opening and closing lines for line-movement analysis.
    """
    __tablename__ = "ml_odds_snapshots"

    id          = Column(Integer,  primary_key=True, autoincrement=True)
    game_id     = Column(String(64), nullable=False, index=True)
    bookmaker   = Column(String(32), nullable=False)
    market      = Column(String(16), nullable=False)  # h2h | spreads | totals
    home_team   = Column(String(64), nullable=True)
    away_team   = Column(String(64), nullable=True)

    # h2h prices
    home_price  = Column(Integer,  nullable=True)   # American odds
    away_price  = Column(Integer,  nullable=True)

    # spread prices + points
    home_point  = Column(Float,    nullable=True)
    away_point  = Column(Float,    nullable=True)
    home_spread_price = Column(Integer, nullable=True)
    away_spread_price = Column(Integer, nullable=True)

    # totals
    over_under  = Column(Float,    nullable=True)
    over_price  = Column(Integer,  nullable=True)
    under_price = Column(Integer,  nullable=True)

    timestamp   = Column(DateTime, nullable=False)
    is_opening  = Column(Boolean,  default=False)
    is_closing  = Column(Boolean,  default=False)
    # True if spread moved > 1.5 pts or ML moved > 10 cents in prob vs previous snapshot
    sharp_move  = Column(Boolean,  default=False)

    created_at  = Column(DateTime, default=datetime.utcnow)


def init_ml_tables() -> None:
    """Create ML-specific tables if they don't already exist."""
    _Base.metadata.create_all(_engine)
    logger.info("ML database tables initialised.")


# ── Utility: American odds → implied probability ───────────────────────────────

def american_to_implied(american: float) -> float:
    """Convert American odds to raw implied probability (includes vig)."""
    if american >= 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


# ── Utility: no-vig probability ───────────────────────────────────────────────

def no_vig(p1: float, p2: float) -> tuple[float, float]:
    """Remove vig from a two-outcome market, return (fair_p1, fair_p2)."""
    total = p1 + p2
    return p1 / total, p2 / total


# ── DataCollector ─────────────────────────────────────────────────────────────

class DataCollector:
    """
    Orchestrates data collection:
      1. Fetches upcoming DraftKings odds and persists snapshots.
      2. Fetches historical game results from ESPN.
      3. Optionally fetches historical odds from The Odds API history endpoint.
    """

    def __init__(self) -> None:
        init_ml_tables()

    # ── Odds ──────────────────────────────────────────────────────────────────

    def fetch_and_store_odds(
        self,
        sport: str,
        force_refresh: bool = False,
    ) -> list[MLOddsSnapshot]:
        """
        Pull current DraftKings odds for *sport* and persist a snapshot row
        for each (game_id, market).  Marks a snapshot as opening if it's the
        first one stored for that game+market combination.

        Returns the list of newly created snapshot ORM objects.
        """
        from data.odds_api import OddsAPIClient

        sport_key = config.SUPPORTED_SPORTS.get(sport.upper())
        if not sport_key:
            raise ValueError(f"Unknown sport '{sport}'. Supported: {list(config.SUPPORTED_SPORTS)}")

        with OddsAPIClient() as client:
            games = client.get_odds(sport_key, force_refresh=force_refresh)

        now = datetime.utcnow()
        snapshots: list[MLOddsSnapshot] = []

        with _Session() as session:
            for g in games:
                home, away = g.home_team, g.away_team

                for market_key in ("h2h", "spreads", "totals"):
                    outcomes = g.markets.get(market_key, [])
                    if not outcomes:
                        continue

                    snap = self._build_snapshot(g.game_id, market_key, home, away, outcomes, now)
                    if snap is None:
                        continue

                    # Check if this is the first snapshot for this game+market
                    existing_count = session.execute(
                        select(MLOddsSnapshot)
                        .where(MLOddsSnapshot.game_id == g.game_id)
                        .where(MLOddsSnapshot.bookmaker == "draftkings")
                        .where(MLOddsSnapshot.market == market_key)
                    ).scalars().first()

                    snap.is_opening = existing_count is None

                    # Detect sharp moves vs previous snapshot
                    snap.sharp_move = self._detect_sharp_move(session, snap)

                    session.add(snap)
                    snapshots.append(snap)

            session.commit()

        logger.info("Stored %d DraftKings odds snapshots for %s.", len(snapshots), sport)
        return snapshots

    def _build_snapshot(
        self,
        game_id: str,
        market_key: str,
        home: str,
        away: str,
        outcomes: list[dict],
        timestamp: datetime,
    ) -> Optional[MLOddsSnapshot]:
        snap = MLOddsSnapshot(
            game_id=game_id,
            bookmaker="draftkings",
            market=market_key,
            home_team=home,
            away_team=away,
            timestamp=timestamp,
        )

        if market_key == "h2h":
            h = next((o for o in outcomes if o.get("name") == home), None)
            a = next((o for o in outcomes if o.get("name") == away), None)
            if not (h and a):
                return None
            snap.home_price = int(h["price"])
            snap.away_price = int(a["price"])

        elif market_key == "spreads":
            h = next((o for o in outcomes if o.get("name") == home), None)
            a = next((o for o in outcomes if o.get("name") == away), None)
            if not (h and a):
                return None
            snap.home_point = float(h.get("point", 0))
            snap.away_point = float(a.get("point", 0))
            snap.home_spread_price = int(h["price"])
            snap.away_spread_price = int(a["price"])

        elif market_key == "totals":
            o = next((x for x in outcomes if x.get("name", "").lower() == "over"), None)
            u = next((x for x in outcomes if x.get("name", "").lower() == "under"), None)
            if not (o and u):
                return None
            snap.over_under = float(o.get("point", 0))
            snap.over_price  = int(o["price"])
            snap.under_price = int(u["price"])

        return snap

    def _detect_sharp_move(self, session: Session, snap: MLOddsSnapshot) -> bool:
        """Return True if this snapshot shows a significant line move vs the most recent prior snap."""
        prev = session.execute(
            select(MLOddsSnapshot)
            .where(MLOddsSnapshot.game_id == snap.game_id)
            .where(MLOddsSnapshot.bookmaker == snap.bookmaker)
            .where(MLOddsSnapshot.market == snap.market)
            .order_by(MLOddsSnapshot.timestamp.desc())
        ).scalars().first()

        if prev is None:
            return False

        if snap.market == "h2h" and snap.home_price and prev.home_price:
            p_now  = american_to_implied(snap.home_price)
            p_prev = american_to_implied(prev.home_price)
            return abs(p_now - p_prev) >= 0.10  # >= 10 cents in implied prob

        if snap.market == "spreads" and snap.home_point is not None and prev.home_point is not None:
            return abs(snap.home_point - prev.home_point) >= 1.5

        return False

    # ── Historical game results ────────────────────────────────────────────────

    def fetch_historical_games(
        self,
        sport: str,
        years_back: int = 3,
        progress_cb: Callable[[str], None] | None = None,
    ) -> list[dict]:
        """
        Pull historical games from ESPN (via existing season_fetcher) and upsert
        into ml_games table.  Returns list of raw game dicts.
        """
        from data.season_fetcher import fetch_all_historical_games

        games = fetch_all_historical_games(sport, years_back=years_back, progress_cb=progress_cb)
        self._upsert_games(sport, games)
        return games

    def _upsert_games(self, sport: str, games: list[dict]) -> None:
        with _Session() as session:
            for g in games:
                date_str = g["date"][:10]
                home_team = g["home_team"]
                away_team = g["away_team"]

                # Check existing
                existing = session.execute(
                    select(MLGame)
                    .where(MLGame.date_str == date_str)
                    .where(MLGame.home_team == home_team)
                    .where(MLGame.away_team == away_team)
                ).scalars().first()

                if existing:
                    continue

                try:
                    dt = datetime.fromisoformat(g["date"].rstrip("Z"))
                except ValueError:
                    dt = datetime.strptime(date_str, "%Y-%m-%d")

                row = MLGame(
                    sport=sport.upper(),
                    home_team=home_team,
                    away_team=away_team,
                    date_str=date_str,
                    game_date=dt,
                    home_score=g.get("home_score"),
                    away_score=g.get("away_score"),
                    home_winner=g.get("home_winner"),
                    season=dt.year,
                )
                session.add(row)

            session.commit()

        logger.info("Upserted %s historical %s games into ml_games.", len(games), sport)

    # ── Historical odds (Odds API history endpoint) ────────────────────────────

    def fetch_historical_odds(
        self,
        sport: str,
        date_iso: str,
        force: bool = False,
    ) -> list[dict]:
        """
        Fetch a historical odds snapshot from The Odds API's /odds-history endpoint.
        NOTE: availability varies by plan.  Results are cached locally.

        Args:
            date_iso: ISO-8601 timestamp, e.g. "2024-01-14T12:00:00Z"
            force:    bypass local cache
        Returns list of raw API dicts.
        """
        import requests

        sport_key = config.SUPPORTED_SPORTS.get(sport.upper(), sport)
        cache_file = Path("cache") / f"hist_odds_{sport_key}_{date_iso[:10]}.json"
        cache_file.parent.mkdir(exist_ok=True)

        if not force and cache_file.exists():
            with cache_file.open() as fh:
                logger.info("History odds from cache: %s", cache_file.name)
                return json.load(fh)

        url = f"{config.ODDS_API_BASE}/sports/{sport_key}/odds-history"
        params = {
            "apiKey":      config.ODDS_API_KEY,
            "regions":     "us",
            "markets":     "h2h,spreads,totals",
            "bookmakers":  "draftkings",
            "date":        date_iso,
            "oddsFormat":  "american",
        }

        for attempt in range(4):
            try:
                resp = requests.get(url, params=params, timeout=20)
                if resp.status_code == 200:
                    data = resp.json()
                    with cache_file.open("w") as fh:
                        json.dump(data, fh)
                    logger.info(
                        "Fetched %d historical %s odds for %s",
                        len(data.get("data", [])), sport, date_iso[:10],
                    )
                    return data.get("data", [])
                if resp.status_code == 429:
                    wait = 2 ** attempt * 30
                    logger.warning("Rate-limited on history endpoint; waiting %ds", wait)
                    time.sleep(wait)
                else:
                    logger.warning("History odds HTTP %s: %s", resp.status_code, resp.text[:200])
                    return []
            except requests.RequestException as exc:
                logger.error("History odds request error: %s", exc)
                if attempt == 3:
                    return []
                time.sleep(2 ** attempt)

        return []

    # ── Query helpers ─────────────────────────────────────────────────────────

    def get_stored_games(self, sport: str, min_season: int = 2021) -> list[dict]:
        """Return historical games from ml_games as plain dicts."""
        with _Session() as session:
            rows = session.execute(
                select(MLGame)
                .where(MLGame.sport == sport.upper())
                .where(MLGame.season >= min_season)
                .where(MLGame.home_winner.isnot(None))
                .order_by(MLGame.game_date)
            ).scalars().all()

        return [
            {
                "date":         r.date_str,
                "home_team":    r.home_team,
                "away_team":    r.away_team,
                "home_score":   r.home_score,
                "away_score":   r.away_score,
                "home_winner":  r.home_winner,
                "season":       r.season,
                "game_date":    r.game_date,
            }
            for r in rows
        ]

    def get_odds_snapshot(self, game_id: str, market: str = "h2h") -> Optional[MLOddsSnapshot]:
        """Return the most recent DraftKings snapshot for a game+market pair."""
        with _Session() as session:
            row = session.execute(
                select(MLOddsSnapshot)
                .where(MLOddsSnapshot.game_id == game_id)
                .where(MLOddsSnapshot.bookmaker == "draftkings")
                .where(MLOddsSnapshot.market == market)
                .order_by(MLOddsSnapshot.timestamp.desc())
            ).scalars().first()
        return row

    def get_line_movement(self, game_id: str, market: str = "h2h") -> list[MLOddsSnapshot]:
        """Return all snapshots for a game+market, ordered oldest → newest."""
        with _Session() as session:
            rows = session.execute(
                select(MLOddsSnapshot)
                .where(MLOddsSnapshot.game_id == game_id)
                .where(MLOddsSnapshot.bookmaker == "draftkings")
                .where(MLOddsSnapshot.market == market)
                .order_by(MLOddsSnapshot.timestamp)
            ).scalars().all()
        return list(rows)
