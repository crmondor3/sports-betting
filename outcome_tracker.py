"""
outcome_tracker.py — persist ML predictions and auto-settle game outcomes.

Creates four tables in sports_betting.db:
  ml_picks            — every ValueBet recommendation with full metadata
  ml_outcomes         — settled win/loss results linked to picks
  ml_strategy_params  — sport-level min_edge + kelly_frac that adapt over time
  ml_performance_log  — rolling metric snapshots for trend analysis

Settlement priority:
  1. Match by game_id (exact, from OddsAPI)
  2. Match by (home_team, away_team) normalised name (fuzzy fallback)
"""
from __future__ import annotations

import logging
import sqlite3
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_DB_PATH = Path("sports_betting.db")


class OutcomeTracker:
    """Persist picks and settle outcomes against completed game results."""

    def __init__(self, db_path: Path = _DB_PATH) -> None:
        self.db_path = db_path
        self._init_tables()

    # ── Schema ────────────────────────────────────────────────────────────────

    def _init_tables(self) -> None:
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS ml_picks (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    sport        TEXT NOT NULL,
                    game_id      TEXT NOT NULL,
                    home_team    TEXT,
                    away_team    TEXT,
                    market       TEXT,
                    side         TEXT,
                    dk_odds      REAL,
                    dk_line      REAL,
                    model_prob   REAL,
                    dk_implied   REAL,
                    edge_pct     REAL,
                    ev           REAL,
                    kelly_frac   REAL,
                    kelly_stake  REAL,
                    confidence   INTEGER,
                    game_time    TEXT,
                    created_at   TEXT DEFAULT (datetime('now')),
                    settled      INTEGER DEFAULT 0,
                    UNIQUE(sport, game_id, market, side)
                );

                CREATE TABLE IF NOT EXISTS ml_outcomes (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    pick_id      INTEGER REFERENCES ml_picks(id),
                    sport        TEXT,
                    game_id      TEXT,
                    market       TEXT,
                    side         TEXT,
                    won          INTEGER,
                    home_score   INTEGER,
                    away_score   INTEGER,
                    settled_at   TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS ml_strategy_params (
                    sport        TEXT PRIMARY KEY,
                    min_edge     REAL DEFAULT 0.03,
                    kelly_frac   REAL DEFAULT 0.25,
                    updated_at   TEXT DEFAULT (datetime('now'))
                );

                CREATE TABLE IF NOT EXISTS ml_performance_log (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    sport        TEXT,
                    market       TEXT,
                    n_bets       INTEGER,
                    roi          REAL,
                    win_rate     REAL,
                    avg_edge     REAL,
                    brier_score  REAL,
                    logged_at    TEXT DEFAULT (datetime('now'))
                );
            """)

    def _conn(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    # ── Persist picks ─────────────────────────────────────────────────────────

    def save_picks(self, bets: list) -> int:
        """Insert ValueBet list into ml_picks. Returns count of new rows."""
        if not bets:
            return 0
        saved = 0
        with self._conn() as conn:
            for b in bets:
                d = asdict(b)
                try:
                    conn.execute("""
                        INSERT OR IGNORE INTO ml_picks
                          (sport, game_id, home_team, away_team, market, side,
                           dk_odds, dk_line, model_prob, dk_implied, edge_pct,
                           ev, kelly_frac, kelly_stake, confidence, game_time)
                        VALUES
                          (:sport, :game_id, :home_team, :away_team, :market, :side,
                           :dk_odds, :dk_line, :model_prob, :dk_implied, :edge_pct,
                           :ev, :kelly_frac, :kelly_stake, :confidence, :game_time)
                    """, d)
                    saved += conn.execute("SELECT changes()").fetchone()[0]
                except Exception as exc:
                    logger.warning("Failed to save pick %s/%s/%s: %s",
                                   b.sport, b.game_id, b.market, exc)
            conn.commit()
        logger.info("Saved %d new picks to ml_picks.", saved)
        return saved

    # ── Settle from OddsAPI scores ────────────────────────────────────────────

    def settle_from_scores_api(self, sport: str, scores: list[dict]) -> int:
        """
        Settle picks using raw OddsAPI /scores response.

        Each score dict has:
          id, home_team, away_team, completed, scores=[{name, score}, ...]
        """
        completed: list[dict] = []
        for g in scores:
            if not g.get("completed"):
                continue
            home = g.get("home_team", "")
            away = g.get("away_team", "")
            score_list = g.get("scores") or []
            h_score = next(
                (s.get("score") for s in score_list if s.get("name") == home), None
            )
            a_score = next(
                (s.get("score") for s in score_list if s.get("name") == away), None
            )
            if h_score is None or a_score is None:
                continue
            try:
                completed.append({
                    "game_id":    g.get("id", ""),
                    "home_team":  home,
                    "away_team":  away,
                    "home_score": int(float(h_score)),
                    "away_score": int(float(a_score)),
                })
            except (ValueError, TypeError):
                continue

        return self.settle_outcomes(sport, completed)

    # ── Core settlement ───────────────────────────────────────────────────────

    def settle_outcomes(self, sport: str, completed_games: list[dict]) -> int:
        """
        Match unsettled picks against completed game dicts and record outcomes.

        Each game dict must have: home_team, away_team, home_score, away_score.
        Optional: game_id (used for priority matching before name fallback).

        Returns count of picks newly settled.
        """
        now_str = datetime.utcnow().isoformat()

        with self._conn() as conn:
            rows = conn.execute("""
                SELECT id, game_id, market, side, home_team, away_team, dk_line, game_time
                FROM ml_picks
                WHERE sport = ? AND settled = 0 AND game_time < ?
                ORDER BY game_time
            """, (sport, now_str)).fetchall()

        if not rows:
            logger.info("No unsettled picks for %s.", sport)
            return 0

        # Build two lookups: by game_id and by (norm_home, norm_away)
        by_id:   dict[str, dict] = {}
        by_name: dict[tuple[str, str], dict] = {}
        for g in completed_games:
            gid = g.get("game_id", "")
            if gid:
                by_id[gid] = g
            key = (_norm(g.get("home_team", "")), _norm(g.get("away_team", "")))
            by_name[key] = g

        settled_count = 0
        for row_id, game_id, market, side, home, away, dk_line, game_time in rows:
            game = (
                by_id.get(game_id)
                or by_name.get((_norm(home), _norm(away)))
                or _fuzzy_match(home, away, by_name)
            )
            if game is None:
                continue

            h_score = game.get("home_score")
            a_score = game.get("away_score")
            if h_score is None or a_score is None:
                continue

            won = _determine_win(market, side, int(h_score), int(a_score), dk_line)
            if won is None:
                continue

            with self._conn() as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO ml_outcomes
                      (pick_id, sport, game_id, market, side, won, home_score, away_score)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (row_id, sport, game_id, market, side, int(won),
                      int(h_score), int(a_score)))
                conn.execute(
                    "UPDATE ml_picks SET settled = 1 WHERE id = ?", (row_id,)
                )
                conn.commit()
            settled_count += 1

        logger.info("Settled %d picks for %s.", settled_count, sport)
        return settled_count

    # ── Query settled data ────────────────────────────────────────────────────

    def get_settled_df(self, sport: str, n_recent: int = 500) -> pd.DataFrame:
        """Return settled picks with outcomes as a DataFrame."""
        with self._conn() as conn:
            df = pd.read_sql_query(
                """
                SELECT p.*, o.won, o.home_score, o.away_score
                FROM ml_picks p
                JOIN ml_outcomes o ON o.pick_id = p.id
                WHERE p.sport = ?
                ORDER BY p.game_time DESC
                LIMIT ?
                """,
                conn,
                params=(sport, n_recent),
            )
        return df

    # ── Strategy params ───────────────────────────────────────────────────────

    def get_strategy_params(self, sport: str) -> dict:
        """Return current adapted strategy params for a sport."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT min_edge, kelly_frac FROM ml_strategy_params WHERE sport = ?",
                (sport,),
            ).fetchone()
        if row:
            return {"min_edge": row[0], "kelly_frac": row[1]}
        return {"min_edge": 0.03, "kelly_frac": 0.25}

    def save_strategy_params(
        self, sport: str, min_edge: float, kelly_frac: float
    ) -> None:
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ml_strategy_params (sport, min_edge, kelly_frac, updated_at)
                VALUES (?, ?, ?, datetime('now'))
                ON CONFLICT(sport) DO UPDATE SET
                  min_edge   = excluded.min_edge,
                  kelly_frac = excluded.kelly_frac,
                  updated_at = excluded.updated_at
                """,
                (sport, min_edge, kelly_frac),
            )
            conn.commit()

    # ── Performance logging ───────────────────────────────────────────────────

    def log_performance(
        self, sport: str, market: str, df: pd.DataFrame
    ) -> None:
        if df.empty or "won" not in df.columns:
            return
        won    = df["won"].astype(float)
        stakes = df["kelly_stake"].astype(float)
        odds   = df["dk_odds"].astype(float)
        pnls   = [_pnl(o, s, bool(w)) for o, s, w in zip(odds, stakes, won)]
        wagered = stakes.sum()
        roi    = (sum(pnls) / wagered * 100) if wagered > 0 else 0
        brier  = float(((df["model_prob"].astype(float) - won) ** 2).mean())
        avg_edge = float(df["edge_pct"].astype(float).mean()) if "edge_pct" in df.columns else 0

        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO ml_performance_log
                  (sport, market, n_bets, roi, win_rate, avg_edge, brier_score)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (sport, market, len(df), roi, float(won.mean()), avg_edge, brier),
            )
            conn.commit()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(s: str) -> str:
    return s.lower().strip() if s else ""


def _fuzzy_match(
    home: str, away: str, by_name: dict[tuple[str, str], dict]
) -> Optional[dict]:
    """Fall back to last-word (nickname) matching for team name differences."""
    home_nick = _norm(home).split()[-1] if home else ""
    away_nick = _norm(away).split()[-1] if away else ""
    for (lh, la), g in by_name.items():
        lh_nick = lh.split()[-1] if lh else ""
        la_nick = la.split()[-1] if la else ""
        if lh_nick == home_nick and la_nick == away_nick:
            return g
    return None


def _pnl(dk_odds: float, stake: float, won: bool) -> float:
    if won:
        net = dk_odds / 100.0 if dk_odds >= 0 else 100.0 / abs(dk_odds)
        return stake * net
    return -stake


def _determine_win(
    market: str,
    side: str,
    home_score: int,
    away_score: int,
    dk_line: Optional[float],
) -> Optional[bool]:
    if market == "h2h":
        if side == "home":
            return home_score > away_score
        if side == "away":
            return away_score > home_score

    elif market == "spreads":
        if dk_line is None:
            return None
        margin = home_score - away_score
        if side == "home":
            return (margin + dk_line) > 0
        return (-margin + dk_line) > 0

    elif market == "totals":
        if dk_line is None:
            return None
        total = home_score + away_score
        if side == "over":
            return total > dk_line
        return total < dk_line

    return None
