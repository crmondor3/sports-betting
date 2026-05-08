"""Database initialisation, session management, and CRUD helpers."""
from __future__ import annotations

import csv
import logging
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

import config
from tracker.models import Base

logger = logging.getLogger(__name__)
_BACKUP_PATH = Path(__file__).parent.parent / "cache" / "bets_backup.csv"

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(
            config.DATABASE_URL,
            connect_args={"check_same_thread": False},  # SQLite only
            echo=False,
        )
    return _engine


def init_db() -> None:
    """Create all tables if they don't exist, and run lightweight migrations."""
    Base.metadata.create_all(bind=get_engine())
    with get_engine().connect() as conn:
        # Add auto_tracked column if absent (idempotent — silently skips if exists)
        try:
            conn.execute(text("ALTER TABLE bets ADD COLUMN auto_tracked BOOLEAN DEFAULT 0"))
            conn.commit()
        except Exception:
            pass
        # Backfill: since manual logging is removed, every bet is a model pick.
        # Rows inserted before this column existed have auto_tracked=0/NULL — fix them.
        try:
            conn.execute(text(
                "UPDATE bets SET auto_tracked = 1 WHERE auto_tracked IS NULL OR auto_tracked = 0"
            ))
            conn.commit()
        except Exception:
            pass
    _restore_from_backup_if_empty()


def _restore_from_backup_if_empty() -> None:
    """If the bets table is empty and a CSV backup exists, repopulate from it."""
    if not _BACKUP_PATH.exists():
        return
    with get_engine().connect() as conn:
        count = conn.execute(text("SELECT COUNT(*) FROM bets")).scalar()
        if count and count > 0:
            return

    from tracker.models import Bet
    from datetime import datetime

    def _dt(v):
        if not v:
            return None
        try:
            return datetime.fromisoformat(v)
        except Exception:
            return None

    def _f(v):
        try:
            return float(v) if v not in ("", None) else None
        except Exception:
            return None

    def _b(v):
        return v in ("True", "1", "true", 1, True) if v not in ("", None) else None

    restored = 0
    try:
        with open(_BACKUP_PATH, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                with session_scope() as s:
                    bet = Bet(
                        sport=row.get("sport", ""),
                        game_id=row.get("game_id", ""),
                        home_team=row.get("home_team", ""),
                        away_team=row.get("away_team", ""),
                        commence_time=_dt(row.get("commence_time")),
                        bet_type=row.get("bet_type", "moneyline"),
                        side=row.get("side", "home"),
                        line=_f(row.get("line")),
                        bookmaker=row.get("bookmaker") or "draftkings",
                        odds=_f(row.get("odds")) or 0.0,
                        model_prob=_f(row.get("model_prob")) or 0.5,
                        implied_prob=_f(row.get("implied_prob")) or 0.5,
                        ev_pct=_f(row.get("ev_pct")) or 0.0,
                        kelly_frac=_f(row.get("kelly_frac")) or 0.0,
                        stake_kelly=_f(row.get("stake_kelly")) or 0.0,
                        stake_flat=_f(row.get("stake_flat")) or 0.0,
                        bankroll_at_bet=_f(row.get("bankroll_at_bet")) or 100.0,
                        settled=_b(row.get("settled")) or False,
                        won=_b(row.get("won")),
                        pnl_kelly=_f(row.get("pnl_kelly")),
                        pnl_flat=_f(row.get("pnl_flat")),
                        settled_at=_dt(row.get("settled_at")),
                        auto_tracked=True,
                    )
                    s.add(bet)
                restored += 1
        logger.info("Restored %d bets from backup CSV", restored)
    except Exception as exc:
        logger.warning("Backup restore failed: %s", exc)


def get_session_factory() -> sessionmaker:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal


@contextmanager
def session_scope() -> Generator[Session, None, None]:
    factory = get_session_factory()
    session: Session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
