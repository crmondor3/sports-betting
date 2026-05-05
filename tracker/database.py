"""Database initialisation, session management, and CRUD helpers."""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

import config
from tracker.models import Base

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
        try:
            conn.execute(text("ALTER TABLE bets ADD COLUMN auto_tracked BOOLEAN DEFAULT 0"))
            conn.commit()
        except Exception:
            pass  # Column already exists


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
