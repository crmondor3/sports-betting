"""SQLAlchemy ORM models for bet tracking and tennis player database."""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    Boolean, Column, DateTime, Float, Integer, String, Text,
    UniqueConstraint, func,
)
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Bet(Base):
    __tablename__ = "bets"

    id = Column(Integer, primary_key=True, autoincrement=True)
    placed_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    sport = Column(String(10), nullable=False)
    game_id = Column(String(64), nullable=False)
    home_team = Column(String(64), nullable=False)
    away_team = Column(String(64), nullable=False)
    commence_time = Column(DateTime, nullable=True)
    bet_type = Column(String(32), nullable=False)   # moneyline / spread / total_over / total_under
    side = Column(String(32), nullable=False)        # home / away / over / under
    line = Column(Float, nullable=True)
    bookmaker = Column(String(32), nullable=False)
    odds = Column(Float, nullable=False)             # American
    model_prob = Column(Float, nullable=False)
    implied_prob = Column(Float, nullable=False)
    ev_pct = Column(Float, nullable=False)
    kelly_frac = Column(Float, nullable=False)

    # Sizing
    stake_kelly = Column(Float, nullable=False)      # Kelly-sized stake
    stake_flat = Column(Float, nullable=False)       # Flat stake
    bankroll_at_bet = Column(Float, nullable=False)

    # Settlement (filled after game)
    settled = Column(Boolean, default=False, nullable=False)
    won = Column(Boolean, nullable=True)
    pnl_kelly = Column(Float, nullable=True)         # Profit/loss on Kelly stake
    pnl_flat = Column(Float, nullable=True)
    closing_odds = Column(Float, nullable=True)      # For CLV calculation
    clv = Column(Float, nullable=True)               # closing_line_value
    settled_at = Column(DateTime, nullable=True)
    notes = Column(Text, nullable=True)

    def __repr__(self) -> str:
        return (
            f"<Bet id={self.id} {self.sport} {self.away_team}@{self.home_team} "
            f"{self.bet_type}/{self.side} {self.odds:+.0f} settled={self.settled}>"
        )


class TennisPlayer(Base):
    """Aggregated career stats for every ATP/WTA player from Sackmann data."""
    __tablename__ = "tennis_players"
    __table_args__ = (UniqueConstraint("name", "tour", name="uq_tennis_player"),)

    id                  = Column(Integer, primary_key=True, autoincrement=True)
    name                = Column(String(100), nullable=False)
    tour                = Column(String(10),  nullable=False)   # "atp" | "wta"
    ranking             = Column(Integer,  nullable=True)
    wins                = Column(Integer,  default=0)
    losses              = Column(Integer,  default=0)
    avg_serve_win_pct   = Column(Float,    nullable=True)
    avg_return_win_pct  = Column(Float,    nullable=True)
    avg_first_serve_pct = Column(Float,    nullable=True)
    avg_aces            = Column(Float,    nullable=True)
    avg_dfs             = Column(Float,    nullable=True)
    # JSON blobs (stored as TEXT)
    surface_stats       = Column(Text,     nullable=True)   # {hard:{wins,losses,spw,rpw},...}
    day_stats           = Column(Text,     nullable=True)   # {Monday:{wins,losses},...}
    first_set_stats     = Column(Text,     nullable=True)   # {won_fs_win_pct, lost_fs_win_pct, fs_win_rate}
    last_updated        = Column(DateTime, nullable=True)

    def __repr__(self) -> str:
        return f"<TennisPlayer {self.tour.upper()} {self.name} rank={self.ranking}>"


class BankrollSnapshot(Base):
    __tablename__ = "bankroll_snapshots"

    id = Column(Integer, primary_key=True, autoincrement=True)
    snapshot_at = Column(DateTime, default=datetime.utcnow, nullable=False)
    bankroll_kelly = Column(Float, nullable=False)
    bankroll_flat = Column(Float, nullable=False)
    total_bets = Column(Integer, nullable=False, default=0)
    wins = Column(Integer, nullable=False, default=0)
    losses = Column(Integer, nullable=False, default=0)
    roi_kelly = Column(Float, nullable=True)
    roi_flat = Column(Float, nullable=True)
    avg_clv = Column(Float, nullable=True)
