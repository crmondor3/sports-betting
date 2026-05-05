"""Bankroll tracker — log bets, settle results, compute ROI/CLV/hit-rate."""
from __future__ import annotations

import csv
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

from sqlalchemy import func

import config
from bets.edge_detector import EdgeBet
from models.ev_calculator import closing_line_value, american_to_decimal
from models.kelly_criterion import flat_stake
from tracker.database import session_scope, init_db
from tracker.models import Bet, BankrollSnapshot

logger = logging.getLogger(__name__)


class BankrollTracker:
    def __init__(
        self,
        starting_bankroll: float = config.STARTING_BANKROLL,
    ):
        self.starting_bankroll = starting_bankroll
        init_db()

    # ── Logging ────────────────────────────────────────────────────────────────

    def log_bet(self, edge: EdgeBet, bankroll: float | None = None) -> int:
        """Persist an EdgeBet. Returns the new Bet.id."""
        bankroll = bankroll or self.get_current_bankroll()
        flat = flat_stake(bankroll)
        with session_scope() as s:
            bet = Bet(
                sport=edge.sport,
                game_id=edge.game_id,
                home_team=edge.home_team,
                away_team=edge.away_team,
                commence_time=edge.commence_time,
                bet_type=edge.bet_type,
                side=edge.side,
                line=edge.line,
                bookmaker=edge.best_book,
                odds=edge.best_odds,
                model_prob=edge.model_prob,
                implied_prob=edge.implied_prob,
                ev_pct=edge.ev_pct,
                kelly_frac=edge.kelly_frac,
                stake_kelly=edge.recommended_stake,
                stake_flat=flat,
                bankroll_at_bet=bankroll,
                settled=False,
            )
            s.add(bet)
            s.flush()
            bet_id = bet.id
        logger.info("Logged bet #%d: %s %s @ %+.0f", bet_id, edge.bet_type, edge.side, edge.best_odds)
        return bet_id

    def settle_bet(
        self,
        bet_id: int,
        won: bool,
        closing_odds: float | None = None,
    ) -> None:
        """Mark a bet as settled and record P&L."""
        with session_scope() as s:
            bet: Bet | None = s.get(Bet, bet_id)
            if bet is None:
                raise ValueError(f"Bet #{bet_id} not found")
            if bet.settled:
                logger.warning("Bet #%d already settled", bet_id)
                return

            decimal = american_to_decimal(bet.odds)
            if won:
                pnl_kelly = bet.stake_kelly * (decimal - 1)
                pnl_flat = bet.stake_flat * (decimal - 1)
            else:
                pnl_kelly = -bet.stake_kelly
                pnl_flat = -bet.stake_flat

            bet.won = won
            bet.pnl_kelly = round(pnl_kelly, 2)
            bet.pnl_flat = round(pnl_flat, 2)
            bet.settled = True
            bet.settled_at = datetime.utcnow()

            if closing_odds is not None:
                bet.closing_odds = closing_odds
                # CLV: positive means we got better than closing
                bet.clv = closing_line_value(bet.odds, closing_odds) * 100

        logger.info("Settled bet #%d — won=%s P&L Kelly $%.2f", bet_id, won, pnl_kelly)

    # ── Bankroll query ─────────────────────────────────────────────────────────

    def get_current_bankroll(self, use_kelly: bool = True) -> float:
        """Current bankroll = starting + sum of settled P&L."""
        with session_scope() as s:
            total_pnl = s.query(
                func.sum(Bet.pnl_kelly if use_kelly else Bet.pnl_flat)
            ).filter(Bet.settled == True).scalar() or 0.0
        return round(self.starting_bankroll + total_pnl, 2)

    # ── Statistics ─────────────────────────────────────────────────────────────

    def get_stats(self) -> dict:
        with session_scope() as s:
            settled = s.query(Bet).filter(Bet.settled == True).all()
            open_count = s.query(Bet).filter(Bet.settled == False).count()

        total = len(settled)
        wins = sum(1 for b in settled if b.won)
        pnl_kelly = sum(b.pnl_kelly or 0 for b in settled)
        pnl_flat = sum(b.pnl_flat or 0 for b in settled)
        stakes_kelly = sum(b.stake_kelly for b in settled)
        stakes_flat = sum(b.stake_flat for b in settled)
        clv_bets = [b.clv for b in settled if b.clv is not None]

        return {
            "total_bets": total,
            "open_bets": open_count,
            "wins": wins,
            "losses": total - wins,
            "hit_rate": f"{wins / total:.1%}" if total else "—",
            "pnl_kelly": round(pnl_kelly, 2),
            "pnl_flat": round(pnl_flat, 2),
            "roi_kelly": f"{pnl_kelly / stakes_kelly:.2%}" if stakes_kelly else "—",
            "roi_flat": f"{pnl_flat / stakes_flat:.2%}" if stakes_flat else "—",
            "avg_clv": f"{sum(clv_bets) / len(clv_bets):.2f}%" if clv_bets else "—",
            "current_bankroll_kelly": self.get_current_bankroll(use_kelly=True),
            "current_bankroll_flat": self.get_current_bankroll(use_kelly=False),
            "starting_bankroll": self.starting_bankroll,
        }

    def snapshot(self) -> None:
        """Save a bankroll snapshot to the DB."""
        stats = self.get_stats()
        if "message" in stats:
            return
        with session_scope() as s:
            snap = BankrollSnapshot(
                bankroll_kelly=stats["current_bankroll_kelly"],
                bankroll_flat=stats["current_bankroll_flat"],
                total_bets=stats["total_bets"],
                wins=stats["wins"],
                losses=stats["losses"],
            )
            s.add(snap)

    # ── CSV export ─────────────────────────────────────────────────────────────

    def export_csv(self, path: str | Path = "bets_export.csv") -> Path:
        with session_scope() as s:
            bets = s.query(Bet).order_by(Bet.placed_at).all()

        out = Path(path)
        fieldnames = [
            "id", "placed_at", "sport", "home_team", "away_team", "bet_type", "side",
            "line", "bookmaker", "odds", "model_prob", "implied_prob", "ev_pct",
            "stake_kelly", "stake_flat", "won", "pnl_kelly", "pnl_flat", "clv",
        ]
        with out.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for b in bets:
                writer.writerow({k: getattr(b, k, "") for k in fieldnames})

        logger.info("Exported %d bets to %s", len(bets), out)
        return out

    def delete_bet(self, bet_id: int) -> None:
        """Permanently remove a bet record."""
        with session_scope() as s:
            bet: Bet | None = s.get(Bet, bet_id)
            if bet is None:
                raise ValueError(f"Bet #{bet_id} not found")
            s.delete(bet)
        logger.info("Deleted bet #%d", bet_id)

    # ── Pending bets ───────────────────────────────────────────────────────────

    def get_open_bets(self) -> list[Bet]:
        with session_scope() as s:
            return s.query(Bet).filter(Bet.settled == False).all()
