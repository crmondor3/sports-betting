"""
historical_backfill.py — seed calibration data from last N days of ESPN results.

For each completed game in the window, we run the current model (Elo + MatchupAnalyzer)
to produce home_win_prob, then record a synthetic settled moneyline bet.  This gives
the calibrator real signal immediately rather than waiting weeks for live bets to settle.

Synthetic bet assumptions:
  • bet_type = "moneyline", side = "home"  (one record per game, home perspective)
  • odds = -110 (DraftKings near-even lines)
  • implied_prob = 0.5238  (fair value of -110 with standard 4.5% vig)
  • stake_kelly = $1.00 fixed  (small so P&L doesn't distort the real bankroll)
  • bookmaker = "espn_historical"
  • auto_tracked = True

Deduplication: skips any (game_id, bet_type, side) already in the DB.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Callable

import config
from data.espn_api import ESPNClient
from models.elo_model import EloModel
from models.matchup_model import MatchupAnalyzer
from models.ev_calculator import calculate_ev, implied_probability
from models.kelly_criterion import kelly_fraction
from tracker.database import session_scope
from tracker.models import Bet

logger = logging.getLogger(__name__)

_SYNTH_ODDS     = -110.0
_SYNTH_STAKE    = 1.0
_SYNTH_BOOK     = "espn_historical"
_SYNTH_IMPLIED  = implied_probability(_SYNTH_ODDS)   # ≈ 0.5238
_SYNTH_DECIMAL  = 1 + 100 / 110                      # ≈ 1.9091


def run_backfill(
    days: int = 30,
    progress_cb: Callable[[str], None] | None = None,
) -> int:
    """
    Fetch ESPN results for the past `days` days, run the model on each game,
    and insert settled bet records.  Returns the count of new rows inserted.

    progress_cb: optional callable(message) for UI progress updates.
    """
    def _log(msg: str) -> None:
        logger.info(msg)
        if progress_cb:
            progress_cb(msg)

    espn   = ESPNClient()
    today  = date.today()
    dates  = [today - timedelta(days=i) for i in range(1, days + 1)]

    # Pre-load existing game_ids to avoid redundant work
    with session_scope() as s:
        existing = {
            (r.game_id, r.bet_type, r.side)
            for r in s.query(Bet.game_id, Bet.bet_type, Bet.side).all()
        }

    new_rows: list[Bet] = []

    for sport_label, sport_key in config.SUPPORTED_SPORTS.items():
        elo    = EloModel(sport=sport_label)
        # Seed Elo from trained model if available
        try:
            from models.trainer import load as _load_b
            _b = _load_b(sport_label)
            if _b and _b.get("final_elo"):
                elo._ratings = dict(_b["final_elo"])
        except Exception:
            pass
        analyzer = MatchupAnalyzer(elo_model=elo)
        inserted = 0

        for gdate in dates:
            games = espn.get_scoreboard(sport_label, gdate)
            for g in games:
                synthetic_id = f"hist_{g['event_id']}"
                key = (synthetic_id, "moneyline", "home")
                if key in existing:
                    continue

                home, away = g["home_team"], g["away_team"]
                if not home or not away:
                    continue

                try:
                    matchup = espn.get_matchup(sport_label, home, away)
                    pred    = analyzer.analyse(matchup)
                except Exception as exc:
                    logger.debug("Model failed %s vs %s: %s", home, away, exc)
                    continue

                home_prob = pred.home_win_prob
                away_prob = 1.0 - home_prob
                home_won  = bool(g["home_winner"])

                try:
                    ct = datetime.fromisoformat(g["date"].rstrip("Z"))
                except Exception:
                    ct = datetime.combine(gdate, datetime.min.time())

                # Record BOTH sides so the calibrator sees the full probability
                # distribution — not just the "looked good" side.
                for side, model_p, won in [
                    ("home", home_prob, home_won),
                    ("away", away_prob, not home_won),
                ]:
                    side_key = (synthetic_id, "moneyline", side)
                    if side_key in existing:
                        continue
                    pnl = round(_SYNTH_STAKE * (_SYNTH_DECIMAL - 1), 4) if won else -_SYNTH_STAKE
                    bet = Bet(
                        sport            = sport_label,
                        game_id          = synthetic_id,
                        home_team        = home,
                        away_team        = away,
                        commence_time    = ct,
                        bet_type         = "moneyline",
                        side             = side,
                        line             = None,
                        bookmaker        = _SYNTH_BOOK,
                        odds             = _SYNTH_ODDS,
                        model_prob       = round(model_p, 4),
                        implied_prob     = round(_SYNTH_IMPLIED, 4),
                        ev_pct           = round(calculate_ev(model_p, _SYNTH_ODDS) * 100, 2),
                        kelly_frac       = round(kelly_fraction(model_p, _SYNTH_ODDS, 0.25), 4),
                        stake_kelly      = _SYNTH_STAKE,
                        stake_flat       = _SYNTH_STAKE,
                        bankroll_at_bet  = 100.0,
                        settled          = True,
                        won              = won,
                        pnl_kelly        = pnl,
                        pnl_flat         = pnl,
                        settled_at       = ct,
                        auto_tracked     = True,
                    )
                    new_rows.append(bet)
                    existing.add(side_key)
                    inserted += 1

        _log(f"{sport_label}: {inserted} historical bets added")

    if new_rows:
        with session_scope() as s:
            for b in new_rows:
                s.add(b)
        _log(f"Backfill complete — {len(new_rows)} total rows inserted")
    else:
        _log("Backfill complete — no new rows (all already present)")

    espn.close()
    return len(new_rows)
