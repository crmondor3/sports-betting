"""
auto_settle.py — look up completed game scores and settle open bets automatically.
Called once at app startup. Uses The Odds API scores endpoint (cached daily).
"""
from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta

import config
from tracker.bankroll import BankrollTracker
from tracker.database import session_scope
from tracker.models import Bet

logger = logging.getLogger(__name__)


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", s.lower().strip()))


def _team_match(a: str, b: str) -> bool:
    an, bn = _norm(a), _norm(b)
    return an == bn or an.split()[-1] == bn.split()[-1]


def _determine_win(bet_type: str, side: str, line: float | None, score: dict) -> bool | None:
    """Return True/False if the bet won/lost, or None if undetermined."""
    scores_list = score.get("scores") or []
    home_team = score.get("home_team", "")
    away_team = score.get("away_team", "")

    hs_raw = next((s["score"] for s in scores_list if _team_match(s["name"], home_team)), None)
    aws_raw = next((s["score"] for s in scores_list if _team_match(s["name"], away_team)), None)

    if hs_raw is None or aws_raw is None:
        return None
    try:
        hs, aws = float(hs_raw), float(aws_raw)
    except (ValueError, TypeError):
        return None

    if bet_type == "moneyline":
        return (hs > aws) if side == "home" else (aws > hs)

    elif bet_type == "spread":
        pts = line or 0.0
        return ((hs + pts) > aws) if side == "home" else ((aws + pts) > hs)

    elif bet_type in ("total_over", "total_under"):
        total = hs + aws
        pts = line or 0.0
        return (total > pts) if bet_type == "total_over" else (total < pts)

    return None


def _snapshot_closing_lines(client) -> None:
    """
    For open bets whose game tip-off is within the next 3 hours,
    save the current DK line as a provisional closing line.
    Called before auto_settle so CLV is recorded even for games
    that have since been removed from DK's live feed.
    """
    from models.ev_calculator import closing_line_value

    window_lo = datetime.utcnow()
    window_hi = datetime.utcnow() + timedelta(hours=3)

    with session_scope() as s:
        pre_tip = (
            s.query(Bet)
            .filter(Bet.settled == False)
            .filter(Bet.closing_odds == None)
            .filter(Bet.commence_time >= window_lo)
            .filter(Bet.commence_time <= window_hi)
            .all()
        )
        if not pre_tip:
            return

        sports_needed = {
            (b.sport, config.SUPPORTED_SPORTS[b.sport])
            for b in pre_tip
            if b.sport in config.SUPPORTED_SPORTS
        }

    for sport_label, sport_key in sports_needed:
        try:
            games = client.get_odds(sport_key)
            odds_by_id: dict[str, dict] = {}
            for g in games:
                odds_by_id[g.game_id] = g
        except Exception as exc:
            logger.warning("closing-line snapshot failed %s: %s", sport_label, exc)
            continue

        with session_scope() as s:
            bets = (
                s.query(Bet)
                .filter(Bet.settled == False)
                .filter(Bet.closing_odds == None)
                .filter(Bet.sport == sport_label)
                .filter(Bet.commence_time >= window_lo)
                .filter(Bet.commence_time <= window_hi)
                .all()
            )
            for bet in bets:
                game = odds_by_id.get(bet.game_id)
                if not game:
                    continue
                closing: float | None = None
                if bet.bet_type == "moneyline":
                    team = bet.home_team if bet.side == "home" else bet.away_team
                    closing = game.get_line("h2h", team)
                elif bet.bet_type == "spread":
                    team = bet.home_team if bet.side == "home" else bet.away_team
                    info = game.get_spread_line(team)
                    closing = info[1] if info else None
                elif bet.bet_type in ("total_over", "total_under"):
                    name = "Over" if bet.bet_type == "total_over" else "Under"
                    closing = game.get_line("totals", name)

                if closing is not None:
                    bet.closing_odds = closing
                    bet.clv = round(closing_line_value(bet.odds, closing) * 100, 2)
                    logger.info(
                        "CLV snapshot bet #%d: open=%+.0f close=%+.0f CLV=%.2f%%",
                        bet.id, bet.odds, closing, bet.clv,
                    )


def auto_settle(tracker: BankrollTracker | None = None) -> int:
    """
    Fetch scores for sports with open bets and settle any completed games.
    Returns the count of newly settled bets.
    Only considers bets whose commence_time is at least 4 hours in the past.
    """
    if tracker is None:
        tracker = BankrollTracker()

    cutoff = datetime.utcnow() - timedelta(hours=4)

    with session_scope() as s:
        open_bets = (
            s.query(Bet)
            .filter(Bet.settled == False)
            .filter(Bet.commence_time <= cutoff)
            .all()
        )
        bet_rows = [
            {
                "id": b.id,
                "sport": b.sport,
                "game_id": b.game_id,
                "home_team": b.home_team,
                "away_team": b.away_team,
                "bet_type": b.bet_type,
                "side": b.side,
                "line": b.line,
            }
            for b in open_bets
        ]

    if not bet_rows:
        logger.info("auto_settle: no eligible open bets")
        return 0

    # Fetch scores for each sport that has open bets
    sports_needed = {
        (br["sport"], config.SUPPORTED_SPORTS[br["sport"]])
        for br in bet_rows
        if br["sport"] in config.SUPPORTED_SPORTS
    }

    from data.odds_api import OddsAPIClient
    scores_by_id: dict[str, dict] = {}

    with OddsAPIClient() as client:
        _snapshot_closing_lines(client)   # capture CLV before games leave DK feed

        for sport_label, sport_key in sports_needed:
            try:
                for sc in client.get_scores(sport_key, days_from=3):
                    if sc.get("completed"):
                        scores_by_id[sc["id"]] = sc
            except Exception as exc:
                logger.warning("auto_settle: scores fetch failed for %s: %s", sport_label, exc)

    settled = 0
    for br in bet_rows:
        score = scores_by_id.get(br["game_id"])
        if not score:
            logger.debug("auto_settle: no score for game_id=%s (%s @ %s)",
                         br["game_id"], br["away_team"], br["home_team"])
            continue

        won = _determine_win(br["bet_type"], br["side"], br["line"], score)
        if won is None:
            logger.debug("auto_settle: could not determine result for bet #%d", br["id"])
            continue

        try:
            tracker.settle_bet(br["id"], won=won)
            settled += 1
            logger.info("auto_settle: bet #%d settled as %s", br["id"], "WIN" if won else "LOSS")
        except Exception as exc:
            logger.warning("auto_settle: settle failed for bet #%d: %s", br["id"], exc)

    return settled
