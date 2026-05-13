"""
run.py — full pipeline entry point.
Odds are fetched once per day (cached). ESPN data is also cached daily.
Focus: high-confidence, positive-EV DraftKings bets only.

Usage:
  python run.py                      # Run pipeline, print top picks
  python run.py --sport NBA          # Single sport
  python run.py --refresh            # Force-refresh today's odds cache
  python run.py --settle 42 W        # Settle bet #42 as a win
  python run.py --export             # Export bets to CSV
  python run.py --stats              # Print bankroll stats
"""
from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

_ET = ZoneInfo("America/New_York")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("sports_betting.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger("run")

import config
from data.odds_api import OddsAPIClient, OddsGame
from data.espn_api import ESPNClient, SPORT_MAP as _ESPN_SPORT_MAP
from data.daily_cache import bust_all
from models.elo_model import EloModel
from models.poisson_model import PoissonModel
from models.matchup_model import MatchupAnalyzer, MatchupPrediction
from models.ev_calculator import calculate_ev, implied_probability, no_vig_probability
from models.kelly_criterion import kelly_fraction, stake_amount
from tracker.bankroll import BankrollTracker
from tracker.database import init_db


# ── Output dataclass ──────────────────────────────────────────────────────────

@dataclass
class Pick:
    """A single actionable bet recommendation."""
    sport: str
    game_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    bet_type: str       # moneyline | spread | total_over | total_under
    side: str
    line: float | None
    dk_odds: float      # American odds on DraftKings
    model_prob: float   # blended probability (consensus + ESPN)
    implied_prob: float
    ev_pct: float
    kelly_frac: float
    stake: float        # $ amount
    confidence: int     # 0-100
    matchup: MatchupPrediction = field(repr=False)
    consensus_prob: float | None = None   # pure market consensus (no ESPN blend)
    n_books: int = 0                      # books contributing to consensus
    market_edge: float = 0.0             # consensus_prob - dk_implied (raw market edge)

    @property
    def game_label(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    @property
    def odds_str(self) -> str:
        return f"+{self.dk_odds:.0f}" if self.dk_odds >= 0 else f"{self.dk_odds:.0f}"

    def _bet_signals(self) -> list[str]:
        """
        Return signals oriented to the bet side, not the game winner.
        A signal that favours the home team is POSITIVE for a home bet
        but NEGATIVE for an away bet — makes the card self-consistent.
        """
        bet_dir = 1 if self.side in ("home", "over") else -1
        lines = []
        for s in self.matchup.signals:
            adjusted = s.direction * bet_dir
            if adjusted > 0:
                prefix = "+"
            elif adjusted < 0:
                prefix = "-"
            else:
                prefix = "~"
            lines.append(f"{prefix} {s.name}: {s.label}")
        return lines

    @property
    def bet_team(self) -> str:
        return self.home_team if self.side == "home" else self.away_team

    def to_dict(self) -> dict:
        edge_pct = round((self.model_prob - self.implied_prob) * 100, 1)
        ct_et = self.commence_time.replace(tzinfo=timezone.utc).astimezone(_ET)
        return {
            "sport":           self.sport,
            "game_id":         self.game_id,
            "home_team":       self.home_team,
            "away_team":       self.away_team,
            "commence_time_iso": self.commence_time.isoformat(),
            "game":            self.game_label,
            "commence":        ct_et.strftime("%m/%d ") + ct_et.strftime("%I:%M %p ET").lstrip("0"),
            "bet_type":        self.bet_type,
            "side":            self.side,
            "bet_team":        self.bet_team,
            "line":            self.line,
            "dk_odds":         self.dk_odds,
            "model_prob":      self.model_prob,
            "consensus_prob":  self.consensus_prob,
            "n_books":         self.n_books,
            "market_edge":     round(self.market_edge * 100, 2),  # as %
            "implied_prob":    self.implied_prob,
            "edge_pct":        edge_pct,
            "ev_pct":          self.ev_pct,
            "kelly_frac":      self.kelly_frac,
            "stake":           self.stake,
            "confidence":      self.confidence,
            "signals":         self._bet_signals(),
            "data_quality":    self.matchup.data_quality,
            "predicted_total": self.matchup.predicted_total,
        }


# ── Pipeline core ─────────────────────────────────────────────────────────────

def _score_pick(pick: Pick) -> float:
    """Combined score used to rank picks. Rewards EV × confidence."""
    return pick.ev_pct * (pick.confidence / 100)


def analyse_game(
    game: OddsGame,
    sport: str,
    elo: EloModel,
    poisson: PoissonModel,
    espn: ESPNClient,
    analyzer: MatchupAnalyzer,
    bankroll: float,
) -> list[Pick]:
    """Run all models on one game and return qualifying picks."""
    picks: list[Pick] = []

    # ── Matchup data (ESPN + Elo) ──────────────────────────────────────────────
    matchup_data = espn.get_matchup(sport, game.home_team, game.away_team)
    prediction = analyzer.analyse(matchup_data)

    espn_home_prob = prediction.home_win_prob
    espn_away_prob = prediction.away_win_prob

    # ── Blend: market consensus + ESPN model ──────────────────────────────────
    # For ESPN sports: 65% consensus + 35% ESPN form/H2H model.
    # For non-ESPN sports (soccer, MMA, etc.): 100% consensus — no ESPN data.
    cons_home = game.consensus_probs.get("home")
    cons_away = game.consensus_probs.get("away")
    n_books   = game.n_books
    cold_espn = matchup_data.data_quality == "cold"

    if cons_home and n_books >= config.MIN_BOOKS_FOR_CONSENSUS:
        weight = 0.0 if cold_espn else 0.35
        home_prob = (1 - weight) * cons_home + weight * espn_home_prob
        away_prob = 1.0 - home_prob
    else:
        home_prob = espn_home_prob
        away_prob = espn_away_prob
        cons_home = None   # mark as unavailable
        cons_away = None

    # ── Moneyline ─────────────────────────────────────────────────────────────
    home_ml = game.get_line("h2h", game.home_team)
    away_ml = game.get_line("h2h", game.away_team)

    if home_ml and away_ml:
        for side, model_p, dk_odds, cons_p in [
            ("home", home_prob, home_ml, cons_home),
            ("away", away_prob, away_ml, cons_away),
        ]:
            imp  = implied_probability(dk_odds)
            ev   = calculate_ev(model_p, dk_odds) * 100
            medge = game.market_edge(side, dk_odds)
            if ev >= config.EV_THRESHOLD * 100 and prediction.confidence >= config.MIN_CONFIDENCE:
                picks.append(Pick(
                    sport=sport, game_id=game.game_id,
                    home_team=game.home_team, away_team=game.away_team,
                    commence_time=game.commence_time,
                    bet_type="moneyline", side=side, line=None,
                    dk_odds=dk_odds, model_prob=model_p,
                    implied_prob=imp, ev_pct=ev,
                    kelly_frac=kelly_fraction(model_p, dk_odds),
                    stake=stake_amount(bankroll, model_p, dk_odds),
                    confidence=prediction.confidence,
                    matchup=prediction,
                    consensus_prob=cons_p,
                    n_books=n_books,
                    market_edge=medge,
                ))

    # ── Spreads ───────────────────────────────────────────────────────────────
    for side, team, model_p in [("home", game.home_team, home_prob), ("away", game.away_team, away_prob)]:
        spread_info = game.get_spread_line(team)
        if spread_info is None:
            continue
        point, dk_odds = spread_info
        imp  = implied_probability(dk_odds)
        ev   = calculate_ev(model_p, dk_odds) * 100
        medge = game.market_edge(side, dk_odds)
        if ev >= config.EV_THRESHOLD * 100 and prediction.confidence >= config.MIN_CONFIDENCE:
            picks.append(Pick(
                sport=sport, game_id=game.game_id,
                home_team=game.home_team, away_team=game.away_team,
                commence_time=game.commence_time,
                bet_type="spread", side=side, line=point,
                dk_odds=dk_odds, model_prob=model_p,
                implied_prob=imp, ev_pct=ev,
                kelly_frac=kelly_fraction(model_p, dk_odds),
                stake=stake_amount(bankroll, model_p, dk_odds),
                confidence=prediction.confidence,
                matchup=prediction,
                consensus_prob=None,
                n_books=n_books,
                market_edge=medge,
            ))

    # ── Totals ────────────────────────────────────────────────────────────────
    total_line = game.get_total_line()
    if total_line:
        cons_over  = game.consensus_probs.get("over")
        cons_under = game.consensus_probs.get("under")
        try:
            poisson_pred = poisson.predict(game.home_team, game.away_team, float(total_line))
        except Exception:
            poisson_pred = None

        if poisson_pred:
            for name, poisson_p, cons_p in [
                ("over",  poisson_pred.over_prob,  cons_over),
                ("under", poisson_pred.under_prob, cons_under),
            ]:
                # Blend Poisson with consensus totals when available
                if cons_p and n_books >= config.MIN_BOOKS_FOR_CONSENSUS:
                    model_p = 0.65 * cons_p + 0.35 * poisson_p
                else:
                    model_p = poisson_p
                    cons_p  = None

                dk_odds = game.get_line("totals", name.capitalize())
                if not dk_odds:
                    continue
                imp  = implied_probability(dk_odds)
                ev   = calculate_ev(model_p, dk_odds) * 100
                medge = game.market_edge(name, dk_odds)
                if ev >= config.EV_THRESHOLD * 100 and prediction.confidence >= config.MIN_CONFIDENCE:
                    picks.append(Pick(
                        sport=sport, game_id=game.game_id,
                        home_team=game.home_team, away_team=game.away_team,
                        commence_time=game.commence_time,
                        bet_type=f"total_{name}", side=name, line=float(total_line),
                        dk_odds=dk_odds, model_prob=model_p,
                        implied_prob=imp, ev_pct=ev,
                        kelly_frac=kelly_fraction(model_p, dk_odds),
                        stake=stake_amount(bankroll, model_p, dk_odds),
                        confidence=prediction.confidence,
                        matchup=prediction,
                        consensus_prob=cons_p,
                        n_books=n_books,
                        market_edge=medge,
                    ))

    return picks


def run_pipeline(
    sport_filter: str | None = None,
    force_refresh: bool = False,
) -> tuple[list[Pick], dict, int | None]:
    """
    Full daily pipeline.
    Returns (top_picks, bankroll_stats, api_requests_remaining).
    top_picks is capped at MAX_PICKS_PER_DAY, ranked by EV × confidence.
    """
    if not config.ODDS_API_KEY:
        logger.error("ODDS_API_KEY not set.")
        raise RuntimeError("ODDS_API_KEY is not set. Add it to .env or Streamlit secrets.")

    tracker = BankrollTracker()
    bankroll = tracker.get_current_bankroll()

    espn = ESPNClient()
    all_picks: list[Pick] = []
    api_remaining: int | None = None

    with OddsAPIClient() as client:
        sports_to_run = (
            {sport_filter.upper(): config.SUPPORTED_SPORTS[sport_filter.upper()]}
            if sport_filter
            else config.SUPPORTED_SPORTS
        )

        for label, sport_key in sports_to_run.items():
            games = client.get_odds(sport_key, force_refresh=force_refresh)
            if client.requests_remaining is not None:
                api_remaining = client.requests_remaining

            if not games:
                logger.info("%s: no DraftKings games today", label)
                continue

            # Load trained GB bundle — contains final Elo ratings from 3 seasons of data
            from models.trainer import load as load_bundle
            bundle = load_bundle(label)

            # Bootstrap Elo — seed from trained ratings if available (avoids cold-start)
            elo = EloModel(sport=label)
            if bundle and bundle.get("final_elo"):
                elo._ratings = dict(bundle["final_elo"])
                logger.info("%s: seeded Elo from %d trained ratings", label, len(elo._ratings))

            poisson = PoissonModel()

            # Try to seed Poisson from ESPN scoring averages (ESPN sports only)
            ppg_map: dict[str, dict] = {}
            if label in _ESPN_SPORT_MAP:
                for game in games:
                    for team in (game.home_team, game.away_team):
                        if team not in ppg_map:
                            form = espn.get_team_form(label, team)
                            if form.ppg > 0:
                                ppg_map[team] = {"pts_per_game": form.ppg}
                if ppg_map:
                    try:
                        poisson.fit_from_averages(ppg_map)
                    except Exception as exc:
                        logger.warning("Poisson fit failed for %s: %s", label, exc)

            analyzer = MatchupAnalyzer(elo_model=elo, trained_bundle=bundle)

            now_utc = datetime.now(timezone.utc)
            for game in games:
                # Skip games that have already started
                ct = game.commence_time
                if ct.tzinfo is None:
                    ct = ct.replace(tzinfo=timezone.utc)
                if ct <= now_utc:
                    logger.debug("Skipping started game: %s vs %s", game.home_team, game.away_team)
                    continue
                try:
                    picks = analyse_game(game, label, elo, poisson, espn, analyzer, bankroll)
                    all_picks.extend(picks)
                except Exception as exc:
                    logger.error("Error analysing %s vs %s: %s",
                                 game.home_team, game.away_team, exc)

    espn.close()

    # De-duplicate (same game + bet_type + side, keep best EV)
    seen: dict[str, Pick] = {}
    for p in all_picks:
        key = f"{p.game_id}_{p.bet_type}_{p.side}"
        if key not in seen or p.ev_pct > seen[key].ev_pct:
            seen[key] = p

    # Rank by EV × confidence, cap at MAX_PICKS_PER_DAY
    ranked = sorted(seen.values(), key=_score_pick, reverse=True)
    top_picks = ranked[:config.MAX_PICKS_PER_DAY]

    logger.info(
        "Pipeline complete: %d candidates -> %d top picks",
        len(seen), len(top_picks),
    )
    return top_picks, tracker.get_stats(), api_remaining


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Sports Betting Model — DraftKings")
    parser.add_argument("--sport", help="NFL | NBA | MLB | NHL")
    parser.add_argument("--refresh", action="store_true", help="Force-refresh today's odds cache")
    parser.add_argument("--settle", nargs=2, metavar=("BET_ID", "RESULT"))
    parser.add_argument("--export", action="store_true")
    parser.add_argument("--stats", action="store_true")
    args = parser.parse_args()

    init_db()
    tracker = BankrollTracker()

    if args.settle:
        bet_id, result = int(args.settle[0]), args.settle[1].upper()
        tracker.settle_bet(bet_id, won=(result == "W"))
        print(f"Bet #{bet_id} settled as {'WIN' if result == 'W' else 'LOSS'}")
        return

    if args.export:
        print(f"Exported to {tracker.export_csv()}")
        return

    if args.stats:
        from rich.pretty import pprint
        pprint(tracker.get_stats())
        return

    picks, stats, api_remaining = run_pipeline(args.sport, force_refresh=args.refresh)

    print(f"\n=== Top {len(picks)} Picks (DraftKings) ===\n")
    for i, p in enumerate(picks, 1):
        print(f"{i}. [{p.sport}] {p.game_label}")
        print(f"   {p.bet_type} -> {p.side.upper()}"
              + (f" {p.line:+g}" if p.line else "")
              + f"  {p.odds_str}")
        print(f"   EV: {p.ev_pct:+.2f}%  |  Model: {p.model_prob:.1%}"
              f"  |  Implied: {p.implied_prob:.1%}"
              f"  |  Confidence: {p.confidence}/100")
        print(f"   Stake (Kelly): ${p.stake:.2f}  |  {p.commence_time:%m/%d %H:%M}")
        for sig in p.matchup.signal_summary:
            print(f"   {sig}")
        print()


if __name__ == "__main__":
    main()
