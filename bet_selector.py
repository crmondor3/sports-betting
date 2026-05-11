"""
bet_selector.py — identify value bets against DraftKings and size them via Kelly.

For each upcoming game:
  1. Compute features using FeatureEngineer.build_prediction_row()
  2. Get ensemble win probability from ModelTrainer.predict()
  3. Compare vs DraftKings implied probability
  4. Flag bets where model edge > min_edge (default 3%)
  5. Calculate EV and Kelly stake
  6. Rank by EV descending
"""
from __future__ import annotations

import csv
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_MIN_EDGE    = 0.03   # 3% edge threshold
_DEFAULT_KELLY_FRAC  = 0.25
_MAX_BET_PCT         = 0.05   # cap single bet at 5% of bankroll
_PREDICTIONS_CSV     = Path("draftkings_predictions.csv")


# ── Data model ────────────────────────────────────────────────────────────────

@dataclass
class ValueBet:
    """A single value bet recommendation against DraftKings."""
    sport:        str
    game_id:      str
    home_team:    str
    away_team:    str
    market:       str          # h2h | spreads | totals
    side:         str          # home | away | over | under
    dk_odds:      float        # American odds on DraftKings
    dk_line:      Optional[float]   # spread point or total line
    model_prob:   float        # ensemble ML probability
    dk_implied:   float        # DK no-vig implied probability
    edge_pct:     float        # model_prob - dk_implied  (%)
    ev:           float        # expected value per unit staked
    kelly_frac:   float        # fractional Kelly percentage
    kelly_stake:  float        # $ stake given bankroll
    confidence:   int          # 0-100 from model
    game_time:    str = ""

    @property
    def matchup(self) -> str:
        return f"{self.away_team} @ {self.home_team}"

    @property
    def odds_str(self) -> str:
        return f"+{self.dk_odds:.0f}" if self.dk_odds >= 0 else f"{self.dk_odds:.0f}"


# ── BetSelector ───────────────────────────────────────────────────────────────

class BetSelector:
    """
    Identifies positive-EV bets against DraftKings lines using ML model output.

    Usage:
        selector = BetSelector(sport="NFL", bankroll=1000, kelly_fraction=0.25)
        bets = selector.analyze(games, historical_games)
        selector.print_top(bets, n=5)
        selector.save_csv(bets)
    """

    def __init__(
        self,
        sport:           str   = "NFL",
        bankroll:        float = 1000.0,
        kelly_fraction:  float = _DEFAULT_KELLY_FRAC,
        min_edge:        float = _DEFAULT_MIN_EDGE,
        markets:         list[str] | None = None,
    ) -> None:
        self.sport          = sport.upper()
        self.bankroll       = bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge       = min_edge
        self.markets        = markets or ["h2h", "spreads", "totals"]

    # ── Main analysis entry point ─────────────────────────────────────────────

    def analyze(
        self,
        odds_games: list[Any],          # list[OddsGame] from OddsAPIClient
        historical_games: list[dict],   # chronological game history for features
    ) -> list[ValueBet]:
        """
        For each upcoming game in *odds_games*, compute features, run the ML model,
        and return all bets that exceed the edge threshold.

        Bets are sorted by EV descending.
        """
        from feature_engineer import FeatureEngineer
        from model_trainer import ModelTrainer

        fe      = FeatureEngineer(sport=self.sport)
        trainer = ModelTrainer(sport=self.sport)

        if not trainer.is_trained("h2h"):
            logger.warning("No trained h2h model for %s. Run main.py --train first.", self.sport)

        bets: list[ValueBet] = []

        for game in odds_games:
            game_id  = game.game_id
            home     = game.home_team
            away     = game.away_team
            game_str = getattr(game, "commence_time", datetime.utcnow()).isoformat()

            # Build feature row for this game
            snap_dict = self._build_snap_dict(game)
            features  = fe.build_prediction_row(
                home=home,
                away=away,
                historical_games=historical_games,
                game_id=game_id,
                odds_snapshot=snap_dict,
            )

            # ── H2H (moneyline) ────────────────────────────────────────────
            if "h2h" in self.markets and trainer.is_trained("h2h"):
                bets += self._eval_h2h(trainer, features, game, game_str)

            # ── Spreads ────────────────────────────────────────────────────
            if "spreads" in self.markets and trainer.is_trained("spreads"):
                bets += self._eval_spreads(trainer, features, game, game_str)

            # ── Totals ─────────────────────────────────────────────────────
            if "totals" in self.markets and trainer.is_trained("totals"):
                bets += self._eval_totals(trainer, features, game, game_str)

        bets.sort(key=lambda b: b.ev, reverse=True)
        return bets

    # ── Market evaluators ─────────────────────────────────────────────────────

    def _eval_h2h(
        self,
        trainer: Any,
        features: dict,
        game: Any,
        game_str: str,
    ) -> list[ValueBet]:
        bets: list[ValueBet] = []
        home_odds = game.get_line("h2h", game.home_team)
        away_odds = game.get_line("h2h", game.away_team)
        if not (home_odds and away_odds):
            return bets

        result   = trainer.predict(features, market="h2h")
        home_prob = result["ensemble_prob"]
        away_prob = 1.0 - home_prob
        conf      = result["confidence"]

        dk_home_impl, dk_away_impl = _no_vig(home_odds, away_odds)

        for side, model_p, dk_impl, dk_odds in [
            ("home", home_prob, dk_home_impl, home_odds),
            ("away", away_prob, dk_away_impl, away_odds),
        ]:
            edge = model_p - dk_impl
            if edge < self.min_edge:
                continue
            ev    = _ev(model_p, dk_odds)
            kf    = _kelly(model_p, dk_odds, self.kelly_fraction, _MAX_BET_PCT)
            stake = round(self.bankroll * kf, 2)

            bets.append(ValueBet(
                sport=self.sport, game_id=game.game_id,
                home_team=game.home_team, away_team=game.away_team,
                market="h2h", side=side, dk_odds=dk_odds, dk_line=None,
                model_prob=round(model_p, 4), dk_implied=round(dk_impl, 4),
                edge_pct=round(edge * 100, 2), ev=round(ev * 100, 2),
                kelly_frac=round(kf * 100, 2), kelly_stake=stake,
                confidence=conf, game_time=game_str,
            ))
        return bets

    def _eval_spreads(
        self,
        trainer: Any,
        features: dict,
        game: Any,
        game_str: str,
    ) -> list[ValueBet]:
        bets: list[ValueBet] = []
        spread_info = game.get_spread_line(game.home_team)
        if not spread_info:
            return bets
        spread_pt, home_spread_price = spread_info
        away_spread_info = game.get_spread_line(game.away_team)
        if not away_spread_info:
            return bets
        _, away_spread_price = away_spread_info

        result   = trainer.predict(features, market="spreads")
        home_cov = result["ensemble_prob"]
        away_cov = 1.0 - home_cov
        conf     = result["confidence"]

        dk_home_impl, dk_away_impl = _no_vig(home_spread_price, away_spread_price)

        for side, model_p, dk_impl, dk_odds in [
            ("home", home_cov, dk_home_impl, home_spread_price),
            ("away", away_cov, dk_away_impl, away_spread_price),
        ]:
            edge = model_p - dk_impl
            if edge < self.min_edge:
                continue
            ev    = _ev(model_p, dk_odds)
            kf    = _kelly(model_p, dk_odds, self.kelly_fraction, _MAX_BET_PCT)
            stake = round(self.bankroll * kf, 2)

            bets.append(ValueBet(
                sport=self.sport, game_id=game.game_id,
                home_team=game.home_team, away_team=game.away_team,
                market="spreads", side=side, dk_odds=dk_odds,
                dk_line=spread_pt if side == "home" else -spread_pt,
                model_prob=round(model_p, 4), dk_implied=round(dk_impl, 4),
                edge_pct=round(edge * 100, 2), ev=round(ev * 100, 2),
                kelly_frac=round(kf * 100, 2), kelly_stake=stake,
                confidence=conf, game_time=game_str,
            ))
        return bets

    def _eval_totals(
        self,
        trainer: Any,
        features: dict,
        game: Any,
        game_str: str,
    ) -> list[ValueBet]:
        bets: list[ValueBet] = []
        total_pt = game.get_total_line()
        over_odds  = game.get_line("totals", "Over")
        under_odds = game.get_line("totals", "Under")
        if not (total_pt and over_odds and under_odds):
            return bets

        result   = trainer.predict(features, market="totals")
        over_p   = result["ensemble_prob"]
        under_p  = 1.0 - over_p
        conf     = result["confidence"]

        dk_over_impl, dk_under_impl = _no_vig(over_odds, under_odds)

        for side, model_p, dk_impl, dk_odds in [
            ("over",  over_p,  dk_over_impl,  over_odds),
            ("under", under_p, dk_under_impl, under_odds),
        ]:
            edge = model_p - dk_impl
            if edge < self.min_edge:
                continue
            ev    = _ev(model_p, dk_odds)
            kf    = _kelly(model_p, dk_odds, self.kelly_fraction, _MAX_BET_PCT)
            stake = round(self.bankroll * kf, 2)

            bets.append(ValueBet(
                sport=self.sport, game_id=game.game_id,
                home_team=game.home_team, away_team=game.away_team,
                market="totals", side=side, dk_odds=dk_odds, dk_line=total_pt,
                model_prob=round(model_p, 4), dk_implied=round(dk_impl, 4),
                edge_pct=round(edge * 100, 2), ev=round(ev * 100, 2),
                kelly_frac=round(kf * 100, 2), kelly_stake=stake,
                confidence=conf, game_time=game_str,
            ))
        return bets

    # ── Output ────────────────────────────────────────────────────────────────

    def print_top(self, bets: list[ValueBet], n: int = 5) -> None:
        """Print the top-N value bets in a formatted table."""
        top = [b for b in bets if b.ev > 0][:n]
        if not top:
            print("No qualifying value bets found today.")
            return

        header = (
            f"{'Matchup':<35} {'Market':<8} {'Side':<6} {'DK Odds':<9} "
            f"{'Mdl%':>6} {'DK%':>6} {'Edge%':>7} {'EV%':>7} "
            f"{'Kelly%':>7} {'Stake $':>8} {'Conf':>5}"
        )
        sep = "-" * len(header)
        print(f"\n{'=' * len(header)}")
        print(f"  TOP {n} DRAFTKINGS VALUE BETS — {self.sport}")
        print(f"{'=' * len(header)}")
        print(header)
        print(sep)
        for b in top:
            line_str = f" {b.dk_line:+.1f}" if b.dk_line is not None else ""
            print(
                f"{b.matchup:<35} {b.market:<8} {b.side:<6} "
                f"{b.odds_str + line_str:<9} "
                f"{b.model_prob*100:>5.1f}% {b.dk_implied*100:>5.1f}% "
                f"{b.edge_pct:>6.2f}% {b.ev:>6.2f}% "
                f"{b.kelly_frac:>6.2f}% {b.kelly_stake:>8.2f} "
                f"{b.confidence:>5}"
            )
        print(sep)

    def save_csv(self, bets: list[ValueBet], path: Path = _PREDICTIONS_CSV) -> None:
        """Save all bets (not just top-N) to CSV and persist to DB for outcome tracking."""
        if not bets:
            logger.info("No bets to save.")
            return
        with path.open("w", newline="") as fh:
            writer = csv.DictWriter(fh, fieldnames=asdict(bets[0]).keys())
            writer.writeheader()
            for b in bets:
                writer.writerow(asdict(b))
        logger.info("Saved %d predictions to %s", len(bets), path)

        try:
            from outcome_tracker import OutcomeTracker
            OutcomeTracker().save_picks(bets)
        except Exception as exc:
            logger.debug("Outcome tracker save skipped: %s", exc)

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _build_snap_dict(game: Any) -> dict:
        """Build the odds_snapshot dict format expected by FeatureEngineer._odds_features()."""
        snap: dict = {}
        for mkt, out_list in game.markets.items():
            snap[mkt] = {}
            home = game.home_team
            if mkt == "h2h":
                h = next((o for o in out_list if o.get("name") == home), None)
                a = next((o for o in out_list if o.get("name") != home), None)
                if h:
                    snap["h2h"]["home_price"] = int(h["price"])
                if a:
                    snap["h2h"]["away_price"] = int(a["price"])
            elif mkt == "spreads":
                h = next((o for o in out_list if o.get("name") == home), None)
                a = next((o for o in out_list if o.get("name") != home), None)
                if h:
                    snap["spreads"]["home_point"] = float(h.get("point", 0))
                    snap["spreads"]["home_price"] = int(h["price"])
                if a:
                    snap["spreads"]["away_point"] = float(a.get("point", 0))
                    snap["spreads"]["away_price"] = int(a["price"])
            elif mkt == "totals":
                o = next((x for x in out_list if x.get("name", "").lower() == "over"), None)
                u = next((x for x in out_list if x.get("name", "").lower() == "under"), None)
                if o:
                    snap["totals"]["over_under"] = float(o.get("point", 0))
                    snap["totals"]["over_price"]  = int(o["price"])
                if u:
                    snap["totals"]["under_price"] = int(u["price"])
        return snap


# ── Standalone math ───────────────────────────────────────────────────────────

def _impl_prob(american: float) -> float:
    """Raw implied probability (vig-inclusive)."""
    if american >= 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


def _no_vig(p1_american: float, p2_american: float) -> tuple[float, float]:
    """Return (fair_p1, fair_p2) with vig removed."""
    r1, r2 = _impl_prob(p1_american), _impl_prob(p2_american)
    t = r1 + r2
    return r1 / t, r2 / t


def _ev(model_prob: float, american_odds: float) -> float:
    """Expected value as fraction of stake: EV = (p × net_return) - (1-p)."""
    if american_odds >= 0:
        net = american_odds / 100.0
    else:
        net = 100.0 / abs(american_odds)
    return model_prob * net - (1 - model_prob)


def _kelly(
    model_prob: float,
    american_odds: float,
    fraction: float = 0.25,
    cap: float = 0.05,
) -> float:
    """Fractional Kelly stake as fraction of bankroll, capped at *cap*."""
    if american_odds >= 0:
        b = american_odds / 100.0
    else:
        b = 100.0 / abs(american_odds)
    if b <= 0:
        return 0.0
    full_kelly = (b * model_prob - (1 - model_prob)) / b
    if full_kelly <= 0:
        return 0.0
    return min(full_kelly * fraction, cap)
