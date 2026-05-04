"""Edge detection — surfaces bets where model probability beats implied odds."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

import config
from data.odds_api import OddsGame
from models.ev_calculator import (
    american_to_decimal,
    calculate_ev,
    implied_probability,
    no_vig_probability,
)
from models.kelly_criterion import kelly_fraction, stake_amount

logger = logging.getLogger(__name__)

BetType = Literal["moneyline", "spread", "total_over", "total_under", "player_prop"]


@dataclass
class EdgeBet:
    sport: str
    game_id: str
    home_team: str
    away_team: str
    commence_time: datetime
    bet_type: BetType
    side: str                    # "home", "away", "over", "under"
    best_book: str
    best_odds: float             # American
    model_prob: float
    implied_prob: float
    ev_pct: float
    kelly_frac: float
    recommended_stake: float     # $ amount based on current bankroll
    is_sharp_book: bool
    # optional spread/total line
    line: float | None = None
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "sport": self.sport,
            "game": f"{self.away_team} @ {self.home_team}",
            "commence": self.commence_time.strftime("%Y-%m-%d %H:%M"),
            "bet_type": self.bet_type,
            "side": self.side,
            "line": self.line,
            "best_book": self.best_book,
            "best_odds": self.best_odds,
            "model_prob": f"{self.model_prob:.1%}",
            "implied_prob": f"{self.implied_prob:.1%}",
            "ev_pct": f"{self.ev_pct:+.2f}%",
            "kelly_frac": f"{self.kelly_frac:.2%}",
            "stake": f"${self.recommended_stake:.2f}",
            "sharp_book": self.is_sharp_book,
            "notes": self.notes,
        }


class EdgeDetector:
    def __init__(
        self,
        ev_threshold: float = config.EV_THRESHOLD,
        kelly_fraction_: float = config.DEFAULT_KELLY_FRACTION,
        bankroll: float = config.STARTING_BANKROLL,
        sharp_books: set[str] = frozenset(),
    ):
        self.ev_threshold = ev_threshold
        self.kelly_fraction = kelly_fraction_
        self.bankroll = bankroll
        self.sharp_books = sharp_books

    # ── Moneyline edges ────────────────────────────────────────────────────────

    def find_moneyline_edges(
        self,
        games: list[OddsGame],
        model_probs: dict[str, tuple[float, float]],  # game_id → (home_prob, away_prob)
    ) -> list[EdgeBet]:
        """
        model_probs: {game_id: (home_win_prob, away_win_prob)}
        Returns EdgeBet list filtered by EV threshold.
        """
        edges: list[EdgeBet] = []
        for game in games:
            probs = model_probs.get(game.game_id)
            if probs is None:
                continue
            home_prob, away_prob = probs

            # Gather all h2h lines
            home_lines: dict[str, float] = {}
            away_lines: dict[str, float] = {}
            for book, markets in game.bookmakers.items():
                for outcome in markets.get("h2h", []):
                    if outcome["name"] == game.home_team:
                        home_lines[book] = outcome["price"]
                    elif outcome["name"] == game.away_team:
                        away_lines[book] = outcome["price"]

            for side, model_p, lines in [
                ("home", home_prob, home_lines),
                ("away", away_prob, away_lines),
            ]:
                if not lines:
                    continue
                best_book = max(lines, key=lambda b: american_to_decimal(lines[b]))
                best_odds = lines[best_book]
                imp_prob = implied_probability(best_odds)
                ev = calculate_ev(model_p, best_odds) * 100

                if ev >= self.ev_threshold * 100:
                    edges.append(EdgeBet(
                        sport=game.sport,
                        game_id=game.game_id,
                        home_team=game.home_team,
                        away_team=game.away_team,
                        commence_time=game.commence_time,
                        bet_type="moneyline",
                        side=side,
                        best_book=best_book,
                        best_odds=best_odds,
                        model_prob=model_p,
                        implied_prob=imp_prob,
                        ev_pct=ev,
                        kelly_frac=kelly_fraction(model_p, best_odds, self.kelly_fraction),
                        recommended_stake=stake_amount(self.bankroll, model_p, best_odds, self.kelly_fraction),
                        is_sharp_book=best_book.lower() in self.sharp_books,
                    ))
        return sorted(edges, key=lambda e: e.ev_pct, reverse=True)

    # ── Totals edges ──────────────────────────────────────────────────────────

    def find_totals_edges(
        self,
        games: list[OddsGame],
        model_probs: dict[str, dict],  # game_id → {line: {"over": p, "under": p}}
    ) -> list[EdgeBet]:
        edges: list[EdgeBet] = []
        for game in games:
            game_preds = model_probs.get(game.game_id, {})
            if not game_preds:
                continue
            for book, markets in game.bookmakers.items():
                for outcome in markets.get("totals", []):
                    line = outcome.get("point")
                    if line is None:
                        continue
                    pred = game_preds.get(line)
                    if pred is None:
                        continue
                    name = outcome["name"].lower()  # "over" / "under"
                    if name not in ("over", "under"):
                        continue
                    model_p = pred.get(name, 0.0)
                    book_odds = outcome["price"]
                    imp_prob = implied_probability(book_odds)
                    ev = calculate_ev(model_p, book_odds) * 100

                    if ev >= self.ev_threshold * 100:
                        edges.append(EdgeBet(
                            sport=game.sport,
                            game_id=game.game_id,
                            home_team=game.home_team,
                            away_team=game.away_team,
                            commence_time=game.commence_time,
                            bet_type=f"total_{name}",  # type: ignore[arg-type]
                            side=name,
                            best_book=book,
                            best_odds=book_odds,
                            model_prob=model_p,
                            implied_prob=imp_prob,
                            ev_pct=ev,
                            kelly_frac=kelly_fraction(model_p, book_odds, self.kelly_fraction),
                            recommended_stake=stake_amount(self.bankroll, model_p, book_odds, self.kelly_fraction),
                            is_sharp_book=book.lower() in self.sharp_books,
                            line=line,
                        ))
        return sorted(edges, key=lambda e: e.ev_pct, reverse=True)

    # ── Spread edges ──────────────────────────────────────────────────────────

    def find_spread_edges(
        self,
        games: list[OddsGame],
        model_probs: dict[str, dict],  # game_id → {"home": p, "away": p}
    ) -> list[EdgeBet]:
        edges: list[EdgeBet] = []
        for game in games:
            preds = model_probs.get(game.game_id, {})
            if not preds:
                continue
            for book, markets in game.bookmakers.items():
                for outcome in markets.get("spreads", []):
                    name = outcome["name"]
                    line = outcome.get("point", 0)
                    if name == game.home_team:
                        side = "home"
                    elif name == game.away_team:
                        side = "away"
                    else:
                        continue
                    model_p = preds.get(side, 0.0)
                    book_odds = outcome["price"]
                    imp_prob = implied_probability(book_odds)
                    ev = calculate_ev(model_p, book_odds) * 100

                    if ev >= self.ev_threshold * 100:
                        edges.append(EdgeBet(
                            sport=game.sport,
                            game_id=game.game_id,
                            home_team=game.home_team,
                            away_team=game.away_team,
                            commence_time=game.commence_time,
                            bet_type="spread",
                            side=side,
                            best_book=book,
                            best_odds=book_odds,
                            model_prob=model_p,
                            implied_prob=imp_prob,
                            ev_pct=ev,
                            kelly_frac=kelly_fraction(model_p, book_odds, self.kelly_fraction),
                            recommended_stake=stake_amount(self.bankroll, model_p, book_odds, self.kelly_fraction),
                            is_sharp_book=book.lower() in self.sharp_books,
                            line=line,
                        ))
        return sorted(edges, key=lambda e: e.ev_pct, reverse=True)

    def find_all_edges(
        self,
        games: list[OddsGame],
        moneyline_probs: dict[str, tuple[float, float]],
        totals_probs: dict[str, dict],
        spread_probs: dict[str, dict],
    ) -> list[EdgeBet]:
        all_edges = (
            self.find_moneyline_edges(games, moneyline_probs)
            + self.find_totals_edges(games, totals_probs)
            + self.find_spread_edges(games, spread_probs)
        )
        return sorted(all_edges, key=lambda e: e.ev_pct, reverse=True)
