"""
Tennis win probability model using Markov chain game/set/match mathematics.
No external dependencies beyond Python stdlib.
"""
from __future__ import annotations

from functools import lru_cache


SURFACE_ADJ: dict[str, float] = {
    "grass":  0.04,
    "clay":  -0.03,
    "hard":   0.00,
    "carpet": 0.02,
}

# Default serve win % when no data is available
_DEFAULT_SPW = 0.62


class TennisModel:
    """
    Predict head-to-head win probabilities from serve/return stats.

    All probabilities are fractional (0..1).
    """

    # ── Game level ─────────────────────────────────────────────────────────────

    @staticmethod
    def p_win_game(p: float) -> float:
        """
        Exact Markov-chain probability that the server wins a game given
        serve-point win probability `p`.

        Closed-form derivation:
          - Wins without deuce: sum_{k=0}^{2} C(3+k, k) * p^4 * q^k
            = p^4 * (1 + 4q + 10q^2)
          - P(reach deuce at 3-3): C(6,3) * p^3 * q^3 = 20 * p^3 * q^3
          - P(win from deuce, geometric series): p^2 / (p^2 + q^2)
        """
        p = max(0.001, min(0.999, p))
        q = 1.0 - p
        denom = p * p + q * q
        if denom <= 0:
            denom = 1e-9
        wins_before_deuce = p**4 * (1 + 4*q + 10*q**2)
        p_deuce = 20 * (p**3) * (q**3)
        p_win_from_deuce = (p * p) / denom
        return wins_before_deuce + p_deuce * p_win_from_deuce

    # ── Set level ──────────────────────────────────────────────────────────────

    @staticmethod
    def p_win_set(spw1: float, spw2: float) -> float:
        """
        P(player 1 wins a tiebreak set) via DP over game states (g1, g2) 0..7.

        Player 1 serves games 1, 3, 5, ... (0-indexed even games).
        At 6-6, a tiebreak is played.
        """
        spw1 = max(0.001, min(0.999, spw1))
        spw2 = max(0.001, min(0.999, spw2))

        pg1_serve   = TennisModel.p_win_game(spw1)  # P1 wins when serving
        pg1_return  = 1.0 - TennisModel.p_win_game(spw2)  # P1 wins when P2 serves

        # dp[g1][g2] = P(player1 wins set | currently at g1-g2)
        # We'll compute via memoised recursion.

        @lru_cache(maxsize=None)
        def dp(g1: int, g2: int, server: int) -> float:
            """server=1 means player1 is serving this game."""
            if g1 == 7:   # player1 won tiebreak
                return 1.0
            if g2 == 7:   # player2 won tiebreak
                return 0.0
            if g1 == 6 and g2 == 6:
                # Tiebreak — approximate with average serve probs
                avg_spw = (spw1 + spw2) / 2.0
                return TennisModel.p_win_game(avg_spw)

            if g1 == 6 and g2 < 5:
                return 1.0
            if g2 == 6 and g1 < 5:
                return 0.0

            # Normal game: who wins?
            if server == 1:
                p_p1_wins_game = pg1_serve
            else:
                p_p1_wins_game = pg1_return

            # Next game, server alternates
            next_server = 2 if server == 1 else 1

            return (
                p_p1_wins_game       * dp(g1 + 1, g2,     next_server)
                + (1 - p_p1_wins_game) * dp(g1,     g2 + 1, next_server)
            )

        result = dp(0, 0, 1)  # player1 serves first
        dp.cache_clear()
        return result

    # ── Match level ────────────────────────────────────────────────────────────

    @staticmethod
    def p_win_match(spw1: float, spw2: float, surface: str = "hard",
                    best_of: int = 3) -> float:
        """
        P(player1 wins match) given their serve-win percentages and surface.
        Applies surface adjustment then uses DP over sets.
        """
        adj = SURFACE_ADJ.get(surface.lower(), 0.0)
        spw1_adj = max(0.01, min(0.99, spw1 + adj))
        spw2_adj = max(0.01, min(0.99, spw2 + adj))

        p_set = TennisModel.p_win_set(spw1_adj, spw2_adj)

        sets_to_win = (best_of + 1) // 2  # 2 for BO3, 3 for BO5

        # DP over set scores (s1, s2)
        # dp_m[s1][s2] = P(p1 wins match from this set score)
        from functools import lru_cache as _lru

        @_lru(maxsize=None)
        def dp_match(s1: int, s2: int) -> float:
            if s1 == sets_to_win:
                return 1.0
            if s2 == sets_to_win:
                return 0.0
            return p_set * dp_match(s1 + 1, s2) + (1 - p_set) * dp_match(s1, s2 + 1)

        result = dp_match(0, 0)
        dp_match.cache_clear()
        return result

    # ── High-level predict ─────────────────────────────────────────────────────

    @staticmethod
    def predict(
        p1_stats: dict,
        p2_stats: dict,
        surface: str = "hard",
        best_of: int = 3,
    ) -> dict:
        """
        Produce a full prediction dict from player stats dicts.

        p1_stats / p2_stats: output of TennisDataClient.get_player_stats().
        """
        surface = surface.lower()
        adj = SURFACE_ADJ.get(surface, 0.0)

        def _spw(stats: dict) -> float:
            overall = stats.get("overall", {})
            spw = overall.get("avg_serve_win_pct")
            if spw is None:
                # Try surface-specific
                surf_rec = stats.get("surface_record", {}).get(surface, {})
                spw = surf_rec.get("serve_win_pct")
            return spw if spw is not None else _DEFAULT_SPW

        def _rpw(stats: dict) -> float:
            overall = stats.get("overall", {})
            rpw = overall.get("avg_return_win_pct")
            if rpw is None:
                surf_rec = stats.get("surface_record", {}).get(surface, {})
                rpw = surf_rec.get("return_win_pct")
            return rpw if rpw is not None else (1.0 - _DEFAULT_SPW)

        spw1 = _spw(p1_stats)
        spw2 = _spw(p2_stats)
        rpw1 = _rpw(p1_stats)
        rpw2 = _rpw(p2_stats)

        # Blend serve and return evidence:
        # Effective SPW for player 1 = avg(their own spw, 1 - opponent's rpw)
        eff_spw1 = (spw1 + (1.0 - rpw2)) / 2.0
        eff_spw2 = (spw2 + (1.0 - rpw1)) / 2.0

        spw1_adj = max(0.01, min(0.99, eff_spw1 + adj))
        spw2_adj = max(0.01, min(0.99, eff_spw2 + adj))

        p1_win = TennisModel.p_win_match(spw1_adj, spw2_adj, surface, best_of)
        p2_win = 1.0 - p1_win

        # Surface favours — compare surface-specific win rate if available
        def _surf_win_pct(stats: dict) -> float | None:
            sr = stats.get("surface_record", {}).get(surface, {})
            w = sr.get("wins", 0)
            l = sr.get("losses", 0)
            total = w + l
            return w / total if total >= 3 else None

        p1_swp = _surf_win_pct(p1_stats)
        p2_swp = _surf_win_pct(p2_stats)

        if p1_swp is not None and p2_swp is not None:
            diff = p1_swp - p2_swp
            if diff > 0.08:
                surface_favors = "player1"
            elif diff < -0.08:
                surface_favors = "player2"
            else:
                surface_favors = "neutral"
        else:
            surface_favors = "neutral"

        # Confidence: based on data quality
        dq1 = p1_stats.get("data_quality", "no_data")
        dq2 = p2_stats.get("data_quality", "no_data")
        quality_score = {"full": 100, "partial": 55, "no_data": 15}
        conf = (quality_score.get(dq1, 15) + quality_score.get(dq2, 15)) // 2

        return {
            "p1_win_prob": round(p1_win, 4),
            "p2_win_prob": round(p2_win, 4),
            "p1_serve_adj": round(spw1_adj, 4),
            "p2_serve_adj": round(spw2_adj, 4),
            "surface_favors": surface_favors,
            "model_confidence": conf,
        }
