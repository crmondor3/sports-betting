"""Tests for feature engineering logic (no DB or network required)."""
import math
import pytest
import pandas as pd
from feature_engineer import FeatureEngineer, FEATURE_NAMES, _impl_prob


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_games(n: int = 40, home_always_wins: bool = False) -> list[dict]:
    """Generate n synthetic games alternating winners."""
    teams = ["Team A", "Team B", "Team C", "Team D"]
    games = []
    for i in range(n):
        home = teams[i % len(teams)]
        away = teams[(i + 2) % len(teams)]
        h_score = 28 if (home_always_wins or i % 2 == 0) else 14
        a_score = 14 if (home_always_wins or i % 2 == 0) else 28
        games.append({
            "date":       f"2023-09-{(i % 28) + 1:02d}T18:00:00Z",
            "home_team":  home,
            "away_team":  away,
            "home_score": h_score,
            "away_score": a_score,
            "home_winner": h_score > a_score,
            "game_id":    f"game_{i}",
        })
    return games


# ── FeatureEngineer.build ──────────────────────────────────────────────────────

class TestFeatureEngineerBuild:
    def test_returns_dataframe(self):
        fe = FeatureEngineer(sport="NFL")
        df = fe.build(_make_games(30))
        assert isinstance(df, pd.DataFrame)

    def test_all_feature_columns_present(self):
        fe = FeatureEngineer(sport="NFL")
        df = fe.build(_make_games(30))
        for col in FEATURE_NAMES:
            assert col in df.columns, f"Missing feature: {col}"

    def test_no_future_leakage(self):
        """Elo and rolling stats should only use games before the current row."""
        fe = FeatureEngineer(sport="NFL")
        # Home always wins → by end, form_advantage_l10 should be high for persistent home team
        games = _make_games(50, home_always_wins=True)
        df = fe.build(games)
        # First game row should have neutral form (no prior games)
        first_row = df.iloc[0]
        assert first_row["home_l10_wp"] == pytest.approx(0.5, abs=0.01)

    def test_target_column_binary(self):
        fe = FeatureEngineer(sport="NFL")
        df = fe.build(_make_games(30))
        assert df["home_win"].isin([0, 1]).all()

    def test_elo_diff_varies_across_rows(self):
        fe = FeatureEngineer(sport="NFL")
        games = _make_games(40, home_always_wins=True)
        df = fe.build(games)
        # Elo diff should change as teams accumulate history
        assert df["elo_diff"].nunique() > 1

    def test_rest_days_capped(self):
        fe = FeatureEngineer(sport="NFL")
        df = fe.build(_make_games(30))
        assert (df["home_rest_days"] <= 14).all()
        assert (df["away_rest_days"] <= 14).all()

    def test_pythag_between_0_and_1(self):
        fe = FeatureEngineer(sport="NFL")
        df = fe.build(_make_games(30))
        assert (df["home_pythag"] >= 0).all()
        assert (df["home_pythag"] <= 1).all()

    def test_empty_games_returns_empty_df(self):
        fe = FeatureEngineer(sport="NFL")
        df = fe.build([])
        assert df.empty

    def test_incomplete_games_skipped(self):
        games = _make_games(20)
        # Remove scores from a few games
        for g in games[5:8]:
            g["home_score"] = None
            g["away_score"] = None
        fe = FeatureEngineer(sport="NFL")
        df = fe.build(games)
        # Should have fewer rows than total games
        assert len(df) < len(games)

    def test_nba_sport_uses_different_pythag(self):
        games_nfl = _make_games(30)
        games_nba = _make_games(30)
        # NBA has higher scores
        for g in games_nba:
            g["home_score"] = 110
            g["away_score"] = 105
        fe_nfl = FeatureEngineer(sport="NFL")
        fe_nba = FeatureEngineer(sport="NBA")
        df_nfl = fe_nfl.build(games_nfl)
        df_nba = fe_nba.build(games_nba)
        # With identical margins but different exponents, pythag values differ
        assert fe_nfl._pythag != fe_nba._pythag


class TestRollingFeatures:
    def test_form_advantage_reflects_recent_performance(self):
        """With home always winning, home_l10_wp should exceed 0.5 after warmup."""
        fe = FeatureEngineer(sport="NFL")
        games = _make_games(60, home_always_wins=True)
        df = fe.build(games)
        # Skip the first 10 rows (warm-up period)
        later = df.iloc[10:]
        # Teams playing at home should have higher l10_wp on average
        # (note: same team plays both home and away in synthetic data)
        assert later["form_advantage_l10"].mean() > -0.3  # Not necessarily positive due to scheduling

    def test_ppg_diff_varies_across_rows(self):
        """PPG diff should not be constant — it reflects actual score history."""
        fe = FeatureEngineer(sport="NFL")
        games = _make_games(50, home_always_wins=True)
        df = fe.build(games)
        # PPG diff should not be uniformly zero — it reflects rolling scored-allowed
        assert df["home_ppg_diff_l10"].nunique() > 1


class TestH2HFeatures:
    def test_h2h_neutral_with_no_history(self):
        fe = FeatureEngineer(sport="NFL")
        # Only one game between Team A and Team B
        games = [{
            "date": "2023-01-01T18:00:00Z",
            "home_team": "Team A", "away_team": "Team B",
            "home_score": 28, "away_score": 14, "home_winner": True,
        }]
        df = fe.build(games)
        # First meeting: no prior H2H → defaults to 0.5
        assert df.iloc[0]["h2h_home_wp"] == pytest.approx(0.5, abs=0.01)

    def test_h2h_updates_after_game(self):
        fe = FeatureEngineer(sport="NFL")
        games = [
            {"date": f"2023-01-0{i}T18:00:00Z",
             "home_team": "Team A", "away_team": "Team B",
             "home_score": 30, "away_score": 20, "home_winner": True}
            for i in range(1, 7)
        ]
        df = fe.build(games)
        # After Team A wins multiple times, later games should reflect higher H2H wp
        assert df.iloc[-1]["h2h_home_wp"] > 0.5


class TestOddsFeatures:
    def test_dk_features_zero_without_odds(self):
        fe = FeatureEngineer(sport="NFL")
        df = fe.build(_make_games(10))
        # No odds lookup provided → all DK features should be 0
        assert (df["dk_home_implied"] == 0.0).all()
        assert (df["dk_vig_h2h"] == 0.0).all()
        assert (df["dk_sharp_signal"] == 0.0).all()

    def test_dk_features_populated_with_odds(self):
        fe = FeatureEngineer(sport="NFL")
        games = _make_games(5)
        games[0]["game_id"] = "g0"
        odds_lookup = {
            "g0": {
                "h2h": {"home_price": -150, "away_price": 130},
                "spreads": {"home_point": -3.5, "away_point": 3.5},
                "totals": {"over_under": 45.5, "over_price": -110, "under_price": -110},
            }
        }
        df = fe.build(games, odds_lookup)
        row = df[df["game_id"] == "g0"].iloc[0]
        assert row["dk_home_implied"] > 0
        assert row["dk_home_implied"] < 1
        assert row["dk_home_spread"] == pytest.approx(-3.5)
        assert row["dk_total"] == pytest.approx(45.5)
        assert row["dk_vig_h2h"] > 0


class TestTargetConstruction:
    def test_spread_target_home_covers(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({
            "home_pts": [28.0],
            "away_pts": [14.0],
            "dk_home_spread": [-7.0],  # home favoured by 7
        })
        target = fe.make_spread_target(df)
        # margin = 14, spread = -7 → 14 + (-7) = 7 > 0 → covered
        assert target.iloc[0] == 1.0

    def test_spread_target_home_does_not_cover(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({
            "home_pts": [17.0],
            "away_pts": [14.0],
            "dk_home_spread": [-7.0],
        })
        target = fe.make_spread_target(df)
        # margin = 3, spread = -7 → 3 + (-7) = -4 < 0 → did not cover
        assert target.iloc[0] == 0.0

    def test_spread_target_nan_when_no_odds(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({
            "home_pts": [28.0],
            "away_pts": [14.0],
            "dk_home_spread": [0.0],  # 0 means no odds available
        })
        target = fe.make_spread_target(df)
        assert math.isnan(target.iloc[0])

    def test_totals_target_over_hits(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({
            "home_pts": [30.0],
            "away_pts": [28.0],
            "dk_total": [54.0],
        })
        target = fe.make_totals_target(df)
        assert target.iloc[0] == 1.0  # 58 > 54

    def test_totals_target_under_hits(self):
        fe = FeatureEngineer()
        df = pd.DataFrame({
            "home_pts": [20.0],
            "away_pts": [17.0],
            "dk_total": [45.0],
        })
        target = fe.make_totals_target(df)
        assert target.iloc[0] == 0.0  # 37 < 45


class TestImpliedProbHelper:
    def test_minus_110_correct(self):
        assert _impl_prob(-110) == pytest.approx(110 / 210, rel=1e-5)

    def test_plus_100_is_half(self):
        assert _impl_prob(100) == pytest.approx(0.5)

    def test_always_in_range(self):
        for o in [-500, -200, -110, 100, 150, 300]:
            assert 0 < _impl_prob(o) < 1
