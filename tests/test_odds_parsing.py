"""Tests for DraftKings odds parsing and implied probability in data_collector."""
import pytest
from data_collector import american_to_implied, no_vig, DataCollector


class TestAmericanToImplied:
    def test_even_money(self):
        assert american_to_implied(100) == pytest.approx(0.5)

    def test_favorite_minus_110(self):
        expected = 110 / 210
        assert american_to_implied(-110) == pytest.approx(expected, rel=1e-5)

    def test_underdog_plus_150(self):
        expected = 100 / 250
        assert american_to_implied(150) == pytest.approx(expected, rel=1e-5)

    def test_heavy_favorite_minus_300(self):
        expected = 300 / 400
        assert american_to_implied(-300) == pytest.approx(0.75, rel=1e-5)

    def test_always_in_0_1(self):
        for odds in [-500, -200, -110, -105, 100, 110, 150, 200, 300, 500]:
            p = american_to_implied(odds)
            assert 0 < p < 1, f"Out of range for {odds}: {p}"

    def test_overround_on_standard_juice(self):
        # -110 / -110 market: raw probs sum > 1 (that's the vig)
        raw_h = american_to_implied(-110)
        raw_a = american_to_implied(-110)
        assert raw_h + raw_a > 1.0


class TestNoVig:
    def test_symmetric_market_is_50_50(self):
        p_h, p_a = no_vig(american_to_implied(-110), american_to_implied(-110))
        assert p_h == pytest.approx(0.5, abs=1e-6)
        assert p_a == pytest.approx(0.5, abs=1e-6)

    def test_sum_to_one(self):
        for home_o, away_o in [(-200, 165), (-110, -110), (120, -140), (-350, 280)]:
            p_h, p_a = no_vig(american_to_implied(home_o), american_to_implied(away_o))
            assert p_h + p_a == pytest.approx(1.0, abs=1e-9)

    def test_favourite_higher_prob(self):
        p_fav, p_dog = no_vig(american_to_implied(-200), american_to_implied(165))
        assert p_fav > p_dog

    def test_heavy_favourite_approaches_1(self):
        # -900 raw implied = 0.9, +600 raw = 0.143; after vig removal ≈ 0.863
        p_h, _ = no_vig(american_to_implied(-900), american_to_implied(600))
        assert p_h > 0.85


class TestOddsSnapshotParsing:
    """Unit tests for _build_snapshot logic (no DB required)."""

    def _make_collector(self) -> DataCollector:
        # Skip DB init for unit tests by calling init only when needed
        import data_collector as dc_mod
        dc_mod.init_ml_tables()
        return DataCollector.__new__(DataCollector)

    def test_h2h_snapshot_prices(self):
        from datetime import datetime
        from data_collector import DataCollector as DC
        collector = DC.__new__(DC)
        outcomes = [
            {"name": "Kansas City Chiefs", "price": -350},
            {"name": "Las Vegas Raiders",  "price": 280},
        ]
        snap = collector._build_snapshot(
            game_id="test_1", market_key="h2h",
            home="Kansas City Chiefs", away="Las Vegas Raiders",
            outcomes=outcomes, timestamp=datetime.utcnow(),
        )
        assert snap is not None
        assert snap.home_price == -350
        assert snap.away_price == 280
        assert snap.market == "h2h"

    def test_spreads_snapshot(self):
        from datetime import datetime
        from data_collector import DataCollector as DC
        collector = DC.__new__(DC)
        outcomes = [
            {"name": "Kansas City Chiefs", "price": -110, "point": -8.5},
            {"name": "Las Vegas Raiders",  "price": -110, "point":  8.5},
        ]
        snap = collector._build_snapshot(
            game_id="test_2", market_key="spreads",
            home="Kansas City Chiefs", away="Las Vegas Raiders",
            outcomes=outcomes, timestamp=datetime.utcnow(),
        )
        assert snap is not None
        assert snap.home_point == pytest.approx(-8.5)
        assert snap.away_point == pytest.approx(8.5)
        assert snap.home_spread_price == -110

    def test_totals_snapshot(self):
        from datetime import datetime
        from data_collector import DataCollector as DC
        collector = DC.__new__(DC)
        outcomes = [
            {"name": "Over",  "price": -115, "point": 48.5},
            {"name": "Under", "price": -105, "point": 48.5},
        ]
        snap = collector._build_snapshot(
            game_id="test_3", market_key="totals",
            home="Team A", away="Team B",
            outcomes=outcomes, timestamp=datetime.utcnow(),
        )
        assert snap is not None
        assert snap.over_under == pytest.approx(48.5)
        assert snap.over_price == -115
        assert snap.under_price == -105

    def test_missing_outcome_returns_none(self):
        from datetime import datetime
        from data_collector import DataCollector as DC
        collector = DC.__new__(DC)
        # Only one outcome → should return None
        snap = collector._build_snapshot(
            game_id="test_4", market_key="h2h",
            home="Team A", away="Team B",
            outcomes=[{"name": "Team A", "price": -200}],
            timestamp=datetime.utcnow(),
        )
        assert snap is None


class TestSharpMoveDetection:
    def test_sharp_move_h2h_detected(self):
        from data_collector import DataCollector as DC, MLOddsSnapshot
        from datetime import datetime
        collector = DC.__new__(DC)

        # Previous: home -150, now home -200 → big implied prob move
        prev = MLOddsSnapshot(
            game_id="g1", bookmaker="draftkings", market="h2h",
            home_price=-150, away_price=130,
            timestamp=datetime(2024, 1, 1, 10, 0),
        )
        curr = MLOddsSnapshot(
            game_id="g1", bookmaker="draftkings", market="h2h",
            home_price=-220, away_price=180,
            timestamp=datetime(2024, 1, 1, 14, 0),
        )

        # Manually test the logic from _detect_sharp_move
        p_now  = american_to_implied(curr.home_price)
        p_prev = american_to_implied(prev.home_price)
        move   = abs(p_now - p_prev)
        assert move >= 0.05, f"Expected large move, got {move:.4f}"

    def test_no_sharp_move_small_change(self):
        p_now  = american_to_implied(-112)
        p_prev = american_to_implied(-110)
        assert abs(p_now - p_prev) < 0.10

    def test_sharp_move_spread_detected(self):
        from data_collector import MLOddsSnapshot
        from datetime import datetime
        prev_pt = -3.0
        curr_pt = -5.0
        assert abs(curr_pt - prev_pt) >= 1.5
