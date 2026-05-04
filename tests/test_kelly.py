"""Unit tests for Kelly Criterion sizing."""
import pytest
from models.kelly_criterion import (
    kelly_fraction,
    stake_amount,
    flat_stake,
    expected_growth,
)


class TestKellyFraction:
    def test_positive_edge_returns_positive_fraction(self):
        # 55% model prob on -110 → positive edge → fraction > 0
        frac = kelly_fraction(0.55, -110)
        assert frac > 0

    def test_no_edge_returns_zero(self):
        # Model prob exactly equals implied prob → no edge
        # -110 implied ≈ 52.38%
        frac = kelly_fraction(0.52, -110)
        assert frac == 0.0

    def test_negative_ev_returns_zero(self):
        frac = kelly_fraction(0.40, -110)
        assert frac == 0.0

    def test_fraction_capped_at_max_kelly(self):
        import config
        # Even with massive edge, should not exceed cap
        frac = kelly_fraction(0.95, 100)
        assert frac <= config.KELLY_CAP

    def test_fractional_kelly_smaller_than_full(self):
        # Quarter Kelly should be ≤ full Kelly
        quarter = kelly_fraction(0.60, -110, fractional=0.25)
        full = kelly_fraction(0.60, -110, fractional=1.0)
        assert quarter <= full

    def test_invalid_prob_raises(self):
        with pytest.raises(ValueError):
            kelly_fraction(0.0, -110)
        with pytest.raises(ValueError):
            kelly_fraction(1.0, 200)

    def test_returns_float(self):
        frac = kelly_fraction(0.55, 100)
        assert isinstance(frac, float)

    def test_higher_odds_higher_fraction_for_same_prob(self):
        # Same probability but higher odds → bigger Kelly (more value)
        frac_minus110 = kelly_fraction(0.60, -110)
        frac_plus150 = kelly_fraction(0.60, 150)
        assert frac_plus150 > frac_minus110


class TestStakeAmount:
    def test_stake_scales_with_bankroll(self):
        s1 = stake_amount(1000, 0.55, -110)
        s2 = stake_amount(2000, 0.55, -110)
        assert s2 == pytest.approx(s1 * 2, rel=1e-5)

    def test_stake_is_zero_with_no_edge(self):
        s = stake_amount(1000, 0.40, -110)
        assert s == 0.0

    def test_stake_is_rounded_to_two_dp(self):
        s = stake_amount(1000, 0.55, -110)
        assert round(s, 2) == s

    def test_stake_never_exceeds_kelly_cap_fraction_of_bankroll(self):
        import config
        bankroll = 1000
        s = stake_amount(bankroll, 0.99, 1000)  # ridiculous edge
        assert s <= bankroll * config.KELLY_CAP + 0.01  # +0.01 for rounding


class TestFlatStake:
    def test_default_two_percent(self):
        assert flat_stake(1000) == pytest.approx(20.0)

    def test_custom_pct(self):
        assert flat_stake(1000, pct=0.05) == pytest.approx(50.0)

    def test_rounded(self):
        s = flat_stake(333.33)
        assert round(s, 2) == s


class TestExpectedGrowth:
    def test_positive_edge_positive_growth(self):
        g = expected_growth(0.55, -110, 0.05)
        assert g > 0

    def test_kelly_fraction_maximises_growth(self):
        """Full Kelly fraction should maximise log-growth."""
        # Find full Kelly
        full_kelly = kelly_fraction(0.60, 100, fractional=1.0)
        growth_kelly = expected_growth(0.60, 100, full_kelly)
        # Half Kelly or double Kelly should be worse
        if full_kelly > 0:
            growth_half = expected_growth(0.60, 100, full_kelly / 2)
            assert growth_kelly >= growth_half

    def test_overbetting_reduces_growth(self):
        # Betting more than Kelly on a +EV bet reduces expected log-growth
        import config
        kelly = kelly_fraction(0.55, -110, fractional=1.0)
        if kelly > 0:
            g_kelly = expected_growth(0.55, -110, kelly)
            g_over = expected_growth(0.55, -110, min(kelly * 2, 0.5))
            assert g_kelly >= g_over
