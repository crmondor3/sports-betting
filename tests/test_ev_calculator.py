"""Unit tests for EV calculator and odds conversion."""
import math
import pytest
from models.ev_calculator import (
    american_to_decimal,
    decimal_to_american,
    implied_probability,
    no_vig_probability,
    calculate_ev,
    ev_percentage,
    closing_line_value,
    vig_percentage,
)


# ── Odds conversion ────────────────────────────────────────────────────────────

class TestAmericanToDecimal:
    def test_plus_100(self):
        assert american_to_decimal(100) == pytest.approx(2.0)

    def test_plus_200(self):
        assert american_to_decimal(200) == pytest.approx(3.0)

    def test_minus_110(self):
        assert american_to_decimal(-110) == pytest.approx(100 / 110 + 1, rel=1e-5)

    def test_minus_200(self):
        assert american_to_decimal(-200) == pytest.approx(1.5)

    def test_even_money(self):
        assert american_to_decimal(100) == 2.0


class TestDecimalToAmerican:
    def test_2_0(self):
        assert decimal_to_american(2.0) == pytest.approx(100.0)

    def test_3_0(self):
        assert decimal_to_american(3.0) == pytest.approx(200.0)

    def test_1_5(self):
        assert decimal_to_american(1.5) == pytest.approx(-200.0)

    def test_roundtrip_positive(self):
        for odds in [110, 150, 200, 350, -120, -150, -200]:
            decimal = american_to_decimal(odds)
            back = decimal_to_american(decimal)
            assert back == pytest.approx(odds, rel=1e-4), f"Round-trip failed for {odds}"


# ── Implied probability ────────────────────────────────────────────────────────

class TestImpliedProbability:
    def test_plus_100_is_50pct(self):
        assert implied_probability(100) == pytest.approx(0.5)

    def test_minus_200_is_67pct(self):
        assert implied_probability(-200) == pytest.approx(2 / 3, rel=1e-5)

    def test_minus_110(self):
        assert implied_probability(-110) == pytest.approx(110 / 210, rel=1e-5)

    def test_always_between_0_and_1(self):
        for odds in [-500, -200, -110, 100, 150, 300, 1000]:
            p = implied_probability(odds)
            assert 0 < p < 1, f"Probability out of range for {odds}: {p}"


class TestNoVigProbability:
    def test_sum_to_one(self):
        p_h, p_a = no_vig_probability(-110, -110)
        assert p_h + p_a == pytest.approx(1.0)

    def test_symmetric_market(self):
        p_h, p_a = no_vig_probability(-110, -110)
        assert p_h == pytest.approx(0.5, rel=1e-5)

    def test_favourite_has_higher_prob(self):
        p_fav, p_dog = no_vig_probability(-200, +160)
        assert p_fav > p_dog

    def test_vig_positive(self):
        vig = vig_percentage(-110, -110)
        assert vig > 0


# ── Expected Value ─────────────────────────────────────────────────────────────

class TestCalculateEV:
    def test_positive_ev_when_model_beats_line(self):
        # Coin flip priced at -110 but we give it 55% — should be +EV
        ev = calculate_ev(0.55, -110)
        assert ev > 0

    def test_negative_ev_when_model_below_line(self):
        ev = calculate_ev(0.45, -110)
        assert ev < 0

    def test_zero_ev_at_fair_odds(self):
        # -110 implied prob ≈ 52.38%; if our model agrees exactly → ~0 EV
        fair_prob = 110 / 210
        ev = calculate_ev(fair_prob, -110)
        # Won't be exactly 0 due to vig definition; just check near-zero
        assert abs(ev) < 0.02

    def test_high_plus_odds_large_ev(self):
        # +500 dog we think has 25% chance
        ev = calculate_ev(0.25, 500)
        assert ev > 0

    def test_invalid_probability_raises(self):
        with pytest.raises(ValueError):
            calculate_ev(0.0, -110)
        with pytest.raises(ValueError):
            calculate_ev(1.0, -110)
        with pytest.raises(ValueError):
            calculate_ev(1.5, -110)

    def test_ev_percentage_scaling(self):
        ev = calculate_ev(0.55, -110)
        evp = ev_percentage(0.55, -110)
        assert evp == pytest.approx(ev * 100)


# ── CLV ────────────────────────────────────────────────────────────────────────

class TestClosingLineValue:
    def test_positive_clv_when_line_moved_against_us(self):
        # Opened -110, closed at -130 — we got better than closing
        clv = closing_line_value(-110, -130)
        assert clv > 0

    def test_negative_clv_when_line_moved_in_our_favour(self):
        clv = closing_line_value(-110, -100)
        assert clv < 0

    def test_zero_clv_same_line(self):
        clv = closing_line_value(-110, -110)
        assert clv == pytest.approx(0.0, abs=1e-10)
