"""Probabilidades implícitas, vig, normal, Kelly y parlays."""

import math

import numpy as np
import pytest
from src.model.prob import (
    american_to_decimal,
    american_to_prob,
    bet_profit,
    devig_one_side,
    devig_two_way,
    expected_value,
    kelly_fraction,
    norm_cdf,
    overround,
    p_cover,
    p_home_win,
    p_over,
    parlay,
    prob_to_american,
    suggested_stake,
)
from src.model.rules import norm_ppf


@pytest.mark.parametrize(
    "odds, prob",
    [
        (-110, 110 / 210),
        (150, 0.4),
        (100, 0.5),
        (-100, 0.5),
        (-1300, 1300 / 1400),
        (650, 100 / 750),
        (-475, 475 / 575),
    ],
)
def test_american_to_prob(odds, prob):
    assert american_to_prob(odds) == pytest.approx(prob)


@pytest.mark.parametrize(
    "odds, decimal", [(-110, 1.9090909), (150, 2.5), (100, 2.0), (-200, 1.5), (-1600, 1.0625)]
)
def test_american_to_decimal(odds, decimal):
    assert american_to_decimal(odds) == pytest.approx(decimal)


@pytest.mark.parametrize("bad", [0, 50, -99])
def test_invalid_american_odds_raise(bad):
    with pytest.raises(ValueError):
        american_to_decimal(bad)


def test_devig_two_way_removes_the_vig():
    assert devig_two_way(-110, -110) == pytest.approx((0.5, 0.5))
    fav, dog = devig_two_way(-300, 240)
    assert fav + dog == pytest.approx(1.0)
    assert fav == pytest.approx(0.75 / (0.75 + 100 / 340))
    assert fav < american_to_prob(-300)  # sin vig, el favorito vale menos que su implícita cruda
    assert overround(-110, -110) == pytest.approx(1.047619)


def test_devig_one_side_uses_assumed_overround():
    assert devig_one_side(-110, overround=1.047619) == pytest.approx(0.5, abs=1e-4)
    assert devig_one_side(-100000) < 1.0


@pytest.mark.parametrize("p", [0.1, 0.35, 0.5, 0.6, 0.9])
def test_prob_to_american_roundtrip(p):
    assert american_to_prob(prob_to_american(p)) == pytest.approx(p, abs=0.003)
    assert prob_to_american(1.0) is None and prob_to_american(0.0) is None


def test_norm_cdf_and_inverse():
    assert norm_cdf(0) == pytest.approx(0.5)
    assert norm_cdf(1.0) == pytest.approx(0.841345, abs=1e-6)
    assert norm_cdf(-1.96) == pytest.approx(0.025, abs=1e-4)
    arr = norm_cdf(np.array([-1.0, 0.0, 1.0]))
    assert arr[0] + arr[2] == pytest.approx(1.0) and arr[1] == pytest.approx(0.5)
    assert norm_ppf(0.841345) == pytest.approx(1.0, abs=1e-4)
    assert norm_ppf(0.5) == pytest.approx(0.0, abs=1e-6)


def test_model_probabilities():
    sigma = 16.0
    assert p_home_win(0, sigma) == pytest.approx(0.5)
    assert p_home_win(16, sigma) == pytest.approx(norm_cdf(1.0))
    # Local favorito por 10 según el modelo: cubre -7.5 más de la mitad de las veces, -10 justo la mitad.
    assert p_cover(10, -7.5, sigma) > 0.5
    assert p_cover(10, -10, sigma) == pytest.approx(0.5)
    assert p_cover(10, -14, sigma) < 0.5
    # El visitante +7.5 es el complemento del local -7.5.
    assert 1 - p_cover(10, -7.5, sigma) == pytest.approx(norm_cdf((7.5 - 10) / sigma))
    assert p_over(50, 45.5, sigma) > 0.5 > p_over(40, 45.5, sigma)


def test_expected_value_and_kelly():
    assert expected_value(0.5, 100) == pytest.approx(0.0)
    assert expected_value(110 / 210, -110) == pytest.approx(0.0)
    assert expected_value(0.55, -110) == pytest.approx(0.05)
    b = 100 / 110
    assert kelly_fraction(0.55, -110) == pytest.approx((b * 0.55 - 0.45) / b)
    assert kelly_fraction(0.50, -110) == 0.0  # sin ventaja no se apuesta
    assert kelly_fraction(0.40, 200) == pytest.approx((2 * 0.4 - 0.6) / 2)


def test_stake_is_quarter_kelly_with_cap():
    full = kelly_fraction(0.55, -110)
    assert suggested_stake(0.55, -110) == pytest.approx(0.25 * full)
    assert suggested_stake(0.70, -110) == 0.02  # tope de 2% del bankroll
    assert suggested_stake(0.50, -110) == 0.0


def test_bet_profit():
    assert bet_profit(-110, True) == pytest.approx(100 / 110)
    assert bet_profit(150, True) == pytest.approx(1.5)
    assert bet_profit(150, False) == -1.0


def test_parlay_multiplies_probabilities_and_prices():
    out = parlay([0.8, 0.5], [-400, 100])
    assert out["prob"] == pytest.approx(0.4)
    assert out["decimal"] == pytest.approx(1.25 * 2.0)
    assert out["prob_casa"] == pytest.approx(1 / 2.5)
    assert out["ev"] == pytest.approx(0.4 * 2.5 - 1)
    missing = parlay([0.8, 0.5], [-400, None])
    assert (
        missing["prob"] == pytest.approx(0.4)
        and missing["decimal"] is None
        and missing["ev"] is None
    )


def test_vig_compounds_in_parlays():
    legs = [-110] * 10
    out = parlay([0.5] * 10, legs)
    assert out["prob"] == pytest.approx(0.5**10)
    assert out["prob_casa"] > out["prob"] * 1.5  # la casa paga como si fuera mucho más probable
    assert math.isclose(out["prob_casa"], (110 / 210) ** 10)
