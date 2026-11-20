"""Probabilities: normal, American odds, vig, Kelly and parlays.

Everything is pure arithmetic (no scipy) so it is easy to test.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Sequence

import numpy as np

_SQRT2 = math.sqrt(2.0)


def norm_cdf(x):
    """Standard normal CDF. Accepts a scalar or an array."""
    if np.isscalar(x):
        return 0.5 * (1.0 + math.erf(float(x) / _SQRT2))
    arr = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + np.vectorize(math.erf, otypes=[float])(arr / _SQRT2))


# -------------------------------------------------------- from the model
def p_home_win(margin: float, sigma: float) -> float:
    """P(home team wins) if the actual margin ~ Normal(projected margin, sigma)."""
    return float(norm_cdf(margin / sigma))


def p_cover(margin: float, spread_home: float, sigma: float) -> float:
    """P(home team covers). spread_home is the home team's line (-7.5 = favored by 7.5).
    It covers if actual_margin + spread_home > 0."""
    return float(norm_cdf((margin + spread_home) / sigma))


def p_over(total: float, line: float, sigma: float) -> float:
    return float(norm_cdf((total - line) / sigma))


# --------------------------------------------------------------- odds
def american_to_decimal(odds: float) -> float:
    """American odds -> decimal (total payout per unit staked). -110 -> 1.909, +150 -> 2.5."""
    odds = float(odds)
    if abs(odds) < 100:
        raise ValueError(f"invalid American odds: {odds}")
    return 1.0 + (odds / 100.0 if odds > 0 else 100.0 / -odds)


def american_to_prob(odds: float) -> float:
    """Implied probability WITH vig. -110 -> 0.5238, +150 -> 0.40."""
    return 1.0 / american_to_decimal(odds)


def prob_to_american(p: float) -> int | None:
    """Fair odds for a probability (to show 'the model line')."""
    if not 0.0 < p < 1.0:
        return None
    if p >= 0.5:
        return round(-100.0 * p / (1.0 - p))
    return round(100.0 * (1.0 - p) / p)


DEFAULT_OVERROUND = 1.045  # typical vig of a two-way market (-110/-110 gives 1.0476)


def devig_two_way(odds_a: float, odds_b: float) -> tuple[float, float]:
    """Remove the vig by spreading it proportionally (multiplicative method).
    -110/-110 -> (0.5, 0.5)."""
    pa, pb = american_to_prob(odds_a), american_to_prob(odds_b)
    total = pa + pb
    return pa / total, pb / total


def devig_one_side(odds: float, overround: float = DEFAULT_OVERROUND) -> float:
    """Without the other side the vig cannot be measured: an overround is assumed."""
    return min(american_to_prob(odds) / overround, 0.999)


def overround(odds_a: float, odds_b: float) -> float:
    return american_to_prob(odds_a) + american_to_prob(odds_b)


# ---------------------------------------------------------------- bets
def expected_value(p: float, odds: float) -> float:
    """Expected profit per unit staked at the actual odds (with vig)."""
    return p * american_to_decimal(odds) - 1.0


def kelly_fraction(p: float, odds: float) -> float:
    """Full Kelly fraction: (b*p - q) / b, with b = decimal - 1. Never negative."""
    b = american_to_decimal(odds) - 1.0
    return max(0.0, (b * p - (1.0 - p)) / b)


def suggested_stake(p: float, odds: float, fraction: float = 0.25, cap: float = 0.02) -> float:
    """Fractional Kelly (1/4 by default) with a cap as a fraction of the bankroll."""
    return min(fraction * kelly_fraction(p, odds), cap)


def bet_profit(odds: float, won: bool) -> float:
    """Result of betting 1 unit."""
    return american_to_decimal(odds) - 1.0 if won else -1.0


def parlay(probs: Sequence[float], odds: Iterable[float | None]) -> dict:
    """Parlay of independent legs.

    prob         = product of the model probabilities
    decimal      = product of the decimal odds (what the book pays)
    implied_prob = 1 / decimal (implied probability of the ticket, with vig)
    ev           = expected profit per unit
    If the odds for any leg are missing, the payout is not computed.
    """
    prob = float(np.prod(list(probs))) if len(probs) else 0.0
    odds = list(odds)
    if odds and all(o is not None for o in odds):
        decimal = float(np.prod([american_to_decimal(o) for o in odds]))
        return {
            "prob": prob,
            "decimal": decimal,
            "implied_prob": 1.0 / decimal,
            "ev": prob * decimal - 1.0,
        }
    return {"prob": prob, "decimal": None, "implied_prob": None, "ev": None}
