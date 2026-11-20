"""Elo with margin of victory.

Idea: each team has a rating; before the game the rating difference gives an
expected probability, and after the game the winner takes points from the
loser in proportion to the surprise. The margin multiplier makes a blowout
move ratings more than a 3-point win, but with diminishing returns and
discounting the fact that the favorite usually wins by more.

Unlike SRS, Elo is sequential: the order of the games matters and the ratings
carry over from one season to the next (regressed to the mean).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .data import SeasonArrays, align

MEAN = 1500.0


@dataclass(frozen=True)
class EloParams:
    k: float = 25.0  # update speed
    hfa: float = 55.0  # home-field advantage in Elo points
    carry: float = 0.65  # fraction of the rating kept between seasons
    scale: float = 25.0  # Elo points per point of margin
    newcomer: float = 1350.0
    fcs0: float = 1150.0
    fcs_k: float = 0.1  # the FCS average barely moves with one game


@dataclass
class EloState:
    rating: np.ndarray
    games: np.ndarray


class EloModel:
    name = "elo"
    label = "Elo with margin of victory"

    def __init__(self, params: EloParams | None = None) -> None:
        self.params = params or EloParams()

    def prior(self, prev: EloState | None, prev_sa: SeasonArrays | None, sa: SeasonArrays) -> dict:
        p = self.params
        if prev is None or prev_sa is None:
            rating = np.full(sa.n, MEAN)
            rating[sa.fcs_index] = p.fcs0
            return {"rating": rating}
        carried = align(prev.rating, prev_sa, sa, default=np.nan)
        rating = MEAN + p.carry * (carried - MEAN)
        rating[np.isnan(rating)] = p.newcomer
        rating[sa.fcs_index] = prev.rating[prev_sa.fcs_index]
        return {"rating": rating}

    def fit(
        self, sa: SeasonArrays, mask: np.ndarray, prior: dict, now_week: int | None = None
    ) -> EloState:
        p = self.params
        rating = prior["rating"].astype(float).copy()
        games = np.zeros(sa.n, dtype=int)
        fcs_i = sa.fcs_index
        for g in np.flatnonzero(mask & sa.completed):  # arrays are already in chronological order
            h, a = sa.h[g], sa.a[g]
            diff = rating[h] - rating[a] + (0.0 if sa.neutral[g] else p.hfa)
            expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
            margin = sa.hp[g] - sa.ap[g]
            won = 1.0 if margin > 0 else 0.0
            winner_diff = diff if margin > 0 else -diff
            mult = math.log(abs(margin) + 1.0) * 2.2 / (0.001 * winner_diff + 2.2)
            delta = p.k * mult * (won - expected)
            rating[h] += delta * (p.fcs_k if h == fcs_i else 1.0)
            rating[a] -= delta * (p.fcs_k if a == fcs_i else 1.0)
            games[h] += 1
            games[a] += 1
        return EloState(rating=rating, games=games)

    def predict(
        self, state: EloState, h: np.ndarray, a: np.ndarray, neutral: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        p = self.params
        h, a = np.asarray(h), np.asarray(a)
        site = (~np.asarray(neutral, dtype=bool)).astype(float)
        margin = (state.rating[h] - state.rating[a] + p.hfa * site) / p.scale
        return margin, np.full(len(h), np.nan)  # Elo does not project totals

    def describe(self) -> dict:
        return {"model": self.name, **asdict(self.params)}
