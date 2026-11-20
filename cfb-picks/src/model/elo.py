"""Elo con margen de victoria.

Idea: cada equipo tiene un rating; antes del juego la diferencia de ratings da
una probabilidad esperada, y después del juego el ganador le quita puntos al
perdedor en proporción a la sorpresa. El multiplicador de margen hace que una
paliza mueva más que una victoria por 3, pero con rendimientos decrecientes y
descontando que el favorito suele ganar por más.

A diferencia del SRS, Elo es secuencial: el orden de los juegos importa y los
ratings se arrastran de una temporada a la siguiente (regresados a la media).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from .data import SeasonArrays, align

MEAN = 1500.0


@dataclass(frozen=True)
class EloParams:
    k: float = 25.0  # velocidad de actualización
    hfa: float = 55.0  # ventaja de local en puntos Elo
    carry: float = 0.65  # fracción del rating que se conserva entre temporadas
    scale: float = 25.0  # puntos Elo por punto de margen
    newcomer: float = 1350.0
    fcs0: float = 1150.0
    fcs_k: float = 0.1  # el promedio FCS casi no se mueve con un juego


@dataclass
class EloState:
    rating: np.ndarray
    games: np.ndarray


class EloModel:
    name = "elo"
    label = "Elo con margen de victoria"

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
        for g in np.flatnonzero(mask & sa.completed):  # los arreglos ya están en orden cronológico
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
        return margin, np.full(len(h), np.nan)  # Elo no proyecta totales

    def describe(self) -> dict:
        return {"modelo": self.name, **asdict(self.params)}
