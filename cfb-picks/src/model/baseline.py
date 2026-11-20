"""Baseline: regresión de puntos ajustada por el nivel del rival (% de victorias).

Es una reconstrucción del prototipo a partir de su descripción (el model.py
original no estaba disponible al escribir esto):

    puntos anotados  = a_equipo + b * (%victorias_rival - 0.5) + c * local
    puntos permitidos = d_equipo + e * (%victorias_rival - 0.5) + f * local

Las pendientes b, e y el efecto de local se estiman con todos los equipos a la
vez; cada equipo tiene su propio intercepto ("lo que anota contra un rival
promedio"). El marcador proyectado promedia la visión de la ofensiva de un
equipo con la de la defensa del otro.

Su debilidad, y la razón para reemplazarlo: el % de victorias no distingue un
8-2 contra calendario fácil de un 8-2 contra calendario difícil.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .data import SeasonArrays


@dataclass
class BaselineState:
    off: np.ndarray  # puntos anotados contra un rival promedio
    deff: np.ndarray  # puntos permitidos contra un rival promedio
    wpct: np.ndarray
    b_off: float
    b_def: float
    home_off: float
    home_def: float
    games: np.ndarray


class BaselineModel:
    name = "baseline"
    label = "Baseline (puntos vs % de victorias del rival)"

    def __init__(self, shrink: float = 1.0) -> None:
        self.shrink = shrink  # juegos "fantasma" hacia el promedio de la liga

    def prior(self, prev, prev_sa, sa: SeasonArrays) -> dict:
        return {}

    def fit(
        self, sa: SeasonArrays, mask: np.ndarray, prior: dict, now_week: int | None = None
    ) -> BaselineState:
        n = sa.n
        idx = np.flatnonzero(mask & sa.completed)
        h, a = sa.h[idx], sa.a[idx]
        hp, ap = sa.hp[idx], sa.ap[idx]
        site = (~sa.neutral[idx]).astype(float)

        games = np.bincount(h, minlength=n) + np.bincount(a, minlength=n)
        wins = np.bincount(h, weights=(hp > ap).astype(float), minlength=n) + np.bincount(
            a, weights=(ap > hp).astype(float), minlength=n
        )
        wpct = np.where(games > 0, wins / np.maximum(games, 1), 0.5)

        # Una fila por equipo-juego (cada juego se ve desde los dos lados).
        team = np.concatenate([h, a])
        opp = np.concatenate([a, h])
        pf = np.concatenate([hp, ap])
        pa = np.concatenate([ap, hp])
        home = np.concatenate([site, -site])
        x = wpct[opp] - 0.5
        mu_pf = float(pf.mean()) if len(pf) else 26.5

        def within(v: np.ndarray) -> np.ndarray:
            means = np.bincount(team, weights=v, minlength=n) / np.maximum(games, 1)
            return v - means[team]

        coef_off = coef_def = np.zeros(2)
        if len(team) >= 60:
            X = np.column_stack([within(x), within(home)])
            coef_off, *_ = np.linalg.lstsq(X, within(pf), rcond=None)
            coef_def, *_ = np.linalg.lstsq(X, within(pa), rcond=None)

        def intercepts(y: np.ndarray, coef: np.ndarray) -> np.ndarray:
            resid = y - coef[0] * x - coef[1] * home
            total = np.bincount(team, weights=resid, minlength=n)
            return (total + self.shrink * mu_pf) / (games + self.shrink)

        return BaselineState(
            off=intercepts(pf, coef_off),
            deff=intercepts(pa, coef_def),
            wpct=wpct,
            b_off=float(coef_off[0]),
            b_def=float(coef_def[0]),
            home_off=float(coef_off[1]),
            home_def=float(coef_def[1]),
            games=games,
        )

    def predict(
        self, s: BaselineState, h: np.ndarray, a: np.ndarray, neutral: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, a = np.asarray(h), np.asarray(a)
        site = (~np.asarray(neutral, dtype=bool)).astype(float)
        xh, xa = s.wpct[h] - 0.5, s.wpct[a] - 0.5  # nivel del local y del visitante
        home_pts = 0.5 * (
            (s.off[h] + s.b_off * xa + s.home_off * site)
            + (s.deff[a] + s.b_def * xh - s.home_def * site)
        )
        away_pts = 0.5 * (
            (s.off[a] + s.b_off * xh - s.home_off * site)
            + (s.deff[h] + s.b_def * xa + s.home_def * site)
        )
        return home_pts - away_pts, home_pts + away_pts

    def describe(self) -> dict:
        return {"modelo": self.name, "shrink": self.shrink}
