"""SRS: ratings ajustados por calendario.

Idea (SRS clásico): el rating de un equipo es su margen promedio más el rating
promedio de sus rivales. Como los ratings de los rivales dependen a su vez de
los suyos, se itera hasta que dejan de cambiar.

Aquí se resuelve directamente el sistema lineal al que converge esa iteración
(mínimos cuadrados con encogimiento), con cuatro mejoras:

1. Tope a las palizas (`cap`): ganar por 50 no dice mucho más que ganar por 28.
2. Encogimiento (`k` juegos "fantasma") hacia un prior: con pocos juegos el
   rating no se dispara.
3. Prior de pretemporada: una fracción (`rho`) del rating final del año anterior.
4. Ventaja de local estimada con los mismos datos.

Un segundo sistema idéntico sobre los TOTALES da el nivel de anotación de cada
equipo, y de ahí sale el marcador: local = (total + margen) / 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data import SeasonArrays, align


@dataclass(frozen=True)
class SrsParams:
    cap: float = 28.0  # tope del margen por juego
    k: float = 2.0  # encogimiento del rating de margen (en juegos)
    k_total: float = 4.0  # encogimiento del rating de totales
    rho: float = 0.6  # fracción del rating del año anterior que entra como prior
    rho_total: float = 0.5
    half_life: float = 0.0  # semanas; 0 = todos los juegos pesan igual
    hfa0: float = 2.5  # prior de la ventaja de local
    hfa_k: float = 40.0
    newcomer: float = -8.0  # prior de un equipo recién llegado a FBS
    fcs0: float = -22.0  # prior del pseudo-equipo FCS si no hay año anterior
    base_total0: float = 53.0
    base_total_k: float = 10.0


@dataclass
class SrsState:
    rating: np.ndarray  # puntos mejor que el promedio (margen)
    pace: np.ndarray  # aporte del equipo al total del juego
    hfa: float
    base_total: float
    games: np.ndarray  # juegos usados por equipo


class SrsModel:
    name = "srs"
    label = "SRS (margen ajustado por calendario)"

    def __init__(self, params: SrsParams | None = None) -> None:
        self.params = params or SrsParams()

    # ------------------------------------------------------------- prior
    def prior(self, prev: SrsState | None, prev_sa: SeasonArrays | None, sa: SeasonArrays) -> dict:
        p = self.params
        if prev is None or prev_sa is None:
            rating = np.zeros(sa.n)
            rating[sa.fcs_index] = p.fcs0
            return {"rating": rating, "pace": np.zeros(sa.n), "base_total": p.base_total0}
        rating = p.rho * align(prev.rating, prev_sa, sa, default=np.nan)
        rating[np.isnan(rating)] = p.newcomer
        rating[sa.fcs_index] = prev.rating[
            prev_sa.fcs_index
        ]  # el promedio FCS casi no cambia de un año a otro
        pace = p.rho_total * align(prev.pace, prev_sa, sa, default=0.0)
        pace[sa.fcs_index] = prev.pace[prev_sa.fcs_index]
        return {"rating": rating, "pace": pace, "base_total": prev.base_total}

    # --------------------------------------------------------------- fit
    def fit(
        self, sa: SeasonArrays, mask: np.ndarray, prior: dict, now_week: int | None = None
    ) -> SrsState:
        p = self.params
        n = sa.n
        idx = np.flatnonzero(mask & sa.completed)
        h, a = sa.h[idx], sa.a[idx]
        site = (~sa.neutral[idx]).astype(float)
        w = np.ones(len(idx))
        if p.half_life > 0 and now_week is not None and len(idx):
            w = 0.5 ** (np.maximum(now_week - sa.week[idx], 0) / p.half_life)

        # ---- margen: m = R_local - R_visita + hfa * (no neutral)
        m = np.clip(sa.hp[idx] - sa.ap[idx], -p.cap, p.cap)
        A = np.zeros((n + 1, n + 1))
        b = np.zeros(n + 1)
        np.add.at(A, (h, h), w)
        np.add.at(A, (a, a), w)
        np.add.at(A, (h, a), -w)
        np.add.at(A, (a, h), -w)
        ws = w * site
        np.add.at(A, (h, n), ws)
        np.add.at(A, (n, h), ws)
        np.add.at(A, (a, n), -ws)
        np.add.at(A, (n, a), -ws)
        A[n, n] += float((ws * site).sum()) + p.hfa_k
        np.add.at(b, h, w * m)
        np.add.at(b, a, -w * m)
        b[n] += float((ws * m).sum()) + p.hfa_k * p.hfa0
        diag = np.arange(n)
        A[diag, diag] += p.k
        b[:n] += p.k * prior["rating"]
        beta = np.linalg.solve(A, b)

        # ---- total: t = base + S_local + S_visita
        t = sa.hp[idx] + sa.ap[idx]
        A2 = np.zeros((n + 1, n + 1))
        b2 = np.zeros(n + 1)
        np.add.at(A2, (h, h), w)
        np.add.at(A2, (a, a), w)
        np.add.at(A2, (h, a), w)
        np.add.at(A2, (a, h), w)
        np.add.at(A2, (h, n), w)
        np.add.at(A2, (n, h), w)
        np.add.at(A2, (a, n), w)
        np.add.at(A2, (n, a), w)
        A2[n, n] += float(w.sum()) + p.base_total_k
        np.add.at(b2, h, w * t)
        np.add.at(b2, a, w * t)
        b2[n] += float((w * t).sum()) + p.base_total_k * prior["base_total"]
        A2[diag, diag] += p.k_total
        b2[:n] += p.k_total * prior["pace"]
        gamma = np.linalg.solve(A2, b2)

        games = np.bincount(h, minlength=n) + np.bincount(a, minlength=n)
        return SrsState(
            rating=beta[:n],
            pace=gamma[:n],
            hfa=float(beta[n]),
            base_total=float(gamma[n]),
            games=games,
        )

    # ----------------------------------------------------------- predict
    def predict(
        self, state: SrsState, h: np.ndarray, a: np.ndarray, neutral: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        h, a = np.asarray(h), np.asarray(a)
        site = (~np.asarray(neutral, dtype=bool)).astype(float)
        margin = state.rating[h] - state.rating[a] + state.hfa * site
        total = state.base_total + state.pace[h] + state.pace[a]
        return margin, total

    def describe(self) -> dict:
        return {"modelo": self.name, **asdict(self.params)}


def srs_iterative(
    sa: SeasonArrays,
    mask: np.ndarray,
    hfa: float,
    cap: float,
    k: float = 0.0,
    prior: np.ndarray | None = None,
    iters: int = 5000,
    tol: float = 1e-10,
) -> np.ndarray:
    """La versión literal del SRS: rating = margen promedio + rating promedio de rivales,
    repetido hasta converger. Existe para documentar (y testear) que `SrsModel.fit`
    llega al mismo resultado resolviendo el sistema de una vez."""
    idx = np.flatnonzero(mask & sa.completed)
    h, a = sa.h[idx], sa.a[idx]
    m = np.clip(sa.hp[idx] - sa.ap[idx], -cap, cap) - hfa * (~sa.neutral[idx])
    prior = np.zeros(sa.n) if prior is None else prior
    games = np.bincount(h, minlength=sa.n) + np.bincount(a, minlength=sa.n)
    rating = np.zeros(sa.n)
    for _ in range(iters):
        acc = np.zeros(sa.n)
        np.add.at(acc, h, m + rating[a])  # el local: su margen + rating del rival
        np.add.at(acc, a, -m + rating[h])  # el visitante: lo mismo visto desde su lado
        new = (acc + k * prior) / np.maximum(games + k, 1e-12)
        new[games + k == 0] = 0.0
        # Promediar con el paso anterior evita que la iteración oscile.
        new = 0.5 * (new + rating)
        if np.max(np.abs(new - rating)) < tol:
            return new
        rating = new
    return rating
