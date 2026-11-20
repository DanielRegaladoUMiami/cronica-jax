"""SRS: schedule-adjusted ratings.

Idea (classic SRS): a team's rating is its average margin plus the average
rating of its opponents. Since the opponents' ratings depend in turn on those
of their own opponents, this is iterated until they stop changing.

Here the linear system that iteration converges to is solved directly
(least squares with shrinkage), with four improvements:

1. Blowout cap (`cap`): winning by 50 does not say much more than winning by 28.
2. Shrinkage (`k` "phantom" games) toward a prior: with few games the
   rating does not blow up.
3. Preseason prior: a fraction (`rho`) of the previous year's final rating.
4. Home-field advantage estimated from the same data.

A second, identical system on the TOTALS gives each team's scoring level,
and the score follows from it: home = (total + margin) / 2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .data import SeasonArrays, align


@dataclass(frozen=True)
class SrsParams:
    cap: float = 28.0  # cap on the margin per game
    k: float = 2.0  # shrinkage of the margin rating (in games)
    k_total: float = 4.0  # shrinkage of the totals rating
    rho: float = 0.6  # fraction of the previous year's rating that goes in as the prior
    rho_total: float = 0.5
    half_life: float = 0.0  # weeks; 0 = all games weigh the same
    hfa0: float = 2.5  # prior for home-field advantage
    hfa_k: float = 40.0
    newcomer: float = -8.0  # prior for a team that just joined FBS
    fcs0: float = -22.0  # prior for the FCS pseudo-team if there is no previous year
    base_total0: float = 53.0
    base_total_k: float = 10.0


@dataclass
class SrsState:
    rating: np.ndarray  # points better than average (margin)
    pace: np.ndarray  # the team's contribution to the game total
    hfa: float
    base_total: float
    games: np.ndarray  # games used per team


class SrsModel:
    name = "srs"
    label = "SRS (schedule-adjusted margin)"

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
        ]  # the FCS average barely changes from one year to the next
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

        # ---- margin: m = R_home - R_away + hfa * (not neutral)
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

        # ---- total: t = base + S_home + S_away
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
        return {"model": self.name, **asdict(self.params)}


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
    """The literal version of SRS: rating = average margin + average opponent rating,
    repeated until it converges. It exists to document (and test) that `SrsModel.fit`
    reaches the same result by solving the system in one go."""
    idx = np.flatnonzero(mask & sa.completed)
    h, a = sa.h[idx], sa.a[idx]
    m = np.clip(sa.hp[idx] - sa.ap[idx], -cap, cap) - hfa * (~sa.neutral[idx])
    prior = np.zeros(sa.n) if prior is None else prior
    games = np.bincount(h, minlength=sa.n) + np.bincount(a, minlength=sa.n)
    rating = np.zeros(sa.n)
    for _ in range(iters):
        acc = np.zeros(sa.n)
        np.add.at(acc, h, m + rating[a])  # home team: its margin + the opponent's rating
        np.add.at(acc, a, -m + rating[h])  # away team: the same seen from its side
        new = (acc + k * prior) / np.maximum(games + k, 1e-12)
        new[games + k == 0] = 0.0
        # Averaging with the previous step keeps the iteration from oscillating.
        new = 0.5 * (new + rating)
        if np.max(np.abs(new - rating)) < tol:
            return new
        rating = new
    return rating
