"""Goal models for soccer.

The main one is a Poisson ratings model (the Dixon-Coles family):

    home goals ~ Poisson(exp(base + home_edge + attack[home] - defense[away]))
    away goals ~ Poisson(exp(base + attack[away] - defense[home]))

Every team gets an attack and a defense rating, fitted together on all past
matches of its league with three refinements:

1. Recent matches count more: a match's weight halves every `half_life` days.
2. Shrinkage: ratings are pulled toward a prior with the strength of `k`
   matches, so a team with few games does not get an extreme rating. A team
   that is new to the league starts below average (`promoted`).
3. A low-score correction (`rho`): plain Poisson gives too few 0-0 and 1-1
   draws, so those scores are nudged up and 1-0 / 0-1 down.

From the two expected-goal numbers comes the probability of every scoreline,
and from that grid everything else: home / draw / away, totals and handicaps.

Two simpler models are kept for comparison: season scoring averages, and Elo.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DAY = 86400.0
MAX_GOALS = 10


@dataclass
class LeagueArrays:
    """One league's matches as numpy arrays, in chronological order."""

    league: str
    team_ids: list[str]
    names: dict[str, str]
    match_id: np.ndarray
    season: np.ndarray
    day: np.ndarray  # match time in days since the Unix epoch
    week: np.ndarray  # the day the match's week starts (Tuesday), same unit
    h: np.ndarray
    a: np.ndarray
    hg: np.ndarray  # nan until the match is played
    ag: np.ndarray
    completed: np.ndarray

    def __post_init__(self) -> None:
        self.index = {tid: i for i, tid in enumerate(self.team_ids)}
        self._design: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.team_ids)

    def design(self) -> np.ndarray:
        """Design matrix of the Poisson model for every match, built once.

        Row g is the home team's goals in match g and row M + g the away team's.
        Columns: base, home edge, one attack per team, one defense per team."""
        if self._design is None:
            m, n = len(self.h), self.n
            X = np.zeros((2 * m, 2 + 2 * n))
            rows = np.arange(m)
            X[:, 0] = 1.0
            X[rows, 1] = 1.0
            X[rows, 2 + self.h] = 1.0
            X[rows, 2 + n + self.a] = -1.0
            X[m + rows, 2 + self.a] = 1.0
            X[m + rows, 2 + n + self.h] = -1.0
            self._design = X
        return self._design


def make_league(matches: pd.DataFrame, league: str) -> LeagueArrays:
    m = (
        matches[matches["league"] == league]
        .sort_values(["date", "match_id"])
        .reset_index(drop=True)
    )
    team_ids = sorted(set(m["home_id"]) | set(m["away_id"]), key=int)
    index = {tid: i for i, tid in enumerate(team_ids)}
    names = {**dict(zip(m["home_id"], m["home_team"])), **dict(zip(m["away_id"], m["away_team"]))}
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    return LeagueArrays(
        league=league,
        team_ids=team_ids,
        names=names,
        match_id=m["match_id"].to_numpy(dtype=object),
        season=m["season"].to_numpy(dtype=int),
        day=((m["date"] - epoch).dt.total_seconds() / DAY).to_numpy(),
        week=((m["week_start"] - epoch).dt.total_seconds() / DAY).to_numpy(),
        h=np.array([index[t] for t in m["home_id"]], dtype=int),
        a=np.array([index[t] for t in m["away_id"]], dtype=int),
        hg=m["home_goals"].astype("float64").to_numpy(),
        ag=m["away_goals"].astype("float64").to_numpy(),
        completed=m["completed"].to_numpy(dtype=bool),
    )


def make_leagues(matches: pd.DataFrame) -> dict[str, LeagueArrays]:
    return {lg: make_league(matches, lg) for lg in sorted(matches["league"].unique())}


# ------------------------------------------------------------- score matrix
def poisson_pmf(lam: float, max_goals: int = MAX_GOALS) -> np.ndarray:
    k = np.arange(max_goals + 1)
    log_p = -lam + k * math.log(max(lam, 1e-12)) - np.array([math.lgamma(i + 1) for i in k])
    return np.exp(log_p)


def score_matrix(lam_home: float, lam_away: float, rho: float = 0.0) -> np.ndarray:
    """P[i, j] = probability the home team scores i and the away team j."""
    grid = np.outer(poisson_pmf(lam_home), poisson_pmf(lam_away))
    if rho:
        grid[0, 0] *= max(1 - lam_home * lam_away * rho, 1e-6)
        grid[0, 1] *= max(1 + lam_home * rho, 1e-6)
        grid[1, 0] *= max(1 + lam_away * rho, 1e-6)
        grid[1, 1] *= max(1 - rho, 1e-6)
    return grid / grid.sum()


_DIFF = np.subtract.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))  # home minus away
_SUM = np.add.outer(np.arange(MAX_GOALS + 1), np.arange(MAX_GOALS + 1))


def outcome_probs(grid: np.ndarray) -> tuple[float, float, float]:
    """(home win, draw, away win)."""
    return float(grid[_DIFF > 0].sum()), float(grid[_DIFF == 0].sum()), float(grid[_DIFF < 0].sum())


def total_probs(grid: np.ndarray, line: float) -> tuple[float, float, float]:
    """(over, push, under) for a goals line. A push only exists on whole-number lines."""
    return (
        float(grid[_SUM > line + 1e-9].sum()),
        float(grid[np.abs(_SUM - line) <= 1e-9].sum()),
        float(grid[_SUM < line - 1e-9].sum()),
    )


def handicap_probs(grid: np.ndarray, home_line: float) -> tuple[float, float, float]:
    """(home covers, push, away covers) for a home handicap such as -1.5 or +0.5."""
    adj = _DIFF + home_line
    return (
        float(grid[adj > 1e-9].sum()),
        float(grid[np.abs(adj) <= 1e-9].sum()),
        float(grid[adj < -1e-9].sum()),
    )


def most_likely_score(grid: np.ndarray) -> tuple[int, int]:
    i, j = np.unravel_index(int(np.argmax(grid)), grid.shape)
    return int(i), int(j)


# ------------------------------------------------------------ Poisson model
@dataclass(frozen=True)
class PoissonParams:
    half_life: float = 240.0  # days for a match's weight to halve
    k: float = 5.0  # shrinkage toward the prior, in matches
    promoted: float = -0.2  # prior attack and defense (log scale) for a team new to the league
    rho: float = -0.06  # low-score correction
    window: float = 1100.0  # days of history used (about three seasons)
    new_after: float = 300.0  # no match older than this many days = new to the league
    home_edge0: float = 0.2  # prior for the home edge (log scale)
    home_edge_k: float = 20.0


@dataclass
class PoissonState:
    base: float
    home_edge: float
    attack: np.ndarray
    defense: np.ndarray
    games: np.ndarray  # matches of each team inside the window
    theta: np.ndarray  # raw parameter vector, reused to warm-start the next fit


class PoissonModel:
    name = "poisson"
    label = "Poisson ratings (attack and defense)"

    def __init__(self, params: PoissonParams | None = None) -> None:
        self.params = params or PoissonParams()

    def fit(
        self, la: LeagueArrays, mask: np.ndarray, cutoff: float, warm: PoissonState | None = None
    ) -> PoissonState:
        """Fit on the completed matches in `mask` that were played before `cutoff` (in days)."""
        p = self.params
        n = la.n
        idx = np.flatnonzero(
            mask & la.completed & (la.day < cutoff) & (la.day >= cutoff - p.window)
        )
        h, a = la.h[idx], la.a[idx]
        hg, ag = la.hg[idx], la.ag[idx]
        age = cutoff - la.day[idx]
        w = 0.5 ** (age / p.half_life)
        games = np.bincount(h, minlength=n) + np.bincount(a, minlength=n)
        # A team with no match older than `new_after` days is new to the league (promoted).
        old = age > p.new_after
        seasoned = (np.bincount(h[old], minlength=n) + np.bincount(a[old], minlength=n)) > 0
        any_old = bool(old.any())
        prior = np.where(seasoned | (not any_old), 0.0, p.promoted)

        m = len(idx)
        size = 2 + 2 * n
        if m == 0:
            theta = np.concatenate([[math.log(1.35), p.home_edge0], prior, prior])
            return PoissonState(theta[0], theta[1], prior.copy(), prior.copy(), games, theta)

        # Two rows per match: the home team's goals and the away team's goals.
        X = la.design()[np.concatenate([idx, len(la.h) + idx])]
        y = np.concatenate([hg, ag])
        ww = np.concatenate([w, w])

        ridge = np.concatenate([[1e-6, p.home_edge_k], np.full(2 * n, p.k)])
        target = np.concatenate([[0.0, p.home_edge0], prior, prior])
        if warm is not None and len(warm.theta) == size:
            theta = warm.theta.copy()
        else:
            mean_goals = max(float((ww * y).sum() / ww.sum()), 0.2)
            theta = np.concatenate([[math.log(mean_goals), p.home_edge0], prior, prior])

        for _ in range(40):
            lam = np.exp(np.clip(X @ theta, -6.0, 3.0))
            grad = X.T @ (ww * (y - lam)) - ridge * (theta - target)
            hess = (X * (ww * lam)[:, None]).T @ X
            hess[np.diag_indices(size)] += ridge
            step = np.clip(np.linalg.solve(hess, grad), -1.0, 1.0)
            theta = theta + step
            if float(np.max(np.abs(step))) < 1e-7:
                break
        return PoissonState(
            base=float(theta[0]),
            home_edge=float(theta[1]),
            attack=theta[2 : 2 + n].copy(),
            defense=theta[2 + n :].copy(),
            games=games,
            theta=theta,
        )

    def expected_goals(self, s: PoissonState, h, a) -> tuple[np.ndarray, np.ndarray]:
        h, a = np.asarray(h), np.asarray(a)
        lam_home = np.exp(s.base + s.home_edge + s.attack[h] - s.defense[a])
        lam_away = np.exp(s.base + s.attack[a] - s.defense[h])
        return lam_home, lam_away

    def grid(self, lam_home: float, lam_away: float) -> np.ndarray:
        return score_matrix(lam_home, lam_away, self.params.rho)

    def describe(self) -> dict:
        return {"model": self.name, **asdict(self.params)}


# --------------------------------------------------- baseline: season averages
@dataclass
class AverageState:
    attack: np.ndarray  # goals scored relative to the league average
    defense: np.ndarray  # goals conceded relative to the league average
    home_avg: float
    away_avg: float
    games: np.ndarray


class AverageModel:
    """The spreadsheet approach: this season's goals for and against, relative to the league."""

    name = "averages"
    label = "Baseline (this season's scoring averages)"

    def __init__(self, shrink: float = 2.0, window: float = 330.0) -> None:
        self.shrink = shrink  # matches at league average added to every team
        self.window = window  # days: roughly the current season

    def fit(self, la: LeagueArrays, mask: np.ndarray, cutoff: float, warm=None) -> AverageState:
        n = la.n
        idx = np.flatnonzero(
            mask & la.completed & (la.day < cutoff) & (la.day >= cutoff - self.window)
        )
        h, a, hg, ag = la.h[idx], la.a[idx], la.hg[idx], la.ag[idx]
        games = np.bincount(h, minlength=n) + np.bincount(a, minlength=n)
        home_avg = float(hg.mean()) if len(idx) >= 20 else 1.5
        away_avg = float(ag.mean()) if len(idx) >= 20 else 1.2
        per_team = (home_avg + away_avg) / 2
        scored = np.bincount(h, weights=hg, minlength=n) + np.bincount(a, weights=ag, minlength=n)
        conceded = np.bincount(h, weights=ag, minlength=n) + np.bincount(a, weights=hg, minlength=n)
        attack = (scored + self.shrink * per_team) / ((games + self.shrink) * per_team)
        defense = (conceded + self.shrink * per_team) / ((games + self.shrink) * per_team)
        return AverageState(attack, defense, home_avg, away_avg, games)

    def expected_goals(self, s: AverageState, h, a) -> tuple[np.ndarray, np.ndarray]:
        h, a = np.asarray(h), np.asarray(a)
        return s.home_avg * s.attack[h] * s.defense[a], s.away_avg * s.attack[a] * s.defense[h]

    def grid(self, lam_home: float, lam_away: float) -> np.ndarray:
        return score_matrix(lam_home, lam_away, 0.0)

    def describe(self) -> dict:
        return {"model": self.name, "shrink": self.shrink}


# ------------------------------------------------------------------------ Elo
@dataclass(frozen=True)
class EloParams:
    k: float = 18.0
    home: float = 60.0  # home advantage in Elo points
    carry: float = 0.85  # share of a rating kept between seasons
    newcomer: float = 1400.0
    draw_width: float = 0.75  # how wide the "draw zone" is (fitted)
    scale: float = 190.0  # Elo points per unit of the outcome scale (fitted)
    goals_per_100: float = 0.33  # expected goal difference per 100 Elo points (fitted)


@dataclass
class EloState:
    rating: np.ndarray
    games: np.ndarray
    avg_total: float


class EloModel:
    """Elo with a goal-difference multiplier. Draws come from an ordered-logit link."""

    name = "elo"
    label = "Elo with goal difference"

    def __init__(self, params: EloParams | None = None) -> None:
        self.params = params or EloParams()

    def fit(self, la: LeagueArrays, mask: np.ndarray, cutoff: float, warm=None) -> EloState:
        p = self.params
        rating = np.full(la.n, np.nan)
        games = np.zeros(la.n, dtype=int)
        season = None
        totals: list[float] = []
        for g in np.flatnonzero(mask & la.completed & (la.day < cutoff)):
            if la.season[g] != season:
                season = la.season[g]
                known = ~np.isnan(rating)
                rating[known] = 1500 + p.carry * (rating[known] - 1500)
            h, a = la.h[g], la.a[g]
            for t in (h, a):
                if np.isnan(rating[t]):
                    rating[t] = 1500.0 if season == la.season[0] else p.newcomer
            diff = rating[h] - rating[a] + p.home
            expected = 1.0 / (1.0 + 10.0 ** (-diff / 400.0))
            gd = la.hg[g] - la.ag[g]
            result = 1.0 if gd > 0 else 0.5 if gd == 0 else 0.0
            mult = 1.0 if abs(gd) <= 1 else 1.5 if abs(gd) == 2 else (11 + abs(gd)) / 8
            delta = p.k * mult * (result - expected)
            rating[h] += delta
            rating[a] -= delta
            games[h] += 1
            games[a] += 1
            totals.append(la.hg[g] + la.ag[g])
        rating = np.where(np.isnan(rating), p.newcomer, rating)
        avg_total = float(np.mean(totals[-380:])) if totals else 2.7
        return EloState(rating, games, avg_total)

    def diff(self, s: EloState, h, a) -> np.ndarray:
        return s.rating[np.asarray(h)] - s.rating[np.asarray(a)] + self.params.home

    def probs(self, diff: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        p = self.params
        x = np.asarray(diff, dtype=float) / p.scale
        home = 1.0 / (1.0 + np.exp(-(x - p.draw_width)))
        away = 1.0 / (1.0 + np.exp(-(-x - p.draw_width)))
        return home, 1.0 - home - away, away

    def expected_goals(self, s: EloState, h, a) -> tuple[np.ndarray, np.ndarray]:
        gd = self.diff(s, h, a) / 100.0 * self.params.goals_per_100
        return (s.avg_total + gd) / 2, (s.avg_total - gd) / 2

    def describe(self) -> dict:
        return {"model": self.name, **asdict(self.params)}
