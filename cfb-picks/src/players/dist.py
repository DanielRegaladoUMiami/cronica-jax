"""From a projection to a probability: how likely is "250+ passing yards"?

A projection is an average. A bet asks about a threshold, so what is needed is
the spread of real results around projections of that size. Two tools:

- Yardage: the empirical distribution of (actual / projected) in the backtest,
  kept separately for small and large projections (a 15-yard projection is far
  more erratic, in proportion, than a 250-yard one). Nothing is assumed about its
  shape: zero-yard games and long touchdowns are in there as often as they happened.
- Counts (attempts, receptions, touchdowns, interceptions): a negative binomial,
  i.e. a Poisson with extra spread, with the spread measured in the backtest.

Both are fitted on backtest rows, so the probabilities carry the model's real
error, not an optimistic one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

YARD_STATS = ("pass_yds", "rush_yds", "rec_yds")
COUNT_STATS = ("pass_att", "pass_cmp", "pass_td", "pass_int", "rush_att", "targets", "rec", "td")
GRID = np.linspace(0.01, 0.99, 99)
# Below these projections the ratio actual/projected is too wild to be useful.
MIN_MEAN = {"pass_yds": 80.0, "rush_yds": 6.0, "rec_yds": 6.0}


@dataclass
class RatioTable:
    """Quantiles of actual/projected, by size of the projection."""

    means: list[float]  # average projection of each bucket, increasing
    quantiles: list[list[float]]  # per bucket: the ratio at each point of GRID

    def cdf(self, ratio: float, mean: float) -> float:
        """P(actual / projected <= ratio) for a projection of size `mean`."""
        values = [float(np.interp(ratio, q, GRID, left=0.0, right=1.0)) for q in self.quantiles]
        if len(values) == 1:
            return values[0]
        return float(
            np.interp(math.log(max(mean, 1e-9)), [math.log(m) for m in self.means], values)
        )

    def quantile(self, p: float, mean: float) -> float:
        """The ratio below which a fraction `p` of results fall."""
        values = [float(np.interp(p, GRID, q)) for q in self.quantiles]
        if len(values) == 1:
            return values[0]
        return float(
            np.interp(math.log(max(mean, 1e-9)), [math.log(m) for m in self.means], values)
        )


def fit_ratio_table(proj: np.ndarray, actual: np.ndarray, buckets: int = 5) -> RatioTable | None:
    proj, actual = np.asarray(proj, dtype=float), np.asarray(actual, dtype=float)
    if len(proj) < 60:
        return None
    buckets = max(1, min(buckets, len(proj) // 120))
    order = np.argsort(proj)
    means, quantiles = [], []
    for chunk in np.array_split(order, buckets):
        ratio = np.maximum(actual[chunk], 0.0) / proj[chunk]
        means.append(float(proj[chunk].mean()))
        quantiles.append([float(v) for v in np.quantile(ratio, GRID)])
    return RatioTable(means, quantiles)


def fit_dispersion(proj: np.ndarray, actual: np.ndarray) -> float:
    """Negative binomial size k from the backtest: variance = mean + mean^2 / k.

    A large k means "almost Poisson". Found by moments, which is stable for counts.
    """
    proj, actual = np.asarray(proj, dtype=float), np.asarray(actual, dtype=float)
    extra = float(((actual - proj) ** 2 - proj).sum())
    denom = float((proj**2).sum())
    if extra <= 0 or denom <= 0:
        return 1e6
    return max(denom / extra, 0.3)


def nb_sf(n: int, mean: float, k: float) -> float:
    """P(X >= n) for a negative binomial with this mean and size k."""
    if n <= 0:
        return 1.0
    if mean <= 0:
        return 0.0
    log_p0 = k * math.log(k / (k + mean))
    log_q = math.log(mean / (k + mean))
    total = 0.0
    for j in range(n):
        total += math.exp(
            math.lgamma(j + k) - math.lgamma(k) - math.lgamma(j + 1) + log_p0 + j * log_q
        )
    return float(min(max(1.0 - total, 0.0), 1.0))


@dataclass
class Distributions:
    """Everything needed to turn projections into probabilities, per statistic."""

    ratios: dict[str, RatioTable]
    dispersion: dict[str, float]

    def prob_at_least(self, stat: str, threshold: float, mean: float) -> float | None:
        """P(stat >= threshold). `threshold` is the whole number a "N+" bet names.

        Returns None when the projection is too small for the tables to say anything.
        """
        if mean is None or not np.isfinite(mean) or mean <= 0:
            return None
        if stat in self.ratios:
            if mean < MIN_MEAN.get(stat, 0.0):
                return None
            # Yards are whole numbers: "at least 250" is "more than 249.5".
            return 1.0 - self.ratios[stat].cdf((threshold - 0.5) / mean, mean)
        if stat in self.dispersion:
            return nb_sf(math.ceil(threshold - 1e-9), mean, self.dispersion[stat])
        return None

    def prob_over(self, stat: str, line: float, mean: float) -> float | None:
        """P(stat > line) for a sportsbook line such as 249.5."""
        return self.prob_at_least(stat, math.floor(line) + 1, mean)

    def quantile(self, stat: str, p: float, mean: float) -> float | None:
        if stat in self.ratios and mean >= MIN_MEAN.get(stat, 0.0):
            return self.ratios[stat].quantile(p, mean) * mean
        return None

    def mean_for_median(self, stat: str, median: float) -> float | None:
        """The projection whose median equals a sportsbook line (which is a median)."""
        if stat not in self.ratios or median <= 0:
            return None
        mean = median
        for _ in range(30):
            ratio = self.ratios[stat].quantile(0.5, mean)
            if ratio <= 0:
                return None
            mean = median / ratio
        return mean


def fit(backtest: pd.DataFrame, eligible) -> Distributions:
    """Fit every table on backtest rows. `eligible(df, stat)` picks the rows of a statistic."""
    ratios: dict[str, RatioTable] = {}
    dispersion: dict[str, float] = {}
    for stat in YARD_STATS:
        if f"p_{stat}" not in backtest:
            continue
        sub = backtest[eligible(backtest, stat) & (backtest[f"p_{stat}"] >= MIN_MEAN[stat])]
        table = fit_ratio_table(sub[f"p_{stat}"].to_numpy(), sub[f"a_{stat}"].to_numpy())
        if table is not None:
            ratios[stat] = table
    for stat in COUNT_STATS:
        if f"p_{stat}" not in backtest:
            continue
        sub = backtest[eligible(backtest, stat) & (backtest[f"p_{stat}"] > 0)]
        if len(sub) >= 60:
            dispersion[stat] = fit_dispersion(
                sub[f"p_{stat}"].to_numpy(), sub[f"a_{stat}"].to_numpy()
            )
    return Distributions(ratios, dispersion)


def calibration(
    dist: Distributions, backtest: pd.DataFrame, eligible, stat: str, thresholds, step: float = 0.1
) -> pd.DataFrame:
    """Predicted vs observed frequency of "stat >= threshold", in probability bins."""
    sub = backtest[eligible(backtest, stat)]
    preds, hits = [], []
    for mean, actual in zip(sub[f"p_{stat}"], sub[f"a_{stat}"]):
        for threshold in thresholds:
            p = dist.prob_at_least(stat, threshold, mean)
            if p is None or p < 0.03 or p > 0.97:
                continue
            preds.append(p)
            hits.append(1.0 if actual >= threshold else 0.0)
    if not preds:
        return pd.DataFrame()
    frame = pd.DataFrame({"p": preds, "hit": hits})
    frame["bin"] = (frame["p"] / step).astype(int).clip(upper=int(1 / step) - 1) * step
    out = frame.groupby("bin").agg(
        n=("hit", "size"), predicted=("p", "mean"), observed=("hit", "mean")
    )
    out["brier"] = ((frame["p"] - frame["hit"]) ** 2).mean()
    return out.reset_index()
