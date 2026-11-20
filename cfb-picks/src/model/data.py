"""Turn games.parquet into per-season numpy arrays (fast for the backtest).

All non-FBS teams are grouped into one "FCS" pseudo-team: we only see 1-2 games
against FBS from each of them, so an individual rating would be noise.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FCS = "FCS"


@dataclass
class SeasonArrays:
    season: int
    team_ids: list[str]  # index -> team_id; the last one is the FCS pseudo-team
    names: dict[str, str]  # team_id -> short name
    game_id: np.ndarray
    week: np.ndarray  # week_index (postseason = last regular-season week + 1)
    h: np.ndarray  # home team index
    a: np.ndarray  # away team index
    hp: np.ndarray  # home team points (nan if not played yet)
    ap: np.ndarray
    neutral: np.ndarray
    completed: np.ndarray
    fcs: np.ndarray  # True if either team is not FBS

    def __post_init__(self) -> None:
        self.index = {tid: i for i, tid in enumerate(self.team_ids)}

    @property
    def n(self) -> int:
        return len(self.team_ids)

    @property
    def fcs_index(self) -> int:
        return self.n - 1

    @property
    def margin(self) -> np.ndarray:
        return self.hp - self.ap

    @property
    def total(self) -> np.ndarray:
        return self.hp + self.ap

    def idx(self, team_id: str) -> int:
        return self.index.get(str(team_id), self.fcs_index)


def make_season(games: pd.DataFrame, season: int) -> SeasonArrays:
    g = games[games["season"] == season].sort_values(["date", "game_id"]).reset_index(drop=True)
    fbs = sorted(
        set(g.loc[g["home_fbs"], "home_id"]) | set(g.loc[g["away_fbs"], "away_id"]), key=int
    )
    team_ids = [*fbs, FCS]
    index = {tid: i for i, tid in enumerate(team_ids)}
    names = {**dict(zip(g["home_id"], g["home_team"])), **dict(zip(g["away_id"], g["away_team"]))}
    names[FCS] = "FCS (average)"
    fcs_i = len(team_ids) - 1
    return SeasonArrays(
        season=season,
        team_ids=team_ids,
        names=names,
        game_id=g["game_id"].to_numpy(dtype=object),
        week=g["week_index"].to_numpy(dtype=int),
        h=np.array([index.get(t, fcs_i) for t in g["home_id"]], dtype=int),
        a=np.array([index.get(t, fcs_i) for t in g["away_id"]], dtype=int),
        hp=g["home_points"].astype("float64").to_numpy(),
        ap=g["away_points"].astype("float64").to_numpy(),
        neutral=g["neutral_site"].to_numpy(dtype=bool),
        completed=g["completed"].to_numpy(dtype=bool),
        fcs=g["fcs_game"].to_numpy(dtype=bool),
    )


def make_seasons(games: pd.DataFrame) -> dict[int, SeasonArrays]:
    return {int(s): make_season(games, int(s)) for s in sorted(games["season"].unique())}


def align(
    prev_values: np.ndarray, prev: SeasonArrays, cur: SeasonArrays, default: float
) -> np.ndarray:
    """Map a ratings vector from the previous season onto the current season's team order.
    Teams new to FBS get `default`."""
    out = np.full(cur.n, default, dtype=float)
    for i, tid in enumerate(cur.team_ids):
        j = prev.index.get(tid)
        if j is not None:
            out[i] = prev_values[j]
    return out
