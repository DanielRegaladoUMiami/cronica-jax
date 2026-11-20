"""Convierte games.parquet en arreglos numpy por temporada (rápidos para el backtest).

Todos los equipos que no son FBS se agrupan en un pseudo-equipo "FCS": de cada
uno solo vemos 1-2 juegos contra FBS, así que un rating individual sería ruido.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FCS = "FCS"


@dataclass
class SeasonArrays:
    season: int
    team_ids: list[str]  # índice -> team_id; el último es el pseudo-equipo FCS
    names: dict[str, str]  # team_id -> nombre corto
    game_id: np.ndarray
    week: np.ndarray  # week_index (postemporada = última semana regular + 1)
    h: np.ndarray  # índice del local
    a: np.ndarray  # índice del visitante
    hp: np.ndarray  # puntos del local (nan si no se ha jugado)
    ap: np.ndarray
    neutral: np.ndarray
    completed: np.ndarray
    fcs: np.ndarray  # True si alguno de los dos no es FBS

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
    names[FCS] = "FCS (promedio)"
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
    """Lleva un vector de ratings de la temporada anterior al orden de equipos de la actual.
    Los equipos nuevos en FBS reciben `default`."""
    out = np.full(cur.n, default, dtype=float)
    for i, tid in enumerate(cur.team_ids):
        j = prev.index.get(tid)
        if j is not None:
            out[i] = prev_values[j]
    return out
