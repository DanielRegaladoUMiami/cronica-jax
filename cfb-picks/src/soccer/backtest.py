"""Walk-forward backtest for the soccer models.

To predict a week, a model only sees matches played before that week started.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .model import LeagueArrays, outcome_probs, score_matrix, total_probs

MIN_BIN = 20  # a calibration bin with fewer cases is noise, not evidence

_COLUMNS = (
    "league",
    "season",
    "week",
    "match_id",
    "h",
    "a",
    "lam_home",
    "lam_away",
    "p_home",
    "p_draw",
    "p_away",
    "p_over25",
    "hg",
    "ag",
    "games_h",
    "games_a",
)


def predict_match(model, state, h: int, a: int) -> dict:
    """Expected goals and the main probabilities for one match."""
    lam_h, lam_a = model.expected_goals(state, [h], [a])
    lam_h, lam_a = float(lam_h[0]), float(lam_a[0])
    if hasattr(model, "probs"):  # Elo has no scoreline grid: its 1X2 comes from the rating gap
        p_home, p_draw, p_away = (float(x[0]) for x in model.probs(model.diff(state, [h], [a])))
        lam_h, lam_a = max(lam_h, 0.15), max(lam_a, 0.15)
        over = total_probs(score_matrix(lam_h, lam_a), 2.5)[0]
    else:
        grid = model.grid(lam_h, lam_a)
        p_home, p_draw, p_away = outcome_probs(grid)
        over = total_probs(grid, 2.5)[0]
    return {
        "lam_home": lam_h,
        "lam_away": lam_a,
        "p_home": p_home,
        "p_draw": p_draw,
        "p_away": p_away,
        "p_over25": over,
    }


def walk_forward(
    model, leagues: dict[str, LeagueArrays], eval_seasons: Iterable[int] | None = None
) -> pd.DataFrame:
    """Predict every played week of every league using only earlier matches."""
    eval_seasons = set(eval_seasons) if eval_seasons is not None else None
    rows: list[dict] = []
    for league, la in leagues.items():
        everything = np.ones(len(la.h), dtype=bool)
        in_scope = la.completed.copy()
        if eval_seasons is not None:
            in_scope &= np.isin(la.season, list(eval_seasons))
        state = None
        for week in np.unique(la.week[in_scope]):
            state = model.fit(la, everything, float(week), warm=state)
            for g in np.flatnonzero(in_scope & (la.week == week)):
                h, a = int(la.h[g]), int(la.a[g])
                rows.append(
                    {
                        "league": league,
                        "season": int(la.season[g]),
                        "week": float(week),
                        "match_id": la.match_id[g],
                        "h": h,
                        "a": a,
                        **predict_match(model, state, h, a),
                        "hg": float(la.hg[g]),
                        "ag": float(la.ag[g]),
                        "games_h": int(state.games[h]),
                        "games_a": int(state.games[a]),
                    }
                )
    return pd.DataFrame(rows, columns=list(_COLUMNS))


def outcome_index(df: pd.DataFrame) -> np.ndarray:
    """0 = home win, 1 = draw, 2 = away win."""
    gd = (df["hg"] - df["ag"]).to_numpy()
    return np.where(gd > 0, 0, np.where(gd == 0, 1, 2))


def metrics(df: pd.DataFrame) -> dict:
    """Forecast quality. RPS and log loss are both 'lower is better'."""
    if df.empty:
        return {"n": 0}
    probs = np.clip(df[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float), 1e-9, 1.0)
    y = outcome_index(df)
    onehot = np.eye(3)[y]
    cum_p, cum_o = np.cumsum(probs, axis=1), np.cumsum(onehot, axis=1)
    gd = (df["hg"] - df["ag"]).to_numpy()
    total = (df["hg"] + df["ag"]).to_numpy()
    lam_gd = (df["lam_home"] - df["lam_away"]).to_numpy()
    lam_total = (df["lam_home"] + df["lam_away"]).to_numpy()
    over = (total > 2.5).astype(float)
    return {
        "n": len(df),
        "accuracy": float((probs.argmax(axis=1) == y).mean()),
        "rps": float(np.mean(0.5 * ((cum_p[:, :2] - cum_o[:, :2]) ** 2).sum(axis=1))),
        "log_loss": float(-np.mean(np.log(probs[np.arange(len(y)), y]))),
        "mae_margin": float(np.abs(lam_gd - gd).mean()),
        "mae_total": float(np.abs(lam_total - total).mean()),
        "bias_total": float((lam_total - total).mean()),
        "brier_over": float(np.mean((df["p_over25"].to_numpy() - over) ** 2)),
        "draw_predicted": float(probs[:, 1].mean()),
        "draw_observed": float((y == 1).mean()),
    }


def calibration(df: pd.DataFrame, step: float = 0.1) -> list[dict]:
    """Pool the three probabilities of every match: when the model says X%, how often does it happen?"""
    if df.empty:
        return []
    probs = df[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float).ravel()
    hit = np.eye(3)[outcome_index(df)].ravel()
    out = []
    edges = np.arange(0.0, 1.0 + 1e-9, step)
    for lo, hi in itertools.pairwise(edges):
        sel = (probs >= lo) & ((probs < hi) if hi < 1.0 - 1e-9 else (probs <= 1.0))
        if sel.sum() >= MIN_BIN:
            n = int(sel.sum())
            obs = float(hit[sel].mean())
            out.append(
                {
                    "from": round(float(lo), 2),
                    "to": round(float(hi), 2),
                    "n": n,
                    "predicted": float(probs[sel].mean()),
                    "observed": obs,
                    "std_error": float(np.sqrt(max(obs * (1 - obs), 1e-9) / n)),
                }
            )
    return out


def by_week(df: pd.DataFrame) -> list[dict]:
    out = []
    for week, grp in df.groupby("week"):
        start = pd.Timestamp(float(week) * 86400, unit="s", tz="UTC")
        out.append({"week": start.strftime("%Y-%m-%d"), **metrics(grp)})
    return out
