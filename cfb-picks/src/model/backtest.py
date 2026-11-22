"""Walk-forward backtest: to predict week W, only weeks before W are used.

The model comparison table is printed with `python -m src.compare`.
"""

from __future__ import annotations

from collections.abc import Iterable
from itertools import pairwise

import numpy as np
import pandas as pd

from .data import SeasonArrays
from .prob import norm_cdf

_COLUMNS = (
    "season",
    "week",
    "game_id",
    "h",
    "a",
    "neutral",
    "fcs",
    "pred_margin",
    "pred_total",
    "margin",
    "total",
    "games_h",
    "games_a",
)


def walk_forward(
    model, seasons: dict[int, SeasonArrays], eval_seasons: Iterable[int] | None = None
) -> pd.DataFrame:
    """Predict every played week using only what came before it. Seasons outside
    `eval_seasons` are still run (they leave the prior for the next one) but are
    not evaluated."""
    eval_seasons = set(eval_seasons) if eval_seasons is not None else None
    cols: dict[str, list] = {k: [] for k in _COLUMNS}
    prev_state = prev_sa = None
    for season in sorted(seasons):
        sa = seasons[season]
        prior = model.prior(prev_state, prev_sa, sa)
        if eval_seasons is None or season in eval_seasons:
            for week in np.unique(sa.week[sa.completed]):
                test = sa.completed & (sa.week == week)
                state = model.fit(sa, sa.week < week, prior, now_week=int(week))
                h, a = sa.h[test], sa.a[test]
                margin, total = model.predict(state, h, a, sa.neutral[test])
                n = int(test.sum())
                cols["season"].append(np.full(n, season))
                cols["week"].append(np.full(n, week))
                cols["game_id"].append(sa.game_id[test])
                cols["h"].append(h)
                cols["a"].append(a)
                cols["neutral"].append(sa.neutral[test])
                cols["fcs"].append(sa.fcs[test])
                cols["pred_margin"].append(margin)
                cols["pred_total"].append(total)
                cols["margin"].append(sa.hp[test] - sa.ap[test])
                cols["total"].append(sa.hp[test] + sa.ap[test])
                cols["games_h"].append(state.games[h])
                cols["games_a"].append(state.games[a])
        if sa.completed.any():
            prev_state = model.fit(
                sa, np.ones(len(sa.week), dtype=bool), prior, now_week=int(sa.week.max()) + 1
            )
            prev_sa = sa
    if not cols["season"]:
        return pd.DataFrame(columns=list(cols))
    return pd.DataFrame({k: np.concatenate(v) for k, v in cols.items()})


def residual_sigma(df: pd.DataFrame) -> tuple[float, float]:
    """Real standard deviation of the error (RMSE) for margin and total."""
    sm = float(np.sqrt(np.mean((df["pred_margin"] - df["margin"]) ** 2)))
    tot = df.dropna(subset=["pred_total"])
    if len(tot):
        st = float(np.sqrt(np.mean((tot["pred_total"] - tot["total"]) ** 2)))
    else:
        st = float("nan")
    return sm, st


def metrics(df: pd.DataFrame, sigma_margin: float | None = None, big: float = 14.0) -> dict:
    """Backtest metrics. `sigma_margin` is the sigma used to turn margins into
    probabilities (for the Brier score); defaults to the set's own. `big` is the
    projected margin from which a team counts as a clear favorite.

    A tie (possible in the NFL) has no winner to pick: it is left out of the
    accuracy and counts as half a win in the Brier score."""
    if df.empty:
        return {"n": 0}
    err = df["pred_margin"] - df["margin"]
    pick_home = df["pred_margin"] >= 0
    home_won = df["margin"] > 0
    decided = df["margin"] != 0
    correct = (pick_home == home_won)[decided]
    clear = (df["pred_margin"].abs() >= big)[decided]
    sm, st = residual_sigma(df)
    sigma = sigma_margin or sm
    p_home = norm_cdf(df["pred_margin"].to_numpy() / sigma)
    outcome = np.where(decided, home_won, 0.5).astype(float)
    tot = df.dropna(subset=["pred_total"])
    tot_err = tot["pred_total"] - tot["total"]
    return {
        "n": len(df),
        "accuracy": float(correct.mean()) if len(correct) else None,
        "accuracy_big": float(correct[clear].mean()) if clear.any() else None,
        "n_big": int(clear.sum()),
        "big_margin": big,
        "mae_margin": float(err.abs().mean()),
        "rmse_margin": sm,
        "bias_margin": float(err.mean()),
        "mae_total": float(tot_err.abs().mean()) if len(tot) else None,
        "rmse_total": st if len(tot) else None,
        "bias_total": float(tot_err.mean()) if len(tot) else None,
        "brier": float(np.mean((p_home - outcome) ** 2)),
    }


def calibration(df: pd.DataFrame, sigma: float, step: float = 0.05, min_n: int = 10) -> list[dict]:
    """When the model gives its favorite X%, how often does that favorite win?

    A bin with fewer than `min_n` games is left out: three games are not a percentage."""
    if df.empty:
        return []
    df = df[df["margin"] != 0]  # a tie has no winner
    p_home = norm_cdf(df["pred_margin"].to_numpy() / sigma)
    p_fav = np.maximum(p_home, 1 - p_home)
    margin = df["margin"].to_numpy()
    fav_won = np.where(p_home >= 0.5, margin > 0, margin < 0)
    edges = np.arange(0.5, 1.0 + 1e-9, step)
    out = []
    for lo, hi in pairwise(edges):
        sel = (p_fav >= lo) & ((p_fav < hi) if hi < 1.0 - 1e-9 else (p_fav <= 1.0))
        if sel.sum() >= min_n:
            n = int(sel.sum())
            obs = float(fav_won[sel].mean())
            out.append(
                {
                    "from": round(float(lo), 2),
                    "to": round(float(hi), 2),
                    "n": n,
                    "predicted": float(p_fav[sel].mean()),
                    "observed": obs,
                    "std_error": float(np.sqrt(max(obs * (1 - obs), 1e-9) / n)),
                }
            )
    return out


def by_week(df: pd.DataFrame, sigma: float, big: float = 14.0) -> list[dict]:
    out = []
    for (season, week), grp in df.groupby(["season", "week"]):
        out.append({"season": int(season), "week": int(week), **metrics(grp, sigma, big)})
    return out


def fbs_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["fcs"]]
