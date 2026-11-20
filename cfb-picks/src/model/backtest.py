"""Backtest walk-forward: para predecir la semana W solo se usan semanas < W.

La tabla comparativa de modelos se imprime con `python -m src.compare`.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterable

import numpy as np
import pandas as pd

from .data import SeasonArrays
from .prob import norm_cdf


def walk_forward(
    model, seasons: dict[int, SeasonArrays], eval_seasons: Iterable[int] | None = None
) -> pd.DataFrame:
    """Predice cada semana jugada usando solo lo anterior. Las temporadas que no
    están en `eval_seasons` se recorren igual (dejan el prior para la siguiente)
    pero no se evalúan."""
    eval_seasons = set(eval_seasons) if eval_seasons is not None else None
    cols: dict[str, list] = {
        k: []
        for k in (
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
    }
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
    """Desviación estándar real del error (RMSE) de margen y de total."""
    sm = float(np.sqrt(np.mean((df["pred_margin"] - df["margin"]) ** 2)))
    tot = df.dropna(subset=["pred_total"])
    st = (
        float(np.sqrt(np.mean((tot["pred_total"] - tot["total"]) ** 2)))
        if len(tot)
        else float("nan")
    )
    return sm, st


def metrics(df: pd.DataFrame, sigma_margin: float | None = None) -> dict:
    """Métricas del backtest. `sigma_margin` es la sigma con la que se calculan las
    probabilidades (para Brier); si falta se usa la del propio conjunto."""
    if df.empty:
        return {"n": 0}
    err = df["pred_margin"] - df["margin"]
    pick_home = df["pred_margin"] >= 0
    home_won = df["margin"] > 0
    correct = pick_home == home_won
    big = df["pred_margin"].abs() >= 14
    sm, st = residual_sigma(df)
    sigma = sigma_margin or sm
    p_home = norm_cdf(df["pred_margin"].to_numpy() / sigma)
    tot = df.dropna(subset=["pred_total"])
    return {
        "n": len(df),
        "acierto": float(correct.mean()),
        "acierto_14": float(correct[big].mean()) if big.any() else None,
        "n_14": int(big.sum()),
        "mae_margen": float(err.abs().mean()),
        "rmse_margen": sm,
        "sesgo_margen": float(err.mean()),
        "mae_total": float((tot["pred_total"] - tot["total"]).abs().mean()) if len(tot) else None,
        "rmse_total": st if len(tot) else None,
        "sesgo_total": float((tot["pred_total"] - tot["total"]).mean()) if len(tot) else None,
        "brier": float(np.mean((p_home - home_won.to_numpy(dtype=float)) ** 2)),
    }


def calibration(df: pd.DataFrame, sigma: float, step: float = 0.05) -> list[dict]:
    """Cuando el modelo dice X% para su favorito, ¿con qué frecuencia gana?"""
    if df.empty:
        return []
    p_home = norm_cdf(df["pred_margin"].to_numpy() / sigma)
    p_fav = np.maximum(p_home, 1 - p_home)
    fav_won = np.where(p_home >= 0.5, df["margin"].to_numpy() > 0, df["margin"].to_numpy() < 0)
    edges = np.arange(0.5, 1.0 + 1e-9, step)
    out = []
    for lo, hi in itertools.pairwise(edges):
        sel = (p_fav >= lo) & ((p_fav < hi) if hi < 1.0 - 1e-9 else (p_fav <= 1.0))
        if sel.any():
            n = int(sel.sum())
            obs = float(fav_won[sel].mean())
            out.append(
                {
                    "desde": round(float(lo), 2),
                    "hasta": round(float(hi), 2),
                    "n": n,
                    "predicho": float(p_fav[sel].mean()),
                    "observado": obs,
                    "error_est": float(np.sqrt(max(obs * (1 - obs), 1e-9) / n)),
                }
            )
    return out


def by_week(df: pd.DataFrame, sigma: float) -> list[dict]:
    out = []
    for (season, week), grp in df.groupby(["season", "week"]):
        m = metrics(grp, sigma)
        out.append({"temporada": int(season), "semana": int(week), **m})
    return out


def fbs_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[~df["fcs"]]
