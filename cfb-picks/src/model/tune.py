"""Tune the hyperparameters on past seasons and save them to params.json.

Usage:
    python -m src.model.tune

Protocol (so the evaluation stays honest):
- The first season is a warm-up: it only leaves the prior for the next one.
- Hyperparameters are chosen by minimizing the walk-forward error from the
  second season through the last completed one (FBS vs FBS games only).
- The current season is NOT touched here: it is evaluated later, out of sample.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, replace

import numpy as np

from .. import config
from ..data.build import load_games
from .backtest import fbs_only, metrics, residual_sigma, walk_forward
from .blend import BlendModel
from .data import make_seasons
from .elo import EloModel, EloParams
from .srs import SrsModel, SrsParams


def _rmse(df, col_pred="pred_margin", col_true="margin") -> float:
    return float(np.sqrt(np.mean((df[col_pred] - df[col_true]) ** 2)))


def tune(current_season: int | None = None, log=print) -> dict:
    current_season = current_season or config.current_season()
    games = load_games()
    seasons = {s: sa for s, sa in make_seasons(games).items() if s < current_season}
    years = sorted(seasons)
    eval_years = years[1:]  # the first one is the warm-up
    log(f"Tuning on {eval_years[0]}-{eval_years[-1]} (warm-up: {years[0]})")

    def score_srs(params: SrsParams) -> tuple[float, float]:
        df = fbs_only(walk_forward(SrsModel(params), seasons, eval_years))
        return _rmse(df), _rmse(df, "pred_total", "total")

    # 1) margin: blowout cap, shrinkage and weight of last season
    best, best_score = SrsParams(), float("inf")
    grid = itertools.product(
        (24.0, 28.0, 35.0, 45.0, 99.0), (1.0, 2.0, 3.0, 4.0, 6.0), (0.55, 0.7, 0.85, 1.0)
    )
    for cap, k, rho in grid:
        params = replace(SrsParams(), cap=cap, k=k, rho=rho)
        s, _ = score_srs(params)
        if s < best_score:
            best, best_score = params, s
    log(f"  SRS margin: cap={best.cap:g} k={best.k:g} rho={best.rho:g} -> RMSE {best_score:.3f}")

    # 2) recency weighting
    for half_life in (5.0, 8.0, 12.0):
        params = replace(best, half_life=half_life)
        s, _ = score_srs(params)
        if s < best_score - 0.005:  # only if it clearly helps: fewer parameters is better
            best, best_score = params, s
    log(f"  SRS recency: half_life={best.half_life:g} -> RMSE {best_score:.3f}")

    # 3) totals (a system independent from the margin one)
    best_total = float("inf")
    best_t = best
    for k_total, rho_total in itertools.product((2.0, 4.0, 6.0, 9.0, 13.0), (0.3, 0.5, 0.7, 0.9)):
        params = replace(best, k_total=k_total, rho_total=rho_total)
        _, t = score_srs(params)
        if t < best_total:
            best_t, best_total = params, t
    best = best_t
    log(
        f"  SRS totals: k_total={best.k_total:g} rho_total={best.rho_total:g}"
        f" -> total RMSE {best_total:.3f}"
    )

    # 4) Elo. The scale (Elo points per point of margin) has a closed-form solution.
    best_elo, best_elo_score = EloParams(), float("inf")
    grid = itertools.product(
        (25.0, 32.0, 40.0, 50.0, 62.0), (40.0, 55.0, 70.0), (0.6, 0.7, 0.8, 0.9, 1.0)
    )
    for k, hfa, carry in grid:
        params = EloParams(k=k, hfa=hfa, carry=carry, scale=1.0)
        df = fbs_only(walk_forward(EloModel(params), seasons, eval_years))
        diff, margin = df["pred_margin"].to_numpy(), df["margin"].to_numpy()
        scale = float((diff @ diff) / (diff @ margin))
        s = float(np.sqrt(np.mean((diff / scale - margin) ** 2)))
        if s < best_elo_score:
            best_elo, best_elo_score = replace(params, scale=scale), s
    log(
        f"  Elo: k={best_elo.k:g} hfa={best_elo.hfa:g} carry={best_elo.carry:g}"
        f" scale={best_elo.scale:.1f} -> RMSE {best_elo_score:.3f}"
    )

    # 5) blend weight by least squares
    srs_df = fbs_only(walk_forward(SrsModel(best), seasons, eval_years))
    elo_df = fbs_only(walk_forward(EloModel(best_elo), seasons, eval_years))
    s = srs_df["pred_margin"].to_numpy()
    e = elo_df["pred_margin"].to_numpy()
    m = srs_df["margin"].to_numpy()
    weight = float(np.clip(((m - e) @ (s - e)) / ((s - e) @ (s - e)), 0.0, 1.0))
    blend = BlendModel(SrsModel(best), EloModel(best_elo), weight)
    blend_df = fbs_only(walk_forward(blend, seasons, eval_years))
    log(f"  Blend: SRS weight={weight:.2f} -> RMSE {_rmse(blend_df):.3f}")

    history = {}
    for name, df in (("srs", srs_df), ("elo", elo_df), ("blend", blend_df)):
        sm, st = residual_sigma(df)
        keep = ("n", "mae_margin", "mae_total", "accuracy")
        history[name] = {
            "margin": sm,
            "total": None if np.isnan(st) else st,
            **{k: v for k, v in metrics(df).items() if k in keep},
        }

    out = {
        "tuned_on": f"{eval_years[0]}-{eval_years[-1]}",
        "srs": asdict(best),
        "elo": asdict(best_elo),
        "srs_blend_weight": weight,
        "history": history,
    }
    previous = json.loads(config.PARAMS_PATH.read_text()) if config.PARAMS_PATH.exists() else {}
    for key in ("selected", "selected_by"):  # keep the model choice made by `src.compare --select`
        if key in previous:
            out[key] = previous[key]
    config.PARAMS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    log(f"Saved to {config.PARAMS_PATH.relative_to(config.ROOT)}")
    return out


if __name__ == "__main__":
    tune()
