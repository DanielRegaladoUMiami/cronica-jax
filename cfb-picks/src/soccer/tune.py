"""Tune the soccer models on past seasons and save the result to params.json.

Usage:
    python -m src.soccer.tune

Protocol, so the evaluation stays honest:
- The first season only warms the ratings up.
- Settings are chosen by the walk-forward log loss of the home/draw/away
  forecast from the second season through the last completed one.
- The current season is not touched here: it is scored later, out of sample.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import asdict, replace

import numpy as np

from . import config
from .backtest import metrics, outcome_index, walk_forward
from .data import load_matches
from .model import (
    AverageModel,
    EloModel,
    EloParams,
    PoissonModel,
    PoissonParams,
    make_leagues,
    outcome_probs,
    score_matrix,
)


def _log_loss_for_rho(df, rho: float) -> float:
    y = outcome_index(df)
    out = 0.0
    for lam_h, lam_a, k in zip(df["lam_home"], df["lam_away"], y, strict=True):
        out -= np.log(max(outcome_probs(score_matrix(lam_h, lam_a, rho))[k], 1e-9))
    return out / len(df)


def _fit_elo_link(diff: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Scale and draw width of the ordered-logit link, by a coarse-to-fine grid search."""
    best = (float("inf"), 190.0, 0.75)
    for scale in np.arange(120.0, 321.0, 20.0):
        for width in np.arange(0.4, 1.21, 0.05):
            x = diff / scale
            home = 1 / (1 + np.exp(-(x - width)))
            away = 1 / (1 + np.exp(-(-x - width)))
            probs = np.clip(np.column_stack([home, 1 - home - away, away]), 1e-9, 1)
            loss = float(-np.mean(np.log(probs[np.arange(len(y)), y])))
            if loss < best[0]:
                best = (loss, float(scale), float(width))
    return best[1], best[2]


def tune(current_season: int | None = None, log=print) -> dict:
    current_season = current_season or config.current_season()
    matches = load_matches()
    matches = matches[matches["season"] < current_season]
    leagues = make_leagues(matches)
    years = sorted(int(s) for s in matches["season"].unique())
    eval_years = years[1:]
    log(f"Tuning on {eval_years[0]}-{eval_years[-1]} (warm-up: {years[0]})")

    def score(params: PoissonParams):
        df = walk_forward(PoissonModel(params), leagues, eval_years)
        return metrics(df)["log_loss"], df

    # 1) how fast old matches fade, and how hard ratings are pulled to the prior
    best, best_loss, best_df = None, float("inf"), None
    base = replace(PoissonParams(), rho=0.0)
    for half_life, k in itertools.product((150.0, 240.0, 365.0, 550.0), (2.0, 5.0, 10.0)):
        params = replace(base, half_life=half_life, k=k)
        loss, df = score(params)
        if loss < best_loss:
            best, best_loss, best_df = params, loss, df
    log(f"  Poisson: half_life={best.half_life:g} k={best.k:g} -> log loss {best_loss:.4f}")

    # 2) where a newly promoted team starts
    for promoted in (0.0, -0.1, -0.3):
        params = replace(best, promoted=promoted)
        loss, df = score(params)
        if loss < best_loss - 1e-5:
            best, best_loss, best_df = params, loss, df
    log(f"  Poisson: promoted={best.promoted:g} -> log loss {best_loss:.4f}")

    # 3) the low-score correction only changes the scoreline grid, not the ratings
    rho_losses = {rho: _log_loss_for_rho(best_df, rho) for rho in (0.0, -0.03, -0.06, -0.09, -0.12)}
    rho = min(rho_losses, key=rho_losses.get)
    best = replace(best, rho=rho)
    log(f"  Poisson: rho={rho:g} -> log loss {rho_losses[rho]:.4f}")

    # 4) Elo: update speed, home advantage and carry-over, then the link to probabilities
    best_elo, best_elo_loss = EloParams(), float("inf")
    for k, home, carry in itertools.product((12.0, 18.0, 26.0), (40.0, 60.0, 80.0), (0.75, 0.9)):
        params = EloParams(k=k, home=home, carry=carry)
        model = EloModel(params)
        frames = []
        for la in leagues.values():
            everything = np.ones(len(la.h), dtype=bool)
            scope = la.completed & np.isin(la.season, eval_years)
            for week in np.unique(la.week[scope]):
                state = model.fit(la, everything, float(week))
                g = np.flatnonzero(scope & (la.week == week))
                frames.append((model.diff(state, la.h[g], la.a[g]), la.hg[g] - la.ag[g]))
        diff = np.concatenate([f[0] for f in frames])
        gd = np.concatenate([f[1] for f in frames])
        y = np.where(gd > 0, 0, np.where(gd == 0, 1, 2))
        scale, width = _fit_elo_link(diff, y)
        x = diff / scale
        home_p = 1 / (1 + np.exp(-(x - width)))
        away_p = 1 / (1 + np.exp(-(-x - width)))
        probs = np.clip(np.column_stack([home_p, 1 - home_p - away_p, away_p]), 1e-9, 1)
        loss = float(-np.mean(np.log(probs[np.arange(len(y)), y])))
        if loss < best_elo_loss:
            slope = float((diff @ gd) / (diff @ diff)) * 100.0
            best_elo = replace(params, scale=scale, draw_width=width, goals_per_100=slope)
            best_elo_loss = loss
    log(
        f"  Elo: k={best_elo.k:g} home={best_elo.home:g} carry={best_elo.carry:g}"
        f" scale={best_elo.scale:g} draw_width={best_elo.draw_width:.2f}"
        f" -> log loss {best_elo_loss:.4f}"
    )

    history = {}
    for name, model in (
        ("poisson", PoissonModel(best)),
        ("elo", EloModel(best_elo)),
        ("averages", AverageModel()),
    ):
        m = metrics(walk_forward(model, leagues, eval_years))
        history[name] = {k: m[k] for k in ("n", "accuracy", "rps", "log_loss", "mae_total")}
        log(
            f"  {name:9s} accuracy {m['accuracy']:.3f} · RPS {m['rps']:.4f} · log loss {m['log_loss']:.4f}"
        )

    out = {
        "tuned_on": f"{eval_years[0]}-{eval_years[-1]}",
        "poisson": asdict(best),
        "elo": asdict(best_elo),
        "history": history,
    }
    previous = json.loads(config.PARAMS_PATH.read_text()) if config.PARAMS_PATH.exists() else {}
    for key in ("selected", "selected_by"):
        if key in previous:
            out[key] = previous[key]
    config.PARAMS_PATH.write_text(json.dumps(out, indent=2) + "\n")
    log(f"Saved to {config.PARAMS_PATH.relative_to(config.base.ROOT)}")
    return out


if __name__ == "__main__":
    tune()
