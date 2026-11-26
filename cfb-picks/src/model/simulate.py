"""Betting backtest: apply the PLAY / PASS / AVOID rule week by week against
real closing lines and record what would have happened.

It is walk-forward like the prediction backtest: ratings for week W only use
weeks before W, and the stability check uses the data available at that time.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ..data.odds import build_markets
from .data import SeasonArrays
from .prob import american_to_decimal
from .rules import Rules, evaluate_game

_ODDS_FIELDS = (
    "spread_home",
    "spread_home_odds",
    "spread_away_odds",
    "total",
    "over_odds",
    "under_odds",
    "ml_home",
    "ml_away",
    "spread_home_open",
    "total_open",
    "ml_home_open",
    "ml_away_open",
)
_FLOAT_FIELDS = ("spread_home", "total", "spread_home_open", "total_open")


def odds_row_to_dict(row) -> dict:
    """A row of odds_history.parquet -> dict with None for anything missing."""
    out = {"provider": getattr(row, "provider", None)}
    for field in _ODDS_FIELDS:
        value = getattr(row, field, None)
        if value is None or pd.isna(value):
            out[field] = None
        else:
            out[field] = float(value) if field in _FLOAT_FIELDS else int(value)
    return out


def drop_last_games(
    sa: SeasonArrays, mask: np.ndarray, teams: tuple[int, ...], n: int = 2
) -> np.ndarray:
    """Copy of `mask` without each team's last n games (arrays are in chronological order)."""
    out = mask.copy()
    for team in teams:
        played = np.flatnonzero(mask & sa.completed & ((sa.h == team) | (sa.a == team)))
        out[played[-n:]] = False
    return out


def settle(market: str, side: str, line: float | None, margin: float, total: float) -> int:
    """1 = win, 0 = push, -1 = loss. `margin` is home minus away."""
    if market == "ml":
        value = margin if side == "home" else -margin
    elif market == "spread":
        value = (margin + line) if side == "home" else -(margin + line)
    else:
        value = (total - line) if side == "over" else (line - total)
    return int(np.sign(value))


def simulate_bets(
    model,
    seasons: dict[int, SeasonArrays],
    odds: pd.DataFrame,
    sigma_m: float,
    sigma_t: float,
    rules: Rules,
    eval_seasons,
) -> pd.DataFrame:
    odds_by_id = {row.game_id: odds_row_to_dict(row) for row in odds[odds["has_odds"]].itertuples()}
    eval_seasons = set(eval_seasons)
    rows: list[dict] = []
    prev_state = prev_sa = None
    for season in sorted(seasons):
        sa = seasons[season]
        prior = model.prior(prev_state, prev_sa, sa)
        if season in eval_seasons:
            for week in np.unique(sa.week[sa.completed]):
                train = sa.completed & (sa.week < week)
                test = [
                    g
                    for g in np.flatnonzero(sa.completed & (sa.week == week) & ~sa.fcs)
                    if sa.game_id[g] in odds_by_id
                ]
                if not test:
                    continue
                state = model.fit(sa, train, prior, now_week=int(week))
                for g in test:
                    h, a, neutral = int(sa.h[g]), int(sa.a[g]), bool(sa.neutral[g])
                    m, t = model.predict(state, [h], [a], [neutral])
                    alt = model.fit(
                        sa, drop_last_games(sa, train, (h, a)), prior, now_week=int(week)
                    )
                    m2, t2 = model.predict(alt, [h], [a], [neutral])
                    markets = build_markets(None, odds_by_id[sa.game_id[g]])
                    total_pred = None if math.isnan(float(t[0])) else float(t[0])
                    ev = evaluate_game(
                        margin=float(m[0]),
                        total=total_pred,
                        margin_alt=float(m2[0]),
                        total_alt=None if total_pred is None else float(t2[0]),
                        markets=markets,
                        sigma_m=sigma_m,
                        sigma_t=sigma_t,
                        rules=rules,
                        min_games_played=int(min(state.games[h], state.games[a])),
                    )
                    margin, total = float(sa.hp[g] - sa.ap[g]), float(sa.hp[g] + sa.ap[g])
                    for market in ("ml", "spread", "total"):
                        pick = ev[market]["lean"]  # the bets follow the model's disagreements
                        side = ev[market].get(pick) if pick else None
                        if not side or side["edge"] is None:
                            continue
                        line = None
                        if market == "spread":
                            line = markets["spread"]["home_line"]
                        elif market == "total":
                            line = markets["total"]["line"]
                        result = settle(market, pick, line, margin, total)
                        if result > 0:
                            profit = american_to_decimal(side["odds"]) - 1.0
                        else:
                            profit = 0.0 if result == 0 else -1.0
                        rows.append(
                            {
                                "season": season,
                                "week": int(week),
                                "game_id": sa.game_id[g],
                                "market": market,
                                "side": pick,
                                "line": line,
                                "odds": side["odds"],
                                "p_model": side["p_model"],
                                "p_market": side["p_market"],
                                "edge": side["edge"],
                                "ev": side["ev"],
                                "stable": side["stable"],
                                "label": side["label"],
                                "stake": side["stake"],
                                "result": result,
                                "profit": profit,
                                "gap": ev["total_gap"] if market == "total" else ev["margin_gap"],
                            }
                        )
        if sa.completed.any():
            prev_state = model.fit(
                sa, np.ones(len(sa.week), dtype=bool), prior, now_week=int(sa.week.max()) + 1
            )
            prev_sa = sa
    return pd.DataFrame(rows)


MIN_BETS = 200  # decided bets a market needs before its trust is measured
GAP_BUCKETS = ((0.0, 0.03), (0.03, 0.06), (0.06, 0.10), (0.10, 0.15), (0.15, 1.0))


def _logit(p) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-4, 1.0 - 1e-4)
    return np.log(p / (1.0 - p))


def _logistic(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Logistic regression by Newton's method: coefficients and their standard errors."""
    beta = np.zeros(x.shape[1])
    for _ in range(50):
        p = 1.0 / (1.0 + np.exp(-(x @ beta)))
        hessian = x.T @ (x * (p * (1.0 - p))[:, None]) + 1e-8 * np.eye(x.shape[1])
        step = np.linalg.solve(hessian, x.T @ (y - p))
        beta = beta + step
        if np.abs(step).max() < 1e-9:
            break
    p = 1.0 / (1.0 + np.exp(-(x @ beta)))
    cov = np.linalg.inv(x.T @ (x * (p * (1.0 - p))[:, None]) + 1e-8 * np.eye(x.shape[1]))
    return beta, np.sqrt(np.diag(cov))


def fit_trust(bets: pd.DataFrame) -> dict:
    """When the model and the market disagreed, how much of the disagreement came true?

    Per market, a logistic regression of the result on the market's log-odds and on
    the gap between the model's log-odds and the market's. The weight of the gap is
    the answer: 1 = the model's number was right, 0 = the market's was. `trust` is
    that weight clipped to [0, 1]. A market with fewer than MIN_BETS decided bets
    gets 0: with no evidence, the market's number is the one to believe.

    `buckets` shows the same thing without a formula: by size of the disagreement,
    what the model said, what the market said and what happened.
    """
    out = {}
    for market in ("ml", "spread", "total"):
        d = bets[(bets["market"] == market) & (bets["result"] != 0)] if len(bets) else bets
        entry = {"n": len(d), "weight": None, "error": None, "trust": 0.0, "buckets": []}
        if len(d) >= MIN_BETS:
            y = (d["result"] > 0).to_numpy(dtype=float)
            market_logit = _logit(d["p_market"])
            gap = _logit(d["p_model"]) - market_logit
            beta, se = _logistic(np.column_stack([market_logit, gap]), y)
            entry["weight"], entry["error"] = float(beta[1]), float(se[1])
            entry["trust"] = float(min(max(beta[1], 0.0), 1.0))
            diff = d["p_model"] - d["p_market"]
            for lo, hi in GAP_BUCKETS:
                sub = d[(diff >= lo) & (diff < hi)]
                if len(sub) >= 20:
                    entry["buckets"].append(
                        {
                            "lo": lo,
                            "hi": hi,
                            "n": len(sub),
                            "model": float(sub["p_model"].mean()),
                            "market": float(sub["p_market"].mean()),
                            "actual": float((sub["result"] > 0).mean()),
                        }
                    )
        out[market] = entry
    return out


def record(bets: pd.DataFrame) -> dict:
    """Summary of a set of flat one-unit bets."""
    n = len(bets)
    if n == 0:
        return {
            "n": 0,
            "won": 0,
            "lost": 0,
            "push": 0,
            "win_pct": None,
            "units": 0.0,
            "roi": None,
            "roi_error": None,
        }
    won, lost = int((bets["result"] > 0).sum()), int((bets["result"] < 0).sum())
    profit = bets["profit"].to_numpy(dtype=float)
    return {
        "n": n,
        "won": won,
        "lost": lost,
        "push": n - won - lost,
        "win_pct": won / max(won + lost, 1),
        "units": float(profit.sum()),
        "roi": float(profit.mean()),
        "roi_error": float(profit.std(ddof=1) / math.sqrt(n)) if n > 1 else None,
    }
