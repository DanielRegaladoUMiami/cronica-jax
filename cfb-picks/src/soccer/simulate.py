"""Betting backtest for soccer: apply the PLAY / PASS / AVOID rule week by week
against real closing odds and record what would have happened.

It is walk-forward: ratings for a week only use earlier matches, and the
stability check is run with the data available at that time.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..model.prob import american_to_decimal
from .data import ODDS_COLUMNS
from .model import LeagueArrays
from .rules import SoccerRules, evaluate_match, settle

_LINES = ("total", "spread_home", "total_open")


def odds_row_to_dict(row) -> dict:
    """A row of soccer_odds_history.parquet -> dict with None for anything missing."""
    out = {}
    for field in ODDS_COLUMNS:
        value = getattr(row, field, None)
        if field == "provider":
            out[field] = None if value is None or pd.isna(value) else str(value)
        elif value is None or pd.isna(value):
            out[field] = None
        else:
            out[field] = float(value) if field in _LINES else int(value)
    return out


def drop_last_matches(
    la: LeagueArrays, mask: np.ndarray, cutoff: float, teams: tuple[int, ...], n: int = 2
) -> np.ndarray:
    """Copy of `mask` without each team's last n matches before `cutoff`."""
    out = mask.copy()
    played = mask & la.completed & (la.day < cutoff)
    for team in teams:
        idx = np.flatnonzero(played & ((la.h == team) | (la.a == team)))
        out[idx[-n:]] = False
    return out


def simulate_bets(
    model, leagues: dict[str, LeagueArrays], odds: pd.DataFrame, rules: SoccerRules, eval_seasons
) -> pd.DataFrame:
    odds_by_id = {r.match_id: odds_row_to_dict(r) for r in odds[odds["has_odds"]].itertuples()}
    eval_seasons = list(eval_seasons)
    rows: list[dict] = []
    for league, la in leagues.items():
        everything = np.ones(len(la.h), dtype=bool)
        scope = la.completed & np.isin(la.season, eval_seasons)
        state = None
        for week in np.unique(la.week[scope]):
            cutoff = float(week)
            state = model.fit(la, everything, cutoff, warm=state)
            for g in np.flatnonzero(scope & (la.week == week)):
                quote = odds_by_id.get(la.match_id[g])
                if quote is None:
                    continue
                h, a = int(la.h[g]), int(la.a[g])
                lam_h, lam_a = model.expected_goals(state, [h], [a])
                grid = model.grid(float(lam_h[0]), float(lam_a[0]))
                alt = model.fit(
                    la, drop_last_matches(la, everything, cutoff, (h, a)), cutoff, warm=state
                )
                alt_h, alt_a = model.expected_goals(alt, [h], [a])
                ev = evaluate_match(
                    grid=grid,
                    grid_alt=model.grid(float(alt_h[0]), float(alt_a[0])),
                    odds=quote,
                    rules=rules,
                    min_games_played=int(min(state.games[h], state.games[a])),
                )
                for market in ("ml", "total", "spread"):
                    pick = ev[market]["pick"]
                    side = ev[market].get(pick) if pick else None
                    if not side or side["edge"] is None:
                        continue
                    line = None
                    if market == "total":
                        line = ev["total"]["line"]
                    elif market == "spread":
                        line = ev["spread"]["home_line"]
                    result = settle(market, pick, line, la.hg[g], la.ag[g])
                    if result > 0:
                        profit = american_to_decimal(side["odds"]) - 1.0
                    else:
                        profit = 0.0 if result == 0 else -1.0
                    rows.append(
                        {
                            "league": league,
                            "season": int(la.season[g]),
                            "week": cutoff,
                            "match_id": la.match_id[g],
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
                            # first red flag, as a short code ("outsider", "thin data", ...)
                            "flag": side["reasons"][0].split(":")[0] if side["reasons"] else "",
                            "stake": side["stake"],
                            "result": result,
                            "profit": profit,
                        }
                    )
    return pd.DataFrame(rows)
