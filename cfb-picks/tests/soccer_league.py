"""A synthetic league with known team strengths, shared by the soccer tests."""

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
from src.soccer.data import MATCH_COLUMNS, week_start

DAY0 = datetime(2023, 8, 11, 19, 0, tzinfo=timezone.utc)


def synthetic_league(n_teams=8, seasons=(2023, 2024, 2025), seed=7, league="syn.1", newcomer=None):
    """A double round-robin per season with known attack/defense strengths.

    `newcomer` = (season, team index): that team has no matches before that season."""
    rng = np.random.default_rng(seed)
    attack = np.linspace(0.45, -0.45, n_teams)
    defense = np.linspace(0.35, -0.35, n_teams)
    rows, match_id = [], 0
    for s_i, season in enumerate(seasons):
        teams = [
            t
            for t in range(n_teams)
            if not (newcomer and t == newcomer[1] and season < newcomer[0])
        ]
        pairs = [(h, a) for h in teams for a in teams if h != a]
        order = rng.permutation(len(pairs))
        for k, idx in enumerate(order):
            h, a = pairs[idx]
            date = DAY0 + timedelta(days=365 * s_i + 7 * (k // (len(teams) // 2)), hours=int(k % 3))
            lam_h = np.exp(0.25 + 0.25 + attack[h] - defense[a])
            lam_a = np.exp(0.25 + attack[a] - defense[h])
            match_id += 1
            rows.append(
                {
                    "match_id": f"{league}-{match_id}",
                    "league": league,
                    "season": season,
                    "date": date,
                    "week_start": week_start(date),
                    "status": "final",
                    "completed": True,
                    "home_id": str(100 + h),
                    "home_team": f"Team {h}",
                    "home_short": f"Team {h}",
                    "home_abbr": f"T{h}",
                    "home_goals": int(rng.poisson(lam_h)),
                    "away_id": str(100 + a),
                    "away_team": f"Team {a}",
                    "away_short": f"Team {a}",
                    "away_abbr": f"T{a}",
                    "away_goals": int(rng.poisson(lam_a)),
                }
            )
    df = pd.DataFrame(rows, columns=MATCH_COLUMNS)
    for col in ("date", "week_start"):
        df[col] = pd.to_datetime(df[col], utc=True)
    return df, attack, defense
