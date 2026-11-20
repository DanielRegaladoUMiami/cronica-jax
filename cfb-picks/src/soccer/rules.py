"""From a scoreline grid and a set of prices to PLAY / PASS / AVOID, for soccer.

Three markets are evaluated: the three-way moneyline (home / draw / away),
the goals total and the handicap. The labels mean the same as in football:

PLAY   = edge of at least 5%, positive expected value at the real price, no red flags.
AVOID  = an edge on paper, but one of the red flags below is up.
PASS   = not enough edge.

The red flags were set on the 2022-2025 betting backtest, where each of them
marked picks that lost far more than the bookmaker's margin:
- the edge is 12 points or more (the market strongly disagrees with the model);
- a moneyline pick on a side the market gives less than 45% (the model
  overrates outsiders, and outsiders carry the largest bookmaker margin);
- the pick flips without each team's last 2 matches, or a team has too few matches.

Line movement since the open is shown on the site but is not a flag: in the two
seasons with opening prices it did not separate good picks from bad ones.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from ..model.prob import american_to_prob
from ..model.rules import _side
from .model import handicap_probs, outcome_probs, total_probs

UNSTABLE = "unstable: the pick flips without each team's last 2 matches"


@dataclass(frozen=True)
class SoccerRules:
    edge_min: float = 0.05  # minimum edge (model chance minus no-vig market chance)
    kelly_fraction: float = 0.25  # quarter Kelly
    stake_cap: float = 0.02  # never more than 2% of bankroll on one play
    gap_edge: float = 0.12  # an edge this large means the market strongly disagrees: AVOID
    ml_min_market: float = 0.45  # moneyline: AVOID sides the market gives less than this
    min_games: int = 6  # matches a team needs inside the model's window

    def as_dict(self) -> dict:
        return asdict(self)


def fair_three_way(home: int | None, draw: int | None, away: int | None):
    """No-vig probabilities of a three-way price, or None if a price is missing."""
    if home is None or draw is None or away is None:
        return None
    raw = np.array([american_to_prob(home), american_to_prob(draw), american_to_prob(away)])
    return raw / raw.sum()


def fair_two_way(a: int | None, b: int | None):
    if a is None or b is None:
        return None
    pa, pb = american_to_prob(a), american_to_prob(b)
    return pa / (pa + pb), pb / (pa + pb)


def is_main_line(odds_a: int | None, odds_b: int | None) -> bool:
    """True when both sides are priced close enough to even to be the main line.

    ESPN's archive sometimes lists an alternate line (say a 4.5 total at +950 / -5000).
    A model edge on a line like that is not a bet anyone would place."""
    if odds_a is None or odds_b is None:
        return True  # nothing to judge by; the caller decides what to do without prices
    return all(0.28 <= american_to_prob(o) <= 0.76 for o in (odds_a, odds_b))


def is_quarter_line(line: float) -> bool:
    """Lines such as 2.25 or -0.75 split the stake in two; they are not evaluated."""
    return abs(line * 2 - round(line * 2)) > 1e-9


def _no_push(win: float, push: float) -> float:
    """Chance of winning given that the bet is not refunded."""
    return win / max(1.0 - push, 1e-9)


def _quote(odds: int | None, source: str) -> dict | None:
    return None if odds is None else {"odds": int(odds), "source": source}


def _best(pair: dict, keys: tuple[str, ...]) -> str | None:
    sides = [(k, pair[k]) for k in keys if pair.get(k)]
    if not sides:
        return None
    with_edge = [(k, s) for k, s in sides if s["edge"] is not None]
    if with_edge:
        return max(with_edge, key=lambda ks: ks[1]["edge"])[0]
    return max(sides, key=lambda ks: ks[1]["p_model"])[0]


def evaluate_match(
    *,
    grid: np.ndarray,
    grid_alt: np.ndarray | None,
    odds: dict | None,
    rules: SoccerRules,
    min_games_played: int = 99,
    source: str = "espn",
) -> dict:
    """Evaluate every bettable side of a match.

    `grid` is the scoreline probability grid; `grid_alt` the same grid refitted
    without each team's last 2 matches (the stability check); `odds` a
    normalized ESPN quote (see `data.parse_odds_entry`) or None.
    """
    odds = odds or {}
    few = min_games_played < rules.min_games
    thin = f"thin data: a team has played fewer than {rules.min_games} matches in the league"

    def alerts_for(edge: float | None, p_market: float | None = None) -> list[str]:
        alerts = []
        if few:
            alerts.append(thin)
        if edge is not None and edge >= rules.gap_edge:
            alerts.append(
                f"the market disagrees: an edge of {edge * 100:.0f} points is rarely real"
            )
        if p_market is not None and p_market < rules.ml_min_market:
            alerts.append(
                f"outsider: the market gives this side only {p_market * 100:.0f}%, and the"
                " model overrates outsiders"
            )
        return alerts

    # ---------------------------------------------------------------- moneyline
    p_model = outcome_probs(grid)
    p_alt = outcome_probs(grid_alt) if grid_alt is not None else (None, None, None)
    fair = fair_three_way(odds.get("ml_home"), odds.get("ml_draw"), odds.get("ml_away"))
    fair_open = fair_three_way(
        odds.get("ml_home_open"), odds.get("ml_draw_open"), odds.get("ml_away_open")
    )
    ml: dict = {}
    for i, key in enumerate(("home", "draw", "away")):
        p_market = None if fair is None else float(fair[i])
        edge = None if p_market is None else p_model[i] - p_market
        ml[key] = _side(
            p_model[i],
            p_alt[i],
            p_market,
            _quote(odds.get(f"ml_{key}"), source),
            rules,
            alerts_for(edge, p_market),
            unstable=UNSTABLE,
        )
    ml["pick"] = _best(ml, ("home", "draw", "away"))
    ml["devig"] = None if fair is None else {"method": "three_way", "source": source}

    # -------------------------------------------------------------------- total
    total: dict = {"over": None, "under": None, "pick": None, "line": None, "devig": None}
    line = odds.get("total")
    main = is_main_line(odds.get("over_odds"), odds.get("under_odds"))
    if line is not None and main and not is_quarter_line(line):
        over, push, under = total_probs(grid, line)
        alt = total_probs(grid_alt, line) if grid_alt is not None else None
        fair2 = fair_two_way(odds.get("over_odds"), odds.get("under_odds"))
        for key, win, idx in (("over", over, 0), ("under", under, 2)):
            p_cond = _no_push(win, push)
            p_cond_alt = None if alt is None else _no_push(alt[idx], alt[1])
            p_market = None if fair2 is None else float(fair2[0 if key == "over" else 1])
            edge = None if p_market is None else p_cond - p_market
            total[key] = _side(
                p_cond,
                p_cond_alt,
                p_market,
                _quote(odds.get(f"{key}_odds"), source),
                rules,
                alerts_for(edge),
                unstable=UNSTABLE,
            )
        total["pick"] = _best(total, ("over", "under"))
        total["line"] = float(line)
        total["devig"] = None if fair2 is None else {"method": "two_sided", "source": source}

    # ----------------------------------------------------------------- handicap
    spread: dict = {"home": None, "away": None, "pick": None, "home_line": None, "devig": None}
    home_line = odds.get("spread_home")
    main = is_main_line(odds.get("spread_home_odds"), odds.get("spread_away_odds"))
    if home_line is not None and main and not is_quarter_line(home_line):
        cover, push, miss = handicap_probs(grid, home_line)
        alt = handicap_probs(grid_alt, home_line) if grid_alt is not None else None
        fair2 = fair_two_way(odds.get("spread_home_odds"), odds.get("spread_away_odds"))
        for key, win, idx in (("home", cover, 0), ("away", miss, 2)):
            p_cond = _no_push(win, push)
            p_cond_alt = None if alt is None else _no_push(alt[idx], alt[1])
            p_market = None if fair2 is None else float(fair2[0 if key == "home" else 1])
            edge = None if p_market is None else p_cond - p_market
            spread[key] = _side(
                p_cond,
                p_cond_alt,
                p_market,
                _quote(odds.get(f"spread_{key}_odds"), source),
                rules,
                alerts_for(edge),
                unstable=UNSTABLE,
            )
        spread["pick"] = _best(spread, ("home", "away"))
        spread["home_line"] = float(home_line)
        spread["devig"] = None if fair2 is None else {"method": "two_sided", "source": source}

    return {
        "ml": ml,
        "total": total,
        "spread": spread,
        "market_probs": None if fair is None else [float(x) for x in fair],
        "market_probs_open": None if fair_open is None else [float(x) for x in fair_open],
    }


def settle(market: str, side: str, line: float | None, home_goals: float, away_goals: float) -> int:
    """1 = win, 0 = push (stake refunded), -1 = loss."""
    gd, goals = home_goals - away_goals, home_goals + away_goals
    if market == "ml":
        outcome = "home" if gd > 0 else "draw" if gd == 0 else "away"
        return 1 if side == outcome else -1
    if market == "total":
        value = goals - line if side == "over" else line - goals
    else:
        value = gd + line if side == "home" else -(gd + line)
    return int(np.sign(round(value, 9)))
