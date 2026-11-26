"""Walk-forward backtest of the player projections.

For every week already played, each player is projected with only what was
known before that week, and the projection is compared with what he did. The
yardstick is the simplest thing a bettor would do: the player's average per
game so far this season. If the model is not better than that, it says so.

Usage:
    python -m src.players.backtest            # error by statistic, model vs season average
    python -m src.players.backtest --ablation # what the opponent and the absences add
"""

from __future__ import annotations

import argparse
from dataclasses import replace

import numpy as np
import pandas as pd

from .. import config
from ..data.build import load_games
from ..data.odds_history import load_odds_history
from . import fetch
from .model import (
    STATS,
    Params,
    abs_week,
    build_priors,
    fit_team_effects,
    infer_positions,
    player_estimates,
    prepare,
    project_team,
)

# Which rows count for each statistic: decided with numbers known BEFORE the game.
PASS_STATS = ("pass_att", "pass_cmp", "pass_yds", "pass_td", "pass_int")
RUSH_STATS = ("rush_att", "rush_yds")
REC_STATS = ("targets", "rec", "rec_yds")
MIN_CARRIES = 4.0
MIN_TARGETS = 2.5
# A teammate counts as "missing" if he played for the team in the last weeks and had a role.
MISSING_WEEKS = 3
MISSING_TSHARE = 0.10
MISSING_CSHARE = 0.15
MISSING_GAMES = 1.2  # recency-weighted games: about two recent ones
START_ATTEMPTS = 15  # a quarterback with this many passes started (or played most of) the game


def implied_points(games: pd.DataFrame, odds: pd.DataFrame) -> dict[tuple[str, str], float]:
    """(game, team) -> points the betting line expects from that team."""
    if odds is None or odds.empty:
        return {}
    df = games.merge(odds[["game_id", "spread_home", "total"]], on="game_id", how="inner")
    df = df[df["spread_home"].notna() & df["total"].notna()]
    out: dict[tuple[str, str], float] = {}
    for gid, home, away, spread, total in zip(
        df["game_id"], df["home_id"], df["away_id"], df["spread_home"], df["total"]
    ):
        out[(str(gid), str(home))] = float(total) / 2.0 - float(spread) / 2.0
        out[(str(gid), str(away))] = float(total) / 2.0 + float(spread) / 2.0
    return out


def recent_teammates(est: pd.DataFrame, team: str, now: int, season: int) -> pd.DataFrame:
    """Non-quarterbacks who touched the ball for this team in its last few weeks."""
    if est.empty:
        return est
    return est[
        (est["team_id"] == team)
        & (est["last_t"] >= now - MISSING_WEEKS)
        & (est["last_t"] >= abs_week(season, 0))
        & (est["pos"] != "QB")
    ]


def had_role(est: pd.DataFrame) -> pd.Series:
    """True for the teammates whose absence leaves a share worth handing out."""
    return (est["s_w"] >= MISSING_GAMES) & (
        (est["raw_tshare"] >= MISSING_TSHARE) | (est["raw_cshare"] >= MISSING_CSHARE)
    )


def split_squad(est: pd.DataFrame, team: str, appeared: set[str], now: int, season: int):
    """What was knowable before a past game: who was missing, and who was just waiting.

    A regular who is not in the box score was out, and that was known before kickoff.
    A backup who is not in it was probably in uniform and simply did not get the
    ball, so he still counts in the split (a "ghost"): leaving him out would hand
    his share to the others with knowledge nobody had before the game.
    """
    recent = recent_teammates(est, team, now, season)
    if recent.empty:
        return [], []
    absent = recent[~recent.index.isin(appeared)]
    missing = list(absent[had_role(absent)].index)
    ghosts = list(absent[~absent["regular"]].index)
    return missing, ghosts


def season_average(past: pd.DataFrame, season: int) -> pd.DataFrame:
    """Per-game average this season; a player with no games yet gets last season's.

    Passing numbers are averaged over the games the quarterback started, so a
    backup's mop-up minutes do not make the yardstick look worse than it is.
    """
    cols = [s for s in STATS if not s.startswith("pass_")]
    passing = [s for s in STATS if s.startswith("pass_")]
    starts = past[past["pass_att"] >= START_ATTEMPTS]

    def avg(frame: pd.DataFrame, which: list[str]) -> pd.DataFrame:
        this = frame[frame["season"] == season].groupby("athlete_id")[which].mean()
        prev = frame[frame["season"] == season - 1].groupby("athlete_id")[which].mean()
        return this.combine_first(prev)

    return avg(past, cols).join(avg(starts, passing), how="left")


def announced_starter(grp: pd.DataFrame, previous: str | None) -> str | None:
    """Who a bettor would have expected at quarterback before kickoff.

    Last game's quarterback if he threw a pass in this one; otherwise whoever threw
    the most. A backup who came in because the starter got hurt is not treated as
    the starter, exactly as it would have been before the game.
    """
    passers = grp[grp["pass_att"] >= 1]
    if passers.empty:
        return None
    if previous is not None and previous in set(passers["athlete_id"]):
        return previous
    return passers.loc[passers["pass_att"].idxmax(), "athlete_id"]


def walk_forward(
    players: pd.DataFrame,
    teams: pd.DataFrame,
    positions: dict[str, str],
    params: Params,
    seasons: list[int],
    market: dict | None = None,
) -> pd.DataFrame:
    """One row per player and game: projection (p_), actual (a_) and season average (b_)."""
    weeks = (
        teams[teams["season"].isin(seasons)][["season", "week_index", "t"]]
        .drop_duplicates()
        .sort_values("t")
    )
    out: list[dict] = []
    # Each team's quarterback in its previous game (most passes), week by week.
    primary = (
        players[players["pass_att"] >= START_ATTEMPTS]
        .sort_values("pass_att")
        .groupby(["team_id", "t"])["athlete_id"]
        .last()
    )
    league_start = players[players["pass_att"] >= START_ATTEMPTS]
    for season, _week_index, now in weeks.itertuples(index=False):
        past = players[players["t"] < now]
        if past.empty:
            continue
        priors = build_priors(past, positions)
        effects = fit_team_effects(teams, now, params)
        est = player_estimates(players, now, params, positions, priors, teams)
        base = season_average(past, season)
        week_players = players[players["t"] == now]
        for (game_id, team), grp in week_players.groupby(["game_id", "team_id"], sort=False):
            first = grp.iloc[0]
            appeared = list(grp["athlete_id"])
            before = primary[team][primary[team].index < now] if team in primary.index else []
            qb = announced_starter(grp, before.iloc[-1] if len(before) else None)
            missing, ghosts = split_squad(est, team, set(appeared), now, season)
            active = appeared + ghosts
            line = None
            if market and (game_id, team) in market:
                line = {"points": market[(game_id, team)]}
            usual = before.iloc[-1] if len(before) else None
            rows, ctx = project_team(
                team, first["opp_id"], bool(first["home"]), effects, est, active, qb, missing,
                positions, priors, params, line, usual,
            )  # fmt: skip
            actual = grp.set_index("athlete_id")
            for row in rows:
                aid = row["athlete_id"]
                ghost = aid not in actual.index
                rec = {
                    "season": season,
                    "week": int(first["week"]),
                    "t": now,
                    "game_id": game_id,
                    "team_id": team,
                    "opp_id": first["opp_id"],
                    "athlete_id": aid,
                    "name": est.at[aid, "name"] if ghost else actual.at[aid, "name"],
                    "ghost": ghost,
                    "pos": row["pos"],
                    "is_qb": row["is_qb"],
                    "games_before": int(est.at[aid, "games"]) if aid in est.index else 0,
                    "missing_targets": ctx["missing_targets"],
                    "missing_carries": ctx["missing_carries"],
                    "opp_pass": ctx["opp_pass"],
                    "opp_rush": ctx["opp_rush"],
                    "has_line": line is not None,
                    "market_factor": ctx["market_factor"],
                    "qb_change": ctx["qb_change"],
                }
                for stat in STATS:
                    rec[f"p_{stat}"] = row[stat]
                    rec[f"a_{stat}"] = 0.0 if ghost else float(actual.at[aid, stat])
                    known = aid in base.index and pd.notna(base.at[aid, stat])
                    if known:
                        rec[f"b_{stat}"] = float(base.at[aid, stat])
                    elif stat.startswith("pass_") and row["is_qb"]:
                        # A quarterback with no start on record: the league's average start.
                        rec[f"b_{stat}"] = float(league_start[league_start["t"] < now][stat].mean())
                    else:
                        rec[f"b_{stat}"] = 0.0
                out.append(rec)
    return pd.DataFrame(out)


def eligible(df: pd.DataFrame, stat: str) -> pd.Series:
    """Rows that count for a statistic, chosen with pre-game numbers only."""
    if stat in PASS_STATS:
        return df["is_qb"]
    if stat in RUSH_STATS:
        return ((df["p_rush_att"] >= MIN_CARRIES) | (df["b_rush_att"] >= MIN_CARRIES)) & ~df[
            "is_qb"
        ]
    if stat in REC_STATS:
        return (df["p_targets"] >= MIN_TARGETS) | (df["b_targets"] >= MIN_TARGETS)
    return (
        (df["p_targets"] >= MIN_TARGETS)
        | (df["b_targets"] >= MIN_TARGETS)
        | (df["p_rush_att"] >= MIN_CARRIES)
        | (df["b_rush_att"] >= MIN_CARRIES)
    )


def metrics(df: pd.DataFrame, stats: list[str] | None = None) -> pd.DataFrame:
    """Error of the model and of the season average, statistic by statistic."""
    rows = []
    for stat in stats or STATS:
        if stat in ("rush_td", "rec_td"):
            continue
        sub = df[eligible(df, stat)]
        if sub.empty:
            continue
        err = sub[f"p_{stat}"] - sub[f"a_{stat}"]
        base = sub[f"b_{stat}"] - sub[f"a_{stat}"]
        rows.append(
            {
                "stat": stat,
                "n": len(sub),
                "mean": sub[f"a_{stat}"].mean(),
                "mae": err.abs().mean(),
                "mae_avg": base.abs().mean(),
                "rmse": float(np.sqrt((err**2).mean())),
                "rmse_avg": float(np.sqrt((base**2).mean())),
                "bias": err.mean(),
            }
        )
    out = pd.DataFrame(rows)
    out["gain"] = 1.0 - out["mae"] / out["mae_avg"]
    return out


def load_inputs(roster: pd.DataFrame | None = None):
    players, teams = prepare(fetch.load_player_games(), fetch.load_team_games())
    positions = infer_positions(players, roster)
    games = load_games(league=config.NFL)
    market = implied_points(games, load_odds_history(league=config.NFL))
    return players, teams, positions, market


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Backtest of the NFL player projections")
    parser.add_argument("--seasons", type=int, nargs="*", default=None)
    parser.add_argument("--ablation", action="store_true")
    args = parser.parse_args(argv)
    roster, _depth, _failed = fetch.fetch_rosters(log=lambda *_: None)
    players, teams, positions, market = load_inputs(roster)
    seasons = args.seasons or sorted(players["season"].unique())[1:]
    pd.set_option("display.width", 200)
    variants = {"model": Params()}
    if args.ablation:
        variants.update(
            {
                "no opponent": replace(Params(), use_opponent=False),
                "no absences": replace(Params(), use_teammates=False),
                "no betting line": replace(Params(), use_market=False),
            }
        )
    for name, params in variants.items():
        df = walk_forward(players, teams, positions, params, seasons, market)
        print(f"\n=== {name} · seasons {seasons} · {df['game_id'].nunique()} games")
        print(metrics(df).round(3).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
