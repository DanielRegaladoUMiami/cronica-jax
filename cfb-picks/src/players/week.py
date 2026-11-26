"""This week's NFL player projections, next to what the market charges.

Usage:
    python -m src.players.week              # the current week
    python -m src.players.week --week 12
    python -m src.players.week --no-kalshi  # projections and DraftKings lines only

Run `python -m src.picks --league nfl` first: it refreshes the schedule this reads.

What comes out (site/data/nfl_players.json), per game and team:
- who is out and who inherits the role, the starting quarterback, the opposing defense;
- each player's projection per statistic, with his season average beside it;
- the DraftKings line and Kalshi's ladder of prices, with the model's probability
  for every rung and what that leaves after Kalshi's fee.

A contract is flagged PLAY only if the model gives it 5+ points of edge after the
fee, the market does not flatly disagree, the player is not questionable and the
book is liquid. There is no betting record for these yet: it starts being kept now.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from .. import config
from ..data import kalshi
from ..data.build import load_games, load_teams
from ..data.espn import EspnError
from ..data.parse import parse_calendar, parse_event_odds, parse_status
from ..picks import _clean, next_open_week, week_for, week_label
from . import backtest as bt
from . import dist as dist_mod
from . import fetch, markets
from .model import (
    Params,
    abs_week,
    build_priors,
    fit_team_effects,
    player_estimates,
    project_team,
)

LEAGUE = config.NFL
OUT_PATH = config.SITE_DATA_DIR / "nfl_players.json"
# The record of flagged contracts is kept with the site, so the scheduled run can pick up
# the published copy and add to it (see the workflow) without committing to the repository.
LOG_PATH = config.SITE_DATA_DIR / "nfl_props_log.json"

# Statistics shown for each position, in order.
SHOWN = {
    "QB": ["pass_yds", "pass_td", "pass_int", "pass_att", "pass_cmp", "rush_yds", "td"],
    "RB": ["rush_yds", "rush_att", "rec", "rec_yds", "td"],
    "WR": ["rec", "rec_yds", "td"],
    "TE": ["rec", "rec_yds", "td"],
}
STAT_NAMES = {
    "pass_yds": "Passing yards",
    "pass_td": "Passing touchdowns",
    "pass_int": "Interceptions thrown",
    "pass_att": "Pass attempts",
    "pass_cmp": "Completions",
    "rush_yds": "Rushing yards",
    "rush_att": "Carries",
    "rec": "Receptions",
    "rec_yds": "Receiving yards",
    "targets": "Targets",
    "td": "Touchdowns (rushing or receiving)",
}
DEPTH_SLOTS = ("qb", "rb", "fb", "wr1", "wr2", "wr3", "wr", "te")
RECENT_GAMES = 2  # touched the ball in one of the team's last two games: part of the plan

# The rule for flagging a contract.
EDGE_PLAY = 0.05
EDGE_LEAN = 0.03
PRICE_RANGE = (0.15, 0.85)  # outside it one tick of price is a large share of the bet
MAX_SPREAD = 0.05  # bid/ask further apart than this: the book is too thin to trust the price
MAX_GAP = 0.12  # model this far from the market: assume the market knows something
MIN_GAMES = 4  # fewer games than this on record: the role is not established
MIN_VOLUME = 50  # contracts traded: below it the price on the screen is nobody's opinion
DK_MARGIN = 0.01  # the DraftKings line has to leave at least this after the fee, too
LADDER_FIELDS = ("t", "p", "yes_ask", "no_ask", "volume", "edge_yes", "edge_no")


def team_sheet(team_id: str, roster: pd.DataFrame, depth: pd.DataFrame, est: pd.DataFrame,
               positions: dict[str, str], now: int, last_games: list[int]) -> dict:  # fmt: skip
    """Who plays for a team this week, who starts at quarterback and who is missing.

    A player is in the plan if he is on the roster, not ruled out, and either
    touched the ball in one of the team's last two games or is first or second
    string on the depth chart. `stale` are the ones who are back in the plan
    without having played lately: their role is discounted (see `live_weights`).
    """
    r = roster[(roster["team_id"] == team_id) & roster["position"].isin(fetch.SKILL_POSITIONS)]
    status = dict(zip(r["athlete_id"], r["injury_status"]))
    names = dict(zip(r["athlete_id"], r["name"]))
    d = depth[depth["team_id"] == team_id] if len(depth) else depth
    top = set(d[(d["order"] <= 2) & d["slot"].isin(DEPTH_SLOTS)]["athlete_id"]) if len(d) else set()
    qb_order = list(d[d["slot"] == "qb"].sort_values("order")["athlete_id"]) if len(d) else []
    on_roster = set(r[~r["practice_squad"]]["athlete_id"])
    cut = (
        last_games[-RECENT_GAMES]
        if len(last_games) >= RECENT_GAMES
        else (last_games[0] if last_games else now)
    )
    recent = (
        set(est.index[(est["last_t"] >= cut) & (est["team_id"] == team_id)]) if len(est) else set()
    )
    candidates = {a for a in on_roster if a in recent or a in top}
    out_ids = {a for a in candidates if status.get(a) in fetch.OUT_STATUSES}
    qbs = [a for a in qb_order if a in on_roster and status.get(a) not in fetch.OUT_STATUSES]
    if not qbs:  # no depth chart: the healthy quarterback who has thrown the most
        pool = [
            a for a in on_roster
            if positions.get(a) == "QB" and status.get(a) not in fetch.OUT_STATUSES and a in est.index
        ]  # fmt: skip
        qbs = sorted(pool, key=lambda a: -float(est.at[a, "s_pass"]))
    qb = qbs[0] if qbs else None
    active = sorted(
        a for a in candidates if a not in out_ids and (positions.get(a) != "QB" or a == qb)
    )
    if qb is not None and qb not in active:
        active.append(qb)
    missing = []
    for a in out_ids:
        if a not in est.index or positions.get(a) == "QB":
            continue
        row = est.loc[[a]]
        if (
            row["team_id"].iloc[0] == team_id
            and row["last_t"].iloc[0] >= now - bt.MISSING_WEEKS
            and bool(bt.had_role(row).iloc[0])
        ):
            missing.append(a)
    usual_qb = None
    if len(est):
        thrown = est[
            (est["team_id"] == team_id) & (est["pos"] == "QB") & (est["last_t"] >= now - 4)
        ]
        if len(thrown):
            usual_qb = thrown["s_pass"].idxmax()
    return {
        "active": active,
        "qb": qb,
        "usual_qb": usual_qb,
        "missing": sorted(missing),
        "out": sorted(out_ids),
        "stale": sorted(a for a in active if a != qb and a not in recent),
        "status": status,
        "names": names,
    }


def live_weights(est: pd.DataFrame, stale: list[str], params: Params) -> pd.DataFrame:
    """Discount the role of players who are in the plan but have not played lately.

    In the backtest a regular who shows up in the box score is known to have played.
    Before the game that is not known for a man who has been away for weeks: he
    counts by how often he has actually appeared, not as a full-time regular.
    """
    known = [a for a in stale if a in est.index]
    if not known:
        return est
    est = est.copy()
    est.loc[known, "weight"] = est.loc[known, "appear"].clip(params.appear_floor, 1.0)
    return est


def mean_for_line(dists: dist_mod.Distributions, stat: str, line: float) -> float | None:
    """The projection that would make a sportsbook line an even bet.

    For yardage the line is close to the market's median, so this is the market's
    own projection. For counts (receptions, attempts) the line is forced onto a
    half number and the price leans to one side, so it is only a rough guide; for
    touchdowns and interceptions it says too little and is not used.
    """
    if stat in ("pass_td", "pass_int", "td") or line <= 0:
        return None
    if stat not in dists.ratios and stat not in dists.dispersion:
        return None
    lo, hi = max(line * 0.3, 0.05), line * 3.0 + 20.0
    for _ in range(50):
        mid = (lo + hi) / 2.0
        p = dists.prob_over(stat, line, mid)
        if p is None or p < 0.5:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0


def ladder_view(
    dists, stat: str, mean: float, rungs: list[dict], dk_mean: float | None
) -> list[dict]:
    """Each Kalshi rung with the model's probability and what is left after the fee."""
    out = []
    for rung in rungs:
        p = dists.prob_at_least(stat, rung["t"], mean)
        if p is None or rung["yes_ask"] is None:
            continue
        ya, na = rung["yes_ask"], rung["no_ask"]
        if not 0.05 <= ya <= 0.95:
            continue
        p_dk = dists.prob_at_least(stat, rung["t"], dk_mean) if dk_mean else None
        row = {
            "t": rung["t"],
            "p": round(p, 4),
            "p_dk": round(p_dk, 4) if p_dk is not None else None,
            "yes_bid": rung["yes_bid"],
            "yes_ask": ya,
            "no_bid": rung["no_bid"],
            "no_ask": na,
            "volume": round(rung["volume"] or 0),
            "edge_yes": round(p - ya - kalshi.taker_fee(ya), 4),
            "edge_no": round((1.0 - p) - na - kalshi.taker_fee(na), 4),
            "ticker": rung["ticker"],
        }
        if p_dk is not None:
            row["edge_yes_dk"] = round(p_dk - ya - kalshi.taker_fee(ya), 4)
            row["edge_no_dk"] = round((1.0 - p_dk) - na - kalshi.taker_fee(na), 4)
        out.append(row)
    return out


def best_contract(ladder: list[dict], flags: dict) -> dict | None:
    """The rung and side where the model sees most value, with its label and the reason."""
    best = None
    for rung in ladder:
        for side in ("yes", "no"):
            price = rung[f"{side}_ask"]
            bid = rung[f"{side}_bid"]
            if price is None or bid is None:
                continue
            if not PRICE_RANGE[0] <= price <= PRICE_RANGE[1] or price - bid > MAX_SPREAD:
                continue
            edge = rung[f"edge_{side}"]
            if best is None or edge > best["edge"]:
                p_model = rung["p"] if side == "yes" else 1.0 - rung["p"]
                best = {
                    "t": rung["t"],
                    "side": side,
                    "price": price,
                    "p": round(p_model, 4),
                    "edge": edge,
                    "edge_dk": rung.get(f"edge_{side}_dk"),
                    "gap": round(abs(rung["p"] - (rung["yes_bid"] + rung["yes_ask"]) / 2.0), 4),
                    "volume": rung["volume"],
                    "ticker": rung["ticker"],
                }
    if best is None or best["edge"] < EDGE_LEAN:
        return None
    # Reasons to stop and look: any of them means the number is not to be trusted yet.
    reasons = []
    if best["gap"] > MAX_GAP:
        reasons.append("the market is far from the model: check the news first")
    if flags.get("questionable"):
        reasons.append("the player is questionable")
    if flags.get("small_sample"):
        reasons.append("few games on record: the role is not established")
    if flags.get("qb_change"):
        reasons.append("the team changes quarterback")
    if flags.get("returning"):
        reasons.append("he has not played in the last two games")
    if best["edge_dk"] is not None and best["edge_dk"] < 0:
        reasons.append("the DraftKings line does not back it")
    # Reasons it is only a lean: the model is alone, or the price is thin.
    soft = []
    if best["edge_dk"] is None:
        soft.append("no DraftKings line to confirm it")
    elif 0 <= best["edge_dk"] < DK_MARGIN:
        soft.append("the DraftKings line barely backs it")
    if (best["volume"] or 0) < MIN_VOLUME:
        soft.append("few contracts traded: the price may not be real")
    if reasons:
        best["label"] = "CHECK"
    elif best["edge"] >= EDGE_PLAY and not soft:
        best["label"] = "PLAY"
    else:
        best["label"] = "LEAN"
    best["reasons"] = reasons + soft
    return best


def round_or_none(value, digits: int = 1):
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)


def build(
    week: int | None = None, season: int | None = None, use_kalshi: bool = True, log=print
) -> dict:
    season = season or config.current_season()
    client = markets.espn_client()
    params = Params()

    fetch.fetch_player_games([season - 1, season], log=log)
    roster, depth, roster_failed = fetch.fetch_rosters(client, log=log)
    players, teams, positions, market_hist = bt.load_inputs(roster)
    games = load_games(league=LEAGUE)
    meta = load_teams(league=LEAGUE)
    meta = meta[meta["season"] == meta["season"].max()].set_index("team_id")
    calendar = parse_calendar(client.scoreboard_week(season, 1))
    if week is None:
        week = next_open_week(games, season, week_for(calendar, datetime.now(timezone.utc), LEAGUE))
    sched = games[(games["season"] == season) & (games["week_index"] == week)].sort_values("date")
    if sched.empty:
        raise EspnError(f"no games found for week {week} of {season}")
    now = int(abs_week(season, int(sched["week_index"].iloc[0])))

    # --- the model's own error, and the tables that turn projections into probabilities
    seasons_bt = [s for s in (season - 1, season) if s in set(players["season"])]
    back = bt.walk_forward(players, teams, positions, params, seasons_bt, market_hist)
    dists = dist_mod.fit(back, bt.eligible)
    this_season = back[back["season"] == season]
    error = {
        row["stat"]: {
            "n": int(row["n"]),
            "mae": round(row["mae"], 2),
            "mae_avg": round(row["mae_avg"], 2),
            "rmse": round(row["rmse"], 2),
            "rmse_avg": round(row["rmse_avg"], 2),
        }
        for row in bt.metrics(back).to_dict("records")
    }
    error_season = {
        row["stat"]: {
            "n": int(row["n"]),
            "mae": round(row["mae"], 2),
            "mae_avg": round(row["mae_avg"], 2),
        }
        for row in (bt.metrics(this_season).to_dict("records") if len(this_season) else [])
    }

    # --- the state of every team and player before this week
    past = players[players["t"] < now]
    priors = build_priors(past, positions)
    effects = fit_team_effects(teams, now, params)
    est = player_estimates(players, now, params, positions, priors, teams)
    base = bt.season_average(past, season)
    games_played = past[past["season"] == season].groupby("athlete_id").size()
    played = teams[(teams["t"] < now) & (teams["season"] == season)]
    schedule = {tid: sorted(grp["t"].tolist()) for tid, grp in played.groupby("team_id")}

    # --- this week's lines (for the team's implied points) and live status
    seasontype = int(sched["season_type"].iloc[0])
    if seasontype == 2:
        boards = [client.scoreboard_week(season, int(sched["week"].iloc[0]), 2, max_age=300)]
    else:  # ESPN numbers the playoff rounds on their own: look at all of them
        boards = [client.scoreboard_week(season, w, seasontype, max_age=300) for w in range(1, 6)]
    events = {str(ev.get("id")): ev for board in boards for ev in board.get("events") or []}

    known = set(meta["abbr"])
    ladders: dict = {}
    warnings: list[str] = []
    if roster_failed:
        warnings.append(
            f"ESPN did not return {len(roster_failed)} rosters: no injury data for those teams."
        )
    if use_kalshi:
        try:
            ladders = markets.kalshi_ladders(kalshi.KalshiClient(), known, log=log)
        except kalshi.KalshiError as exc:
            warnings.append(f"Kalshi did not answer ({exc}): no prices shown.")

    out_games = []
    flagged = []
    for game in sched.to_dict("records"):
        gid = str(game["game_id"])
        event = events.get(gid) or {}
        status = parse_status(event) if event else str(game["status"])
        odds = parse_event_odds(event) if event else None
        implied = {}
        if odds and odds.get("spread_home") is not None and odds.get("total") is not None:
            implied[str(game["home_id"])] = odds["total"] / 2.0 - odds["spread_home"] / 2.0
            implied[str(game["away_id"])] = odds["total"] / 2.0 + odds["spread_home"] / 2.0
        try:
            dk = markets.draftkings_lines(client, gid)
        except EspnError:
            dk = {}
            warnings.append(
                f"{game['away_abbr']} @ {game['home_abbr']}: no DraftKings player lines."
            )
        key = (str(game["away_abbr"]), str(game["home_abbr"]))
        game_ladders = ladders.get(key, {})
        team_blocks = []
        for side, opp_side in (("away", "home"), ("home", "away")):
            tid, oid = str(game[f"{side}_id"]), str(game[f"{opp_side}_id"])
            sheet = team_sheet(tid, roster, depth, est, positions, now, schedule.get(tid, []))
            line = {"points": implied[tid]} if tid in implied else None
            rows, ctx = project_team(
                tid, oid, side == "home", effects, live_weights(est, sheet["stale"], params),
                sheet["active"], sheet["qb"], sheet["missing"], positions, priors, params, line,
                sheet["usual_qb"],
            )  # fmt: skip
            qb_change = (
                sheet["qb"] is not None
                and sheet["usual_qb"] is not None
                and sheet["qb"] != sheet["usual_qb"]
            )
            people = []
            for row in rows:
                aid = row["athlete_id"]
                pos = row["pos"]
                name = sheet["names"].get(aid) or (est.at[aid, "name"] if aid in est.index else aid)
                relevant = row["is_qb"] or row["targets"] >= 1.5 or row["rush_att"] >= 3.0
                has_market = (
                    any((aid, s) in dk for s in STAT_NAMES)
                    or markets.plain_name(name) in game_ladders
                )
                if pos == "QB" and not row["is_qb"]:
                    continue
                if not relevant and not has_market:
                    continue
                n_games = int(est.at[aid, "games"]) if aid in est.index else 0
                flags = {
                    "questionable": sheet["status"].get(aid) == "Questionable",
                    "small_sample": n_games < MIN_GAMES,
                    "qb_change": bool(qb_change),
                    "returning": aid in sheet["stale"],
                }
                stats = {}
                for stat in SHOWN.get(pos, SHOWN["WR"]):
                    mean = float(row[stat])
                    line_dk = dk.get((aid, stat))
                    rungs = game_ladders.get(markets.plain_name(name), {}).get(stat, [])
                    if mean < 0.05 and not line_dk and not rungs:
                        continue
                    dk_mean = mean_for_line(dists, stat, line_dk["line"]) if line_dk else None
                    ladder = ladder_view(dists, stat, mean, rungs, dk_mean)
                    best = best_contract(ladder, flags) if status == "scheduled" else None
                    # Keep the ladder short and light for the page: the rungs closest to
                    # an even bet, with only what is shown.
                    ladder = sorted(
                        sorted(ladder, key=lambda r: abs(r["p"] - 0.5))[:7], key=lambda r: r["t"]
                    )
                    ladder = [{k: r[k] for k in LADDER_FIELDS} for r in ladder]
                    entry = {
                        "proj": round(mean, 2 if mean < 10 else 1),
                        "avg": round_or_none(base.at[aid, stat], 2) if aid in base.index else None,
                        "lo": round_or_none(dists.quantile(stat, 0.25, mean)),
                        "hi": round_or_none(dists.quantile(stat, 0.75, mean)),
                        "dk": line_dk["line"] if line_dk else None,
                        "dk_open": line_dk["open"] if line_dk else None,
                        "p_over_dk": round_or_none(dists.prob_over(stat, line_dk["line"], mean), 3)
                        if line_dk
                        else None,
                        "ladder": ladder,
                        "best": best,
                    }
                    stats[stat] = entry
                    if best is not None:
                        flagged.append(
                            {
                                "label": best["label"],
                                "edge": best["edge"],
                                "game_id": gid,
                                "game": f"{game['away_abbr']} @ {game['home_abbr']}",
                                "kickoff": game["date"].isoformat(),
                                "athlete_id": aid,
                                "name": name,
                                "pos": pos,
                                "team": meta.at[tid, "abbr"],
                                "stat": stat,
                                "proj": entry["proj"],
                                "dk": entry["dk"],
                                **best,
                            }
                        )
                if not stats:
                    continue
                people.append(
                    {
                        "id": aid,
                        "name": name,
                        "pos": pos,
                        "status": sheet["status"].get(aid),
                        "games": int(games_played.get(aid, 0)),
                        "target_share": round(row["tshare"], 3),
                        "carry_share": round(row["cshare"], 3),
                        "flags": [k for k, v in flags.items() if v],
                        "stats": stats,
                    }
                )
            order = {"QB": 0, "RB": 1, "WR": 2, "TE": 3}
            people.sort(
                key=lambda p: (
                    order.get(p["pos"], 9),
                    -(
                        p["stats"].get("rec_yds", {}).get("proj", 0)
                        + p["stats"].get("rush_yds", {}).get("proj", 0)
                    ),
                )
            )
            team_blocks.append(
                {
                    "side": side,
                    "id": tid,
                    "abbr": meta.at[tid, "abbr"],
                    "name": meta.at[tid, "team"],
                    "color": meta.at[tid, "color"],
                    "implied_points": round_or_none(implied.get(tid)),
                    "qb": sheet["names"].get(sheet["qb"]) if sheet["qb"] else None,
                    "qb_change": bool(qb_change),
                    "usual_qb": (est.at[sheet["usual_qb"], "name"] if qb_change else None),
                    "pass_att": round(ctx["pass_att"], 1),
                    "rush_att": round(ctx["rush_att"], 1),
                    "pass_yds": round(ctx["pass_yds"]),
                    "rush_yds": round(ctx["rush_yds"]),
                    # The opposing defense against the league average: +0.06 = allows 6% more.
                    "opp_pass": round(ctx["opp_pass"], 3),
                    "opp_rush": round(ctx["opp_rush"], 3),
                    "missing": [
                        {
                            "name": sheet["names"].get(a) or est.at[a, "name"],
                            "pos": positions.get(a),
                            "status": sheet["status"].get(a),
                            "target_share": round(float(est.at[a, "raw_tshare"]), 3),
                            "carry_share": round(float(est.at[a, "raw_cshare"]), 3),
                        }
                        for a in sheet["missing"]
                    ],
                    "out": [
                        {
                            "name": sheet["names"].get(a),
                            "pos": positions.get(a),
                            "status": sheet["status"].get(a),
                        }
                        for a in sheet["out"]
                        if a not in sheet["missing"]
                    ],
                    "players": people,
                }
            )
        out_games.append(
            {
                "game_id": gid,
                "kickoff": game["date"].isoformat(),
                "status": status,
                "away": game["away_abbr"],
                "home": game["home_abbr"],
                "spread_home": odds.get("spread_home") if odds else None,
                "total": odds.get("total") if odds else None,
                "has_kalshi": bool(game_ladders),
                "has_dk": bool(dk),
                "teams": team_blocks,
            }
        )
        log(
            f"{game['away_abbr']:>3} @ {game['home_abbr']:<3} {status:<11} players"
            f" {sum(len(t['players']) for t in team_blocks):2d} · DraftKings lines {len(dk):3d}"
            f" · Kalshi players {len(game_ladders):2d}"
        )

    flagged.sort(key=lambda f: -f["edge"])
    record = update_log(flagged, players, season, week)
    return {
        "sport": "nfl",
        "season": season,
        "week": week,
        "week_label": week_label(calendar, week, LEAGUE),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "stat_names": STAT_NAMES,
        "rule": {
            "edge_play": EDGE_PLAY,
            "edge_lean": EDGE_LEAN,
            "price_range": PRICE_RANGE,
            "max_spread": MAX_SPREAD,
            "max_gap": MAX_GAP,
            "min_games": MIN_GAMES,
            "fee_rate": kalshi.TAKER_RATE,
        },
        "backtest": {
            "seasons": seasons_bt,
            "games": int(back["game_id"].nunique()),
            "error": error,
            "error_season": error_season,
            "ingredients": ingredients(
                players, teams, positions, params, seasons_bt, market_hist, back
            ),
            "calibration": calibration_summary(dists, back),
        },
        "record": record,
        "flagged": flagged,
        "games": out_games,
        "warnings": warnings,
    }


def ingredients(players, teams, positions, params, seasons, market, full) -> dict:
    """What knowing the absences and the opponent is worth, measured where it matters."""
    from dataclasses import replace

    key = ["game_id", "athlete_id"]

    def versus(other: pd.DataFrame, mask: pd.Series, stats: list[str]) -> dict:
        a = full[mask]
        b = other.merge(a[key], on=key)
        ma, mb = bt.metrics(a, stats).set_index("stat"), bt.metrics(b, stats).set_index("stat")
        return {
            s: {
                "n": int(ma.at[s, "n"]),
                "mae": round(float(ma.at[s, "mae"]), 2),
                "mae_without": round(float(mb.at[s, "mae"]), 2),
                "bias_without": round(float(mb.at[s, "bias"]), 2),
            }
            for s in stats
            if s in ma.index and s in mb.index
        }

    no_abs = bt.walk_forward(
        players, teams, positions, replace(params, use_teammates=False), seasons, market
    )
    no_opp = bt.walk_forward(
        players, teams, positions, replace(params, use_opponent=False), seasons, market
    )
    team_games = full.groupby(["game_id", "team_id"])[
        ["missing_targets", "missing_carries"]
    ].first()
    return {
        "absent_receivers": versus(
            no_abs, full["missing_targets"] >= 0.15, ["targets", "rec", "rec_yds"]
        ),
        "absent_runners": versus(no_abs, full["missing_carries"] >= 0.25, ["rush_att", "rush_yds"]),
        "share_absent_receivers": round(float((team_games["missing_targets"] >= 0.15).mean()), 3),
        "share_absent_runners": round(float((team_games["missing_carries"] >= 0.25).mean()), 3),
        "opponent": versus(no_opp, full["season"] > 0, ["pass_yds", "rush_yds", "rec_yds"]),
    }


def calibration_summary(dists, back: pd.DataFrame) -> dict:
    """How often "N+" happened when the model gave it each probability."""
    thresholds = {
        "pass_yds": range(150, 351, 25),
        "rush_yds": range(10, 131, 10),
        "rec_yds": range(10, 131, 10),
        "rec": range(1, 10),
        "td": (1, 2),
    }
    out = {}
    for stat, grid in thresholds.items():
        table = dist_mod.calibration(dists, back, bt.eligible, stat, grid)
        if table.empty:
            continue
        gap = float(
            (table["n"] * (table["predicted"] - table["observed"]).abs()).sum() / table["n"].sum()
        )
        out[stat] = {
            "n": int(table["n"].sum()),
            "gap": round(gap, 3),
            "bins": [
                {
                    "predicted": round(r.predicted, 3),
                    "observed": round(r.observed, 3),
                    "n": int(r.n),
                }
                for r in table.itertuples()
            ],
        }
    return out


# ------------------------------------------------------------- betting record
LOG_COLUMNS = [
    "season", "week", "game_id", "kickoff", "athlete_id", "name", "team", "stat", "t", "side",
    "label", "price", "p", "edge", "logged_at",
]  # fmt: skip


def update_log(flagged: list[dict], players: pd.DataFrame, season: int, week: int) -> dict:
    """Keep every flagged contract, and settle the ones whose game has been played.

    The price kept is the last one seen before kickoff. This is the only honest way
    to find out whether the flags make money: write them down before the game.
    """
    log = read_log()
    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    new = pd.DataFrame(
        [
            {
                **{c: f.get(c) for c in LOG_COLUMNS},
                "season": season,
                "week": week,
                "logged_at": stamp,
            }
            for f in flagged
        ],
        columns=LOG_COLUMNS,
    )
    keys = ["season", "week", "athlete_id", "stat", "t", "side"]
    if len(new):
        # This week's flags replace what was logged for the games still to be played.
        keep = log[
            ~(
                (log["season"] == season)
                & (log["week"] == week)
                & log["game_id"].isin(new["game_id"])
            )
        ]
        log = pd.concat([keep, new], ignore_index=True).drop_duplicates(keys, keep="last")
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        rows = _clean(log[LOG_COLUMNS].to_dict("records"))
        lines = ",\n".join(json.dumps(r, ensure_ascii=False, separators=(",", ":")) for r in rows)
        LOG_PATH.write_text("[\n" + lines + "\n]\n", encoding="utf-8")
    since = None if log.empty else f"week {int(log['week'].min())}"
    if log.empty:
        return {label: _record(log) for label in ("plays", "leans", "checks")} | {"since": since}
    actual = players.assign(td=players["rush_td"] + players["rec_td"])
    merged = log.merge(
        actual[["game_id", "athlete_id", *[s for s in STAT_NAMES if s in actual.columns]]],
        on=["game_id", "athlete_id"],
        how="inner",
    )
    if merged.empty:
        return {label: _record(merged) for label in ("plays", "leans", "checks")} | {"since": since}
    value = merged.apply(lambda r: r[r["stat"]], axis=1)
    hit = np.where(merged["side"] == "yes", value >= merged["t"], value < merged["t"])
    fee = kalshi.TAKER_RATE * merged["price"] * (1 - merged["price"])
    merged = merged.assign(
        won=hit, profit=np.where(hit, 1 - merged["price"], -merged["price"]) - fee
    )
    return {
        "plays": _record(merged[merged["label"] == "PLAY"]),
        "leans": _record(merged[merged["label"] == "LEAN"]),
        "checks": _record(merged[merged["label"] == "CHECK"]),
        "since": since,
    }


def read_log(path=None) -> pd.DataFrame:
    """The flagged contracts written down so far. A missing or damaged file starts a new one."""
    path = path or LOG_PATH
    try:
        rows = json.loads(path.read_text(encoding="utf-8")) if path.exists() else []
    except (OSError, ValueError):
        rows = []
    frame = pd.DataFrame(rows, columns=LOG_COLUMNS)
    for col in ("game_id", "athlete_id"):
        frame[col] = frame[col].astype(str)
    return frame


def _record(frame: pd.DataFrame) -> dict:
    if frame is None or frame.empty or "won" not in frame:
        return {"n": 0, "won": 0, "lost": 0, "profit": 0.0, "staked": 0.0}
    return {
        "n": len(frame),
        "won": int(frame["won"].sum()),
        "lost": int((~frame["won"]).sum()),
        "profit": round(float(frame["profit"].sum()), 2),  # in contracts of $1
        "staked": round(float(frame["price"].sum()), 2),
    }


def annotate_team_picks(data: dict, path=None, log=print) -> int:
    """Tell the team picks which teams start a different quarterback than usual.

    The team model rates a team by its season: it does not know that the starter is
    out. A pick on such a team (or on the total of its game) is moved to AVOID with
    the reason spelled out. Returns how many picks were changed.
    """
    path = path or LEAGUE.site_json("picks.json")
    if not path.exists():
        return 0
    picks = json.loads(path.read_text(encoding="utf-8"))
    if picks.get("season") != data["season"] or picks.get("week") != data["week"]:
        return 0
    changes = {
        g["game_id"]: {t["side"]: t for t in g["teams"] if t["qb_change"]} for g in data["games"]
    }
    changed = 0
    for game in picks.get("games") or []:
        sides = changes.get(str(game.get("id"))) or {}
        if not sides:
            continue
        notes = [
            f"{t['name']} start {t['qb']} at quarterback instead of {t['usual_qb']}"
            for t in sides.values()
        ]
        game["qb_changes"] = notes
        alerts = game.setdefault("alerts", [])
        for note in notes:
            text = f"{note}: the team model does not know it"
            if text not in alerts:
                alerts.append(text)
        for market in ("ml", "spread", "total"):
            block = game.get(market) or {}
            for key in ("home", "away", "over", "under"):
                side = block.get(key)
                if not isinstance(side, dict) or side.get("label") not in ("PLAY", "AVOID"):
                    continue
                # A side bet is touched by either quarterback; so is the total.
                reason = "; ".join(notes) + ": the model rates the team with its usual quarterback"
                if reason not in side.setdefault("reasons", []):
                    side["reasons"].append(reason)
                if side["label"] == "PLAY":
                    side["label"] = "AVOID"
                    side["stake"] = 0.0
                    changed += 1
    path.write_text(json.dumps(picks, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    log(f"Team picks: {changed} moved from PLAY to AVOID because of a quarterback change")
    return changed


def print_summary(data: dict) -> None:
    print(f"\nNFL players, {data['week_label']} {data['season']}: {len(data['games'])} games")
    err = data["backtest"]["error"]
    print("Backtest (average miss, model vs the player's season average):")
    for stat in ("pass_yds", "rush_yds", "rec_yds", "rec", "pass_td"):
        if stat in err:
            print(
                f"  {STAT_NAMES[stat]:<22s} {err[stat]['mae']:6.2f} vs {err[stat]['mae_avg']:6.2f}"
            )
    for label in ("PLAY", "LEAN", "CHECK"):
        rows = [f for f in data["flagged"] if f["label"] == label]
        print(f"\n{label} ({len(rows)}):")
        for f in rows[:14]:
            side = f"{f['t']}+" if f["side"] == "yes" else f"under {f['t']}"
            print(
                f"  {f['name']:<22s} {f['team']:<3s} {STAT_NAMES[f['stat']][:18]:<18s} {side:<10s}"
                f" {f['price'] * 100:3.0f}c · model {f['p'] * 100:3.0f}% · edge {f['edge'] * 100:+5.1f}"
                f" · proj {f['proj']} (DK {f['dk']}) · {f['game']}"
                + (f" · {'; '.join(f['reasons'])}" if f["reasons"] else "")
            )
    for warning in data["warnings"]:
        print("WARNING:", warning)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="This week's NFL player projections")
    parser.add_argument("--week", type=int, default=None)
    parser.add_argument("--season", type=int, default=config.current_season())
    parser.add_argument("--no-kalshi", action="store_true")
    args = parser.parse_args(argv)
    try:
        data = build(args.week, args.season, use_kalshi=not args.no_kalshi)
    except EspnError as exc:
        print(f"ERROR: the ESPN API failed and no data is made up: {exc}", file=sys.stderr)
        return 1
    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    data = _clean(data)
    # One line: this is the largest file the site loads, and nobody reads it by hand.
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT_PATH.write_text(text + "\n", encoding="utf-8")
    annotate_team_picks(data)
    print_summary(data)
    print(f"\nExported: {OUT_PATH.relative_to(config.ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
