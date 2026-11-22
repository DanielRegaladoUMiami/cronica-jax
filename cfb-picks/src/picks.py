"""Weekly picks.

Usage:
    python -m src.picks --week 12        # college football, a specific week
    python -m src.picks                  # college football, the current week
    python -m src.picks --league nfl     # the NFL, the current week
    python -m src.picks --skip-build     # do not download games again

What it does:
1. Updates the league's games and the closing lines of games already played.
2. Fits the selected model on everything played so far.
3. For each game: projected score, probabilities, edge against your lines
   (data/odds_manual.csv, or data/nfl_odds_manual.csv) or ESPN's, a stability
   check and a label.
4. Exports site/data/picks.json and backtest.json (nfl_picks.json and
   nfl_backtest.json for the NFL).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from . import config
from .data import build as build_mod
from .data.build import postseason_rounds
from .data.espn import EspnClient, EspnError
from .data.odds import (
    build_markets,
    compare_sources,
    load_manual_odds,
    match_manual,
    normalize_team,
    team_lookup,
)
from .data.odds_history import fetch_odds_history, load_odds_history
from .data.parse import parse_calendar, parse_event, parse_event_odds
from .model.backtest import (
    by_week,
    calibration,
    fbs_only,
    metrics,
    residual_sigma,
    walk_forward,
)
from .model.data import make_seasons
from .model.prob import american_to_decimal, norm_cdf, prob_to_american
from .model.registry import build_models, load_params, selected_name
from .model.rules import AVOID, PASS, PLAY, Rules, evaluate_game
from .model.simulate import drop_last_games, record, simulate_bets

LABELS = (PLAY, AVOID, PASS)
NO_DATA_REASONS = {
    "in_progress": "already started",
    "postponed": "postponed",
    "canceled": "canceled",
}


# ------------------------------------------------------------------- helpers
def _clean(obj):
    """Make an object JSON-ready: no NaN, no numpy types, sensible precision."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        value = float(obj)
        return None if math.isnan(value) or math.isinf(value) else round(value, 4)
    if obj is pd.NA or obj is None:
        return None
    return obj


def week_for(calendar: dict, now: datetime, league: config.League = config.CFB) -> int:
    """Current week as a week_index. After the regular season: the bowl bucket for
    college football (last week + 1), the current playoff round for the NFL."""
    regular = sorted(calendar["regular"], key=lambda w: w["week"])
    if not regular or now < regular[0]["start"]:
        return 1
    for entry in regular:
        if entry["start"] <= now < entry["end"]:
            return entry["week"]
    rounds = postseason_rounds(calendar, league)
    for entry in rounds:
        if now < entry["end"]:
            return entry["week_index"]
    return rounds[-1]["week_index"] if rounds else regular[-1]["week"] + 1


def next_open_week(games: pd.DataFrame, season: int, week: int) -> int:
    """`week`, or the following one when every game of `week` is already final.

    A week on ESPN's calendar runs until midweek, so the day after its last game
    there is nothing left in it to pick."""
    this_season = games[games["season"] == season]
    current = this_season[this_season["week_index"] == week]
    following = this_season[this_season["week_index"] == week + 1]
    if len(current) and current["completed"].all() and len(following):
        return week + 1
    return week


def week_label(calendar: dict, week: int, league: config.League = config.CFB) -> str:
    regular = [w["week"] for w in calendar["regular"]]
    if not regular or week <= max(regular):
        return f"Week {week}"
    if league.playoff_rounds:
        for entry in postseason_rounds(calendar, league):
            if entry["week_index"] == week and entry["label"]:
                return entry["label"]
    return "Postseason"


def _points(value) -> float:
    """A hand-written point adjustment; empty or unreadable counts as 0."""
    try:
        return float(value or 0)
    except ValueError:
        return 0.0


def _week_rows(path, season: int, week: int) -> list[dict]:
    """Rows of a hand-edited CSV that belong to this season and week."""
    if not path.exists():
        return []
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for raw in csv.DictReader(fh):
            try:
                if int(raw["season"]) == season and int(raw["week"]) == week:
                    rows.append(raw)
            except (KeyError, TypeError, ValueError):
                continue
    return rows


def load_notes(season: int, week: int, league: config.League = config.CFB) -> list[dict]:
    """The league's notes.csv: a manual point adjustment and a note per game
    (injuries, weather, motivation)."""
    return [
        {
            "away": (raw.get("away") or "").strip(),
            "home": (raw.get("home") or "").strip(),
            "home_margin_adj": _points(raw.get("home_margin_adj")),
            "total_adj": _points(raw.get("total_adj")),
            "notes": (raw.get("notes") or "").strip(),
        }
        for raw in _week_rows(league.notes_path, season, week)
    ]


def load_parlays(
    season: int, week: int, league: config.League = config.CFB
) -> dict[str, list[str]]:
    """The league's parlays.csv: saved moneyline parlays (one row per leg)."""
    out: dict[str, list[str]] = {}
    for raw in _week_rows(league.parlays_path, season, week):
        if (raw.get("team") or "").strip():
            out.setdefault((raw.get("name") or "Parlay").strip(), []).append(raw["team"].strip())
    return out


def _records(games: pd.DataFrame, season: int) -> dict[str, str]:
    """Each team's record: "7-3", or "7-3-1" once it has a tie."""
    done = games[(games["season"] == season) & games["completed"]]
    tally: dict[str, list[int]] = {}
    for row in done.itertuples():
        diff = int(row.home_points) - int(row.away_points)
        for tid, sign in ((row.home_id, 1), (row.away_id, -1)):
            wlt = tally.setdefault(tid, [0, 0, 0])
            wlt[0 if diff * sign > 0 else 1 if diff * sign < 0 else 2] += 1
    return {
        tid: f"{w}-{loss}" + (f"-{ties}" if ties else "") for tid, (w, loss, ties) in tally.items()
    }


# -------------------------------------------------------------- backtest.json
def build_backtest(
    games: pd.DataFrame,
    season: int,
    rules: Rules,
    league: config.League = config.CFB,
    log=print,
) -> tuple[dict, dict]:
    """Run the walk-forward backtest for every model and the betting backtest for
    the selected one. Returns (backtest.json, context for the picks)."""
    params = load_params(league)
    big = league.big_favorite
    seasons = make_seasons(games)
    years = sorted(seasons)
    eval_years = years[1:] if len(years) > 1 else years  # the first season only leaves the prior
    history_years = [y for y in eval_years if y < season]
    models = build_models(params)
    chosen = selected_name(params)
    if chosen not in models:
        chosen = "srs"

    comparison = []
    frames = {}
    for name, model in models.items():
        wf = walk_forward(model, seasons, eval_years)
        frames[name] = wf
        hist = fbs_only(wf[wf["season"] < season])
        cur = wf[wf["season"] == season]
        # sigma known BEFORE the season starts, so the Brier score is out of sample too
        sigma_hist = residual_sigma(hist)[0] if len(hist) else None
        comparison.append(
            {
                "model": name,
                "label": model.label,
                "selected": name == chosen,
                "current": metrics(fbs_only(cur), sigma_hist, big),
                "current_with_fcs": metrics(cur, sigma_hist, big),
                "history": metrics(hist, big=big) if len(hist) else {"n": 0},
            }
        )

    wf = frames[chosen]
    pool = fbs_only(wf[wf["week"] >= 2])  # week 1 has no games from that season yet: more error
    sigma_m, sigma_t = residual_sigma(pool)
    cur = fbs_only(wf[wf["season"] == season])
    hist = fbs_only(wf[wf["season"] < season])
    cur_metrics = metrics(cur, sigma_m, big)
    cal_cur, cal_hist = calibration(cur, sigma_m), calibration(hist, sigma_m)
    cal_error = None
    if cal_hist:
        weighted = sum(abs(b["predicted"] - b["observed"]) * b["n"] for b in cal_hist)
        cal_error = weighted / max(sum(b["n"] for b in cal_hist), 1)

    # ---- simulated bets against closing lines
    odds = load_odds_history(league=league)
    odds_seasons = sorted(int(s) for s in odds.loc[odds["has_odds"], "season"].unique())
    bets = pd.DataFrame()
    if odds_seasons:
        bets = simulate_bets(models[chosen], seasons, odds, sigma_m, sigma_t, rules, odds_seasons)

    def summary(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {}
        out = {}
        for market in ("ml", "spread", "total"):
            sel = frame[frame["market"] == market]
            out[market] = {label: record(sel[sel["label"] == label]) for label in LABELS}
        out["all"] = {label: record(frame[frame["label"] == label]) for label in LABELS}
        return out

    cur_bets = bets[bets["season"] == season] if len(bets) else bets
    no_bets = pd.DataFrame(columns=["result", "profit"])
    weekly = []
    cumulative = cumulative_ml = 0.0
    for row in by_week(cur, sigma_m, big):
        week = row["week"]
        plays = no_bets
        if len(cur_bets):
            plays = cur_bets[(cur_bets["week"] == week) & (cur_bets["label"] == PLAY)]
        rec = record(plays)
        rec_ml = record(plays[plays["market"] == "ml"]) if len(plays) else record(no_bets)
        cumulative += rec["units"]
        cumulative_ml += rec_ml["units"]
        weekly.append(
            {
                "week": week,
                "n": row["n"],
                "correct": row["correct"],
                "wrong": row["wrong"],
                "accuracy": row["accuracy"],
                "mae_margin": row["mae_margin"],
                "mae_total": row["mae_total"],
                "plays": rec["n"],
                "won": rec["won"],
                "lost": rec["lost"],
                "units": rec["units"],
                "cumulative": cumulative,
                # the same record for moneyline plays alone, for the moneyline-only view
                "ml": {
                    "plays": rec_ml["n"],
                    "won": rec_ml["won"],
                    "lost": rec_ml["lost"],
                    "units": rec_ml["units"],
                    "cumulative": cumulative_ml,
                },
            }
        )

    market = None
    if len(cur_bets):
        lines = odds[["game_id", "spread_home", "total"]].rename(columns={"total": "total_line"})
        with_line = cur.merge(lines, on="game_id").dropna(subset=["spread_home"])
        if len(with_line):
            mkt_margin = -with_line["spread_home"]
            # A pick'em has no favorite and a tie has no winner: neither is a hit or a miss.
            decided = (mkt_margin != 0) & (with_line["margin"] != 0)
            hits = ((mkt_margin > 0) == (with_line["margin"] > 0))[decided]
            market = {
                "n": len(with_line),
                "correct": int(hits.sum()),
                "wrong": int(len(hits) - hits.sum()),
                "mae_margin": float((mkt_margin - with_line["margin"]).abs().mean()),
                "accuracy": float(hits.mean()) if len(hits) else None,
                "mae_total": float((with_line["total_line"] - with_line["total"]).abs().mean()),
                "model_mae_margin": float(
                    (with_line["pred_margin"] - with_line["margin"]).abs().mean()
                ),
                "model_mae_total": float(
                    (with_line["pred_total"] - with_line["total"]).abs().mean()
                ),
            }

    backtest = {
        "sport": league.key,
        "league": league.name,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "big_margin": big,
        "selected": chosen,
        "selected_label": models[chosen].label,
        "tuned_on": params.get("tuned_on"),
        "history_seasons": history_years,
        "comparison": comparison,
        "market": market,
        "weeks": weekly,
        "calibration": {"current": cal_cur, "history": cal_hist, "mean_error": cal_error},
        "sigma": {"margin": sigma_m, "total": sigma_t, "n": len(pool)},
        "bets": {
            "current": summary(cur_bets),
            "history": summary(bets),
            "seasons_with_lines": odds_seasons,
            "note": "Flat one-unit bets against the DraftKings/consensus closing line from ESPN.",
        },
        "rules": rules.as_dict(),
        "prototype_reference": league.prototype,
    }
    context = {
        "model": models[chosen],
        "seasons": seasons,
        "sigma_m": sigma_m,
        "sigma_t": sigma_t,
        "metrics": cur_metrics,
        "cal_error": cal_error,
        "evidence": summary(bets),
        "odds_seasons": odds_seasons,
    }
    log(f"Model: {models[chosen].label} · margin sigma {sigma_m:.1f} · total sigma {sigma_t:.1f}")
    return backtest, context


# ----------------------------------------------------------------- picks.json
def build_picks(
    games: pd.DataFrame,
    teams: pd.DataFrame,
    season: int,
    week: int,
    context: dict,
    rules: Rules,
    client: EspnClient,
    league: config.League = config.CFB,
    log=print,
) -> dict:
    model, seasons = context["model"], context["seasons"]
    sigma_m, sigma_t = context["sigma_m"], context["sigma_t"]
    sa = seasons[season]
    warnings: list[str] = []

    # Model state with EVERYTHING played so far (including this week's midweek games).
    prior = _season_prior(model, seasons, season)
    train = sa.completed.copy()
    state = model.fit(sa, train, prior, now_week=week)
    rating = getattr(state, "rating", None)
    if rating is None and hasattr(state, "srs"):
        rating = state.srs.rating
    rank_of = {}
    if rating is not None:
        order = np.argsort(-rating[: sa.n - 1])
        rank_of = {int(i): pos + 1 for pos, i in enumerate(order)}

    # Live ESPN odds for the week.
    regular = games[(games["season"] == season) & (games["season_type"] == 2)]
    n_regular = int(regular["week"].max())
    espn_odds: dict[str, dict] = {}
    live_status: dict[str, str] = {}
    label = f"Week {week}" if week <= n_regular else "Postseason"
    try:
        calendar = parse_calendar(client.scoreboard_week(season, 1))
        label = week_label(calendar, week, league)
        if week <= n_regular:
            payloads = [client.scoreboard_week(season, week, max_age=300)]
        else:
            payloads = [
                client.scoreboard_week(season, e["week"], seasontype=3, max_age=300)
                for e in postseason_rounds(calendar, league)
                if e["week_index"] == week
            ]
        for payload in payloads:
            for event in payload.get("events") or []:
                row = parse_event(event, league.nicknames)
                if row is None:
                    continue
                live_status[row["game_id"]] = row["status"]
                odds = parse_event_odds(event)
                if odds:
                    espn_odds[row["game_id"]] = odds
    except EspnError as exc:
        warnings.append(f"ESPN did not return this week's odds ({exc}); using manual lines only")

    week_games = games[(games["season"] == season) & (games["week_index"] == week)]
    week_games = week_games.sort_values(["date", "game_id"])
    manual_rows = load_manual_odds(season=season, week=week, league=league)
    manual_by_game, manual_warnings = match_manual(manual_rows, week_games, teams)
    warnings.extend(manual_warnings)

    lookup = team_lookup(teams)
    notes_by_game: dict[str, dict] = {}
    pair_to_game = {(g.away_id, g.home_id): g.game_id for g in week_games.itertuples()}
    for note in load_notes(season, week, league):
        pair = (lookup.get(normalize_team(note["away"])), lookup.get(normalize_team(note["home"])))
        gid = pair_to_game.get(pair)
        if gid is None:
            warnings.append(
                f"{league.notes_path.name}: could not find {note['away']} @ {note['home']}"
                f" in week {week}"
            )
        else:
            notes_by_game[gid] = note

    team_info = teams.set_index("team_id").to_dict("index")
    records = _records(games, season)

    def team_block(tid: str, name: str, abbr: str, rank, idx: int) -> dict:
        info = team_info.get(tid, {})
        is_fbs = idx != sa.fcs_index
        return {
            "id": tid,
            "name": name,
            "abbr": abbr,
            "color": info.get("color") or "",
            "alt_color": info.get("alt_color") or "",
            "rank": None if pd.isna(rank) else int(rank),
            "record": records.get(tid),
            "fbs": is_fbs,
            "rating": float(rating[idx]) if rating is not None and is_fbs else None,
            "rating_rank": rank_of.get(idx) if is_fbs else None,
            "games": int(state.games[idx]) if is_fbs else None,
        }

    out_games = []
    for g in week_games.itertuples():
        status = live_status.get(g.game_id, g.status)
        if status == "final":
            continue  # already played: it will show up in the track record
        h, a = sa.idx(g.home_id), sa.idx(g.away_id)
        manual = manual_by_game.get(g.game_id)
        espn = espn_odds.get(g.game_id)
        note = notes_by_game.get(g.game_id, {})
        base = {
            "id": g.game_id,
            "date": g.date.isoformat().replace("+00:00", "Z"),
            "status": status,
            "neutral": bool(g.neutral_site),
            "on_my_list": manual is not None,
            "away": team_block(g.away_id, g.away_team, g.away_abbr, g.away_rank, a),
            "home": team_block(g.home_id, g.home_team, g.home_abbr, g.home_rank, h),
            "notes": note.get("notes", ""),
        }
        if status != "scheduled":
            reason = NO_DATA_REASONS.get(status, "no data from the API")
            out_games.append({**base, "no_data": True, "no_data_reason": reason})
            continue

        neutral = bool(g.neutral_site)
        margin, total = model.predict(state, [h], [a], [neutral])
        alt_state = model.fit(sa, drop_last_games(sa, train, (h, a)), prior, now_week=week)
        margin_alt, total_alt = model.predict(alt_state, [h], [a], [neutral])
        adj_m, adj_t = float(note.get("home_margin_adj", 0.0)), float(note.get("total_adj", 0.0))
        margin_f, margin_alt_f = float(margin[0]) + adj_m, float(margin_alt[0]) + adj_m
        has_total = not math.isnan(float(total[0]))
        total_f = float(total[0]) + adj_t if has_total else None
        total_alt_f = float(total_alt[0]) + adj_t if has_total else None

        markets = build_markets(manual, espn)
        fcs_game = bool(g.fcs_game)
        ev = evaluate_game(
            margin=margin_f,
            total=total_f,
            margin_alt=margin_alt_f,
            total_alt=total_alt_f,
            markets=markets,
            sigma_m=sigma_m,
            sigma_t=sigma_t,
            rules=rules,
            min_games_played=int(min(state.games[h], state.games[a])) if not fcs_game else 99,
        )
        alerts: list[str] = []
        if fcs_game:
            fcs_note = "FCS opponent: the model uses an FCS average, so this projection is rough"
            alerts.append(fcs_note)
            for market in ("ml", "spread", "total"):
                for side in ev[market].values():
                    if isinstance(side, dict) and side.get("label") == PLAY:
                        side.update(
                            {
                                "label": AVOID,
                                "stake": 0.0,
                                "kelly": 0.0,
                                "reasons": [fcs_note, *side["reasons"]],
                            }
                        )
        swing = abs(margin_f - margin_alt_f)
        if swing >= 4.0:
            alerts.append(
                f"unstable model: the projection moves {swing:.1f} pts"
                " without each team's last 2 games"
            )
        if ev["spread_move"] is not None and abs(ev["spread_move"]) >= rules.move_spread:
            toward = g.home_team if ev["spread_move"] > 0 else g.away_team
            alerts.append(
                f"the line moved {abs(ev['spread_move']):.1f} pts toward {toward} since it opened"
            )
        if ev["total_move"] is not None and abs(ev["total_move"]) >= rules.move_total:
            verb = "rose" if ev["total_move"] > 0 else "dropped"
            alerts.append(
                f"the total {verb} {abs(ev['total_move']):.1f} pts since it opened"
                " (weather or injuries?)"
            )
        if ev["margin_gap"] is not None and abs(ev["margin_gap"]) >= rules.gap_margin:
            alerts.append(
                f"the model is {abs(ev['margin_gap']):.1f} pts away from the market spread"
            )

        home_pts, away_pts = _scoreline(margin_f, total_f, league.min_points)
        p_home = float(norm_cdf(margin_f / sigma_m))
        espn_ref = None
        if espn:
            espn_ref = {
                "provider": espn.get("provider"),
                "spread_home": espn.get("spread_home"),
                "total": espn.get("total"),
                "ml_home": espn.get("ml_home"),
                "ml_away": espn.get("ml_away"),
                "spread_open": espn.get("spread_home_open"),
                "total_open": espn.get("total_open"),
            }
        out_games.append(
            {
                **base,
                "no_data": False,
                "fcs": fcs_game,
                "projection": {
                    "home": home_pts,
                    "away": away_pts,
                    "margin": margin_f,
                    "total": total_f,
                    "p_home": p_home,
                    "p_away": 1 - p_home,
                    "winner": "home" if margin_f >= 0 else "away",
                    "fair_ml_home": prob_to_american(p_home),
                    "fair_ml_away": prob_to_american(1 - p_home),
                    "margin_without_last2": margin_alt_f,
                    "total_without_last2": total_alt_f,
                    "home_margin_adj": adj_m,
                    "total_adj": adj_t,
                },
                "market": {
                    "margin": ev["market_margin"],
                    "margin_gap": ev["margin_gap"],
                    "total_gap": ev["total_gap"],
                    "spread_move": ev["spread_move"],
                    "total_move": ev["total_move"],
                    "espn": espn_ref,
                },
                "ml": {**ev["ml"], "devig": markets["ml"]["devig"]},
                "spread": {
                    **ev["spread"],
                    "home_line": markets["spread"]["home_line"],
                    "line_source": markets["spread"]["line_source"],
                    "devig": markets["spread"]["devig"],
                },
                "total": {
                    **ev["total"],
                    "line": markets["total"]["line"],
                    "line_source": markets["total"]["line_source"],
                    "devig": markets["total"]["devig"],
                },
                "price_notes": compare_sources(manual, espn, home=g.home_team, away=g.away_team),
                "alerts": alerts,
            }
        )

    parlays = []
    for name, legs in load_parlays(season, week, league).items():
        parlay, parlay_warnings = evaluate_parlay(name, legs, out_games, lookup)
        warnings.extend(parlay_warnings)
        parlays.append(parlay)

    met = context["metrics"]
    return {
        "sport": league.key,
        "league": league.name,
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "week": week,
        "week_label": label,
        "model": {
            "name": model.name,
            "label": model.label,
            "sigma_margin": sigma_m,
            "sigma_total": sigma_t,
            "accuracy": met.get("accuracy"),
            "correct": met.get("correct"),
            "wrong": met.get("wrong"),
            "ties": met.get("ties"),
            "accuracy_big": met.get("accuracy_big"),
            "big_margin": league.big_favorite,
            "mae_margin": met.get("mae_margin"),
            "mae_total": met.get("mae_total"),
            "n_backtest": met.get("n"),
            "calibration_error": context["cal_error"],
            "training_games": int(train.sum()),
        },
        "rules": rules.as_dict(),
        "evidence": {"seasons": context["odds_seasons"], **context["evidence"]},
        "warnings": warnings,
        "games": out_games,
        "parlays": parlays,
    }


def _season_prior(model, seasons: dict, season: int) -> dict:
    """Run through the earlier seasons to leave the prior for the current one."""
    prev_state = prev_sa = None
    for year in sorted(y for y in seasons if y < season):
        sa = seasons[year]
        prior = model.prior(prev_state, prev_sa, sa)
        if sa.completed.any():
            everything = np.ones(len(sa.week), dtype=bool)
            prev_state = model.fit(sa, everything, prior, now_week=int(sa.week.max()) + 1)
            prev_sa = sa
    return model.prior(prev_state, prev_sa, seasons[season])


def _scoreline(
    margin: float, total: float | None, floor: float = 3.0
) -> tuple[float | None, float | None]:
    """Projected score from margin and total, never leaving a team below a floor."""
    if total is None:
        return None, None
    home, away = (total + margin) / 2.0, (total - margin) / 2.0
    if away < floor:
        away, home = floor, floor + margin
    elif home < floor:
        home, away = floor, floor - margin
    return home, away


def evaluate_parlay(
    name: str, legs: list[str], games: list[dict], lookup: dict[str, str]
) -> tuple[dict, list[str]]:
    """A moneyline parlay: model probability, no-vig market probability and what the book pays."""
    warnings: list[str] = []
    by_team: dict[str, tuple[dict, str]] = {}
    for game in games:
        if game.get("no_data"):
            continue
        by_team[game["home"]["id"]] = (game, "home")
        by_team[game["away"]["id"]] = (game, "away")
    out_legs = []
    for team in legs:
        tid = lookup.get(normalize_team(team))
        if tid is None or tid not in by_team:
            warnings.append(f"parlay '{name}': no pending game found for '{team}' this week")
            out_legs.append({"team": team, "no_data": True})
            continue
        game, side_key = by_team[tid]
        side = game["ml"][side_key]
        out_legs.append(
            {
                "team": game[side_key]["name"],
                "abbr": game[side_key]["abbr"],
                "game_id": game["id"],
                "side": side_key,
                "opponent": game["away" if side_key == "home" else "home"]["name"],
                "venue": side_key,
                "p_model": side["p_model"],
                "p_market": side["p_market"],
                "odds": side["odds"],
                "source": side["source"],
                "edge": side["edge"],
                "label": side["label"],
                "no_data": False,
            }
        )
    ok = [leg for leg in out_legs if not leg.get("no_data")]
    out = {"name": name, "legs": out_legs, "complete": bool(ok) and len(ok) == len(out_legs)}
    if not ok:
        return out, warnings
    p_model = float(np.prod([leg["p_model"] for leg in ok]))
    out["p_model"] = p_model
    if all(leg["p_market"] is not None for leg in ok):
        out["p_market"] = float(np.prod([leg["p_market"] for leg in ok]))
    if all(leg["odds"] is not None for leg in ok):
        decimal = float(np.prod([american_to_decimal(leg["odds"]) for leg in ok]))
        out.update(
            {
                "decimal": decimal,
                "implied_prob": 1.0 / decimal,
                "ev_model": p_model * decimal - 1.0,
                "ev_market": out["p_market"] * decimal - 1.0 if "p_market" in out else None,
            }
        )
    out["weakest"] = min(ok, key=lambda leg: leg["p_model"])["team"]
    with_edge = [leg for leg in ok if leg["edge"] is not None]
    if with_edge:
        out["worst_value"] = min(with_edge, key=lambda leg: leg["edge"])["team"]
    return out, warnings


# ------------------------------------------------------------------------ CLI
def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate this week's picks and the site's JSON")
    parser.add_argument("--league", choices=sorted(config.LEAGUES), default="cfb")
    parser.add_argument("--week", type=int, default=None, help="week (default: the current one)")
    parser.add_argument("--season", type=int, default=config.current_season())
    parser.add_argument(
        "--skip-build", action="store_true", help="do not download games or closing lines again"
    )
    parser.add_argument("--offline", action="store_true", help="use the cache only")
    args = parser.parse_args(argv)

    league = config.LEAGUES[args.league]
    rules = Rules(**league.rules)
    client = EspnClient(offline=args.offline, league=league)
    try:
        if not args.skip_build and not args.offline:
            history = [y for y in league.history_seasons if y < args.season]
            build_mod.build(args.season, history, league=league)
            fetch_odds_history(args.season, league=league)
        games = build_mod.load_games(league=league)
        teams = build_mod.load_teams(league=league)
        week = args.week
        if week is None:
            calendar = parse_calendar(client.scoreboard_week(args.season, 1))
            week = week_for(calendar, datetime.now(timezone.utc), league)
            week = next_open_week(games, args.season, week)
        backtest, context = build_backtest(games, args.season, rules, league)
        picks = build_picks(games, teams, args.season, week, context, rules, client, league)
    except EspnError as exc:
        print(f"ERROR: the ESPN API failed and no data is made up: {exc}", file=sys.stderr)
        return 1

    config.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in (("picks.json", picks), ("backtest.json", backtest)):
        text = json.dumps(_clean(payload), ensure_ascii=False, indent=1) + "\n"
        league.site_json(name).write_text(text, encoding="utf-8")

    print_summary(picks, league)
    return 0


def _side_name(game: dict, market: str, key: str) -> str:
    if market == "total":
        return f"{'Over' if key == 'over' else 'Under'} {game['total']['line']:g}"
    team = game[key]["name"]
    if market == "ml":
        return team
    home_line = game["spread"]["home_line"]
    return f"{team} {(home_line if key == 'home' else -home_line):+g}"


def print_summary(picks: dict, league: config.League = config.CFB) -> None:
    games = [g for g in picks["games"] if not g.get("no_data")]
    print(
        f"\n{league.name}, {picks['week_label']} {picks['season']}: {len(games)} games with a"
        f" projection, {sum(g['on_my_list'] for g in games)} with your lines"
    )
    found = []
    for g in games:
        for market, keys in (
            ("ml", ("home", "away")),
            ("spread", ("home", "away")),
            ("total", ("over", "under")),
        ):
            for key in keys:
                side = g[market].get(key)
                if side and side["label"] in (PLAY, AVOID):
                    found.append((side["label"], side["edge"], market, key, g))
    for label in (PLAY, AVOID):
        rows = sorted((f for f in found if f[0] == label), key=lambda f: -f[1])
        print(f"\n{label} ({len(rows)}):")
        for _, edge, market, key, g in rows[:25]:
            side = g[market][key]
            why = f" · {side['reasons'][0]}" if side["reasons"] else ""
            print(
                f"  {market:6s} {_side_name(g, market, key)} ({side['odds']:+d})"
                f" · model {side['p_model']:.0%} vs market {side['p_market']:.0%}"
                f" · edge {edge:+.1%} · {g['away']['abbr']} @ {g['home']['abbr']}{why}"
            )
    for parlay in picks["parlays"]:
        if "p_model" not in parlay:
            continue
        parts = [f"model {parlay['p_model']:.1%}"]
        if parlay.get("p_market"):
            parts.append(f"no-vig market {parlay['p_market']:.1%}")
        if parlay.get("implied_prob"):
            parts.append(f"the book pays as if {parlay['implied_prob']:.1%}")
        parts.append(f"weakest leg: {parlay['weakest']}")
        print(f"\nParlay '{parlay['name']}': " + " · ".join(parts))
    for warning in picks["warnings"]:
        print(f"WARNING: {warning}")
    out = [
        league.site_json(name).relative_to(config.ROOT) for name in ("picks.json", "backtest.json")
    ]
    print(f"\nExported: {out[0]} and {out[1]}")


if __name__ == "__main__":
    raise SystemExit(main())
