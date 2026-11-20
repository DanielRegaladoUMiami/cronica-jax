"""Soccer picks for the next few days.

Usage:
    python -m src.soccer.picks                # download, backtest, generate picks
    python -m src.soccer.picks --skip-build   # reuse the data already on disk

What it does:
1. Updates the matches of every league and the closing odds of played ones.
2. Fits the selected model on everything played so far, league by league.
3. For each upcoming match: expected goals, the chance of each result, edge
   against ESPN's odds, a stability check and a label.
4. Exports site/data/soccer_picks.json and site/data/soccer_backtest.json.

What the model cannot see (injuries, suspensions, rotation before a European
night) goes in data/soccer_notes.csv: a note per match and, optionally, goals
added to or taken from either team's expected goals.
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

from .. import config as base
from ..data.espn import EspnError
from ..data.odds import normalize_team
from ..model.prob import prob_to_american
from ..model.simulate import record
from ..picks import _clean
from . import config
from .backtest import by_week, calibration, metrics, outcome_index, walk_forward
from .data import build, fetch_odds_history, load_odds_history, load_teams
from .model import (
    AverageModel,
    EloModel,
    EloParams,
    PoissonModel,
    PoissonParams,
    make_leagues,
    most_likely_score,
    outcome_probs,
)
from .rules import SoccerRules, evaluate_match, fair_three_way, fair_two_way
from .simulate import drop_last_matches, simulate_bets

LABELS = ("PLAY", "AVOID", "PASS")
HORIZON_DAYS = 7  # how far ahead fixtures are shown
MOVE_PROB = 0.03  # a market move worth mentioning: 3 points of chance since the open...
MOVE_TOTAL = 0.5  # ...or half a goal on the goals line. Shown as a note, not a red flag.
NO_DATA_REASONS = {
    "in_progress": "already started",
    "postponed": "postponed",
    "canceled": "canceled",
}
NOTES_PATH = base.DATA_DIR / "soccer_notes.csv"


def load_params() -> dict:
    if config.PARAMS_PATH.exists():
        return json.loads(config.PARAMS_PATH.read_text())
    return {}


def build_models(params: dict | None = None) -> dict:
    params = load_params() if params is None else params
    poisson = (
        PoissonModel(PoissonParams(**params["poisson"])) if "poisson" in params else PoissonModel()
    )
    elo = EloModel(EloParams(**params["elo"])) if "elo" in params else EloModel()
    return {"averages": AverageModel(), "poisson": poisson, "elo": elo}


def _quality(probs: np.ndarray, y: np.ndarray) -> dict:
    """Accuracy, ranked probability score and log loss of home/draw/away forecasts."""
    probs = np.clip(probs, 1e-9, 1.0)
    onehot = np.eye(3)[y]
    cum_p, cum_o = np.cumsum(probs, axis=1), np.cumsum(onehot, axis=1)
    return {
        "accuracy": float((probs.argmax(axis=1) == y).mean()),
        "rps": float(np.mean(0.5 * ((cum_p[:, :2] - cum_o[:, :2]) ** 2).sum(axis=1))),
        "log_loss": float(-np.mean(np.log(probs[np.arange(len(y)), y]))),
    }


def _base_rates(history: pd.DataFrame, current: pd.DataFrame) -> dict | None:
    """The floor every model must beat: quoting the same league-average chances for every match.

    The averages come from the past seasons, so the current season is out of sample."""
    source = history if len(history) else current
    if source.empty:
        return None
    y = outcome_index(source)
    rates = np.bincount(y, minlength=3) / len(y)
    out = {"label": "League averages for every match", "rates": [float(r) for r in rates]}
    for key, frame in (("current", current), ("history", history)):
        if len(frame):
            probs = np.tile(rates, (len(frame), 1))
            out[key] = {"n": len(frame), **_quality(probs, outcome_index(frame))}
    return out


def _market_metrics(frame: pd.DataFrame) -> dict | None:
    """Forecast quality of the closing odds themselves, on the same matches as the model."""
    rows = []
    for r in frame.itertuples():
        fair = fair_three_way(r.ml_home, r.ml_draw, r.ml_away)
        if fair is None:
            continue
        over = None
        if r.total == 2.5:
            two = fair_two_way(r.over_odds, r.under_odds)
            over = None if two is None else two[0]
        rows.append(
            (fair[0], fair[1], fair[2], over, r.hg, r.ag, r.p_home, r.p_draw, r.p_away, r.p_over25)
        )
    if not rows:
        return None
    cols = [
        "p_home",
        "p_draw",
        "p_away",
        "over",
        "hg",
        "ag",
        "m_home",
        "m_draw",
        "m_away",
        "m_over",
    ]
    df = pd.DataFrame(rows, columns=cols)
    y = outcome_index(df)

    market = _quality(df[["p_home", "p_draw", "p_away"]].to_numpy(dtype=float), y)
    model = _quality(df[["m_home", "m_draw", "m_away"]].to_numpy(dtype=float), y)
    out = {"n": len(df), **market, **{f"model_{k}": v for k, v in model.items()}}
    with_total = df.dropna(subset=["over"])
    if len(with_total) >= 50:
        went_over = ((with_total["hg"] + with_total["ag"]) > 2.5).astype(float)
        out["brier_over"] = float(np.mean((with_total["over"].astype(float) - went_over) ** 2))
        out["model_brier_over"] = float(np.mean((with_total["m_over"] - went_over) ** 2))
        out["n_total"] = len(with_total)
    return out


# ------------------------------------------------------------------ backtest
def build_backtest(
    matches: pd.DataFrame, season: int, rules: SoccerRules, log=print
) -> tuple[dict, dict]:
    """Walk-forward backtest for every model plus the betting backtest for the selected one."""
    params = load_params()
    leagues = make_leagues(matches)
    years = sorted(int(s) for s in matches["season"].unique())
    eval_years = years[1:] if len(years) > 1 else years  # the first season only warms ratings up
    history_years = [y for y in eval_years if y < season]
    models = build_models(params)
    chosen = params.get("selected", "poisson")
    if chosen not in models:
        chosen = "poisson"

    comparison, frames = [], {}
    for name, model in models.items():
        wf = walk_forward(model, leagues, eval_years)
        frames[name] = wf
        comparison.append(
            {
                "model": name,
                "label": model.label,
                "selected": name == chosen,
                "current": metrics(wf[wf["season"] == season]),
                "history": metrics(wf[wf["season"] < season]),
            }
        )

    wf = frames[chosen]
    cur, hist = wf[wf["season"] == season], wf[wf["season"] < season]
    cal_cur, cal_hist = calibration(cur), calibration(hist)
    cal_error = None
    if cal_hist:
        weighted = sum(abs(b["predicted"] - b["observed"]) * b["n"] for b in cal_hist)
        cal_error = weighted / max(sum(b["n"] for b in cal_hist), 1)

    by_league = []
    for league, name in config.LEAGUES.items():
        m = metrics(cur[cur["league"] == league])
        if m.get("n"):
            by_league.append(
                {
                    "league": league,
                    "name": name,
                    **{k: m[k] for k in ("n", "accuracy", "rps", "mae_total")},
                }
            )

    # ---- simulated bets against closing odds
    odds = load_odds_history()
    with_odds = odds[odds["has_odds"]]
    odds_seasons = sorted(int(s) for s in with_odds["season"].unique() if int(s) in eval_years)
    bets = pd.DataFrame()
    if odds_seasons:
        bets = simulate_bets(models[chosen], leagues, odds, rules, odds_seasons)

    def summary(frame: pd.DataFrame) -> dict:
        if frame.empty:
            return {}
        out = {}
        for market in ("ml", "total", "spread"):
            sel = frame[frame["market"] == market]
            out[market] = {label: record(sel[sel["label"] == label]) for label in LABELS}
        out["all"] = {label: record(frame[frame["label"] == label]) for label in LABELS}
        return out

    cur_bets = bets[bets["season"] == season] if len(bets) else bets
    weekly, cumulative = [], 0.0
    for row in by_week(cur):
        week_days = (
            pd.Timestamp(row["week"], tz="UTC") - pd.Timestamp("1970-01-01", tz="UTC")
        ).days
        if len(cur_bets):
            plays = cur_bets[(cur_bets["week"] == float(week_days)) & (cur_bets["label"] == "PLAY")]
            rec = record(plays)
        else:
            rec = record(pd.DataFrame(columns=["result", "profit"]))
        cumulative += rec["units"]
        weekly.append(
            {
                "week": row["week"],
                "n": row["n"],
                "accuracy": row["accuracy"],
                "rps": row["rps"],
                "mae_total": row["mae_total"],
                "plays": rec["n"],
                "won": rec["won"],
                "lost": rec["lost"],
                "units": rec["units"],
                "cumulative": cumulative,
            }
        )

    market = None
    if len(with_odds):
        cols = ["match_id", "ml_home", "ml_draw", "ml_away", "total", "over_odds", "under_odds"]
        market = _market_metrics(cur.merge(with_odds[cols], on="match_id"))
        market_hist = _market_metrics(hist.merge(with_odds[cols], on="match_id"))
    else:
        market_hist = None

    backtest = {
        "sport": "soccer",
        "generated": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "season": season,
        "season_label": f"{season}-{str(season + 1)[-2:]}",
        "selected": chosen,
        "selected_label": models[chosen].label,
        "tuned_on": params.get("tuned_on"),
        "history_seasons": history_years,
        "comparison": comparison,
        "market": market,
        "market_history": market_hist,
        "reference": _base_rates(hist, cur),
        "by_league": by_league,
        "weeks": weekly,
        "calibration": {"current": cal_cur, "history": cal_hist, "mean_error": cal_error},
        "bets": {
            "current": summary(cur_bets),
            "history": summary(bets),
            "seasons_with_lines": odds_seasons,
            "note": "Flat one-unit bets against the closing odds published by ESPN.",
        },
        "rules": rules.as_dict(),
    }
    context = {
        "model": models[chosen],
        "leagues": leagues,
        "metrics": metrics(cur),
        "cal_error": cal_error,
        "evidence": summary(bets),
        "odds_seasons": odds_seasons,
    }
    m = context["metrics"]
    if m.get("n"):
        log(
            f"Model: {models[chosen].label} · {season} accuracy {m['accuracy']:.1%} · RPS {m['rps']:.4f}"
        )
    return backtest, context


# --------------------------------------------------------------------- picks
def _goals(text) -> float:
    try:
        value = float(str(text).strip())
    except (TypeError, ValueError):
        return 0.0
    return value if math.isfinite(value) else 0.0


def load_notes(path=NOTES_PATH) -> list[dict]:
    """data/soccer_notes.csv: columns home, away, home_goals_adj, away_goals_adj, notes.

    The adjustments are goals added to (or taken from) a team's expected goals,
    for what the model cannot know: a striker out, a rotated side."""
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return [
        {
            "home": (raw.get("home") or "").strip(),
            "away": (raw.get("away") or "").strip(),
            "home_goals_adj": _goals(raw.get("home_goals_adj")),
            "away_goals_adj": _goals(raw.get("away_goals_adj")),
            "notes": (raw.get("notes") or "").strip(),
        }
        for raw in rows
        if (raw.get("home") or "").strip() and (raw.get("away") or "").strip()
    ]


def team_lookup(teams: pd.DataFrame) -> dict[str, str]:
    """Normalized name -> team_id. Accepts the full name, the short name and a unique abbreviation."""
    lookup: dict[str, str] = {}
    abbr_count = teams["abbr"].str.lower().value_counts()
    for row in teams.itertuples():
        lookup[normalize_team(row.team)] = row.team_id
        lookup.setdefault(normalize_team(row.short), row.team_id)
        if abbr_count.get(str(row.abbr).lower(), 0) == 1:
            lookup.setdefault(normalize_team(row.abbr), row.team_id)
    return lookup


def _records(matches: pd.DataFrame, season: int) -> dict[str, str]:
    done = matches[(matches["season"] == season) & matches["completed"]]
    tally: dict[str, list[int]] = {}
    for row in done.itertuples():
        gd = int(row.home_goals) - int(row.away_goals)
        for tid, sign in ((row.home_id, 1), (row.away_id, -1)):
            wdl = tally.setdefault(tid, [0, 0, 0])
            wdl[0 if gd * sign > 0 else 1 if gd == 0 else 2] += 1
    return {tid: f"{w}-{d}-{loss}" for tid, (w, d, loss) in tally.items()}


def build_picks(
    matches: pd.DataFrame,
    teams: pd.DataFrame,
    season: int,
    live_odds: dict[str, dict],
    context: dict,
    rules: SoccerRules,
    now: datetime | None = None,
    notes: list[dict] | None = None,
) -> dict:
    model, leagues = context["model"], context["leagues"]
    now = now or datetime.now(timezone.utc)
    warnings: list[str] = []
    lookup = team_lookup(teams)
    notes_by_match: dict[tuple[str, str], dict] = {}
    for note in notes or []:
        home, away = (
            lookup.get(normalize_team(note["home"])),
            lookup.get(normalize_team(note["away"])),
        )
        if home is None or away is None:
            warnings.append(f"soccer_notes.csv: could not match {note['away']} at {note['home']}")
        else:
            notes_by_match[(home, away)] = note
    cutoff = now.timestamp() / 86400.0
    team_info = teams.set_index("team_id").to_dict("index")
    records = _records(matches, season)
    horizon = pd.Timestamp(now) + pd.Timedelta(days=HORIZON_DAYS)
    window = matches[
        (matches["season"] == season)
        & (matches["date"] >= pd.Timestamp(now) - pd.Timedelta(hours=3))
        & (matches["date"] <= horizon)
        & matches["status"].isin(["scheduled", "in_progress", "postponed", "canceled"])
    ].sort_values(["date", "match_id"])

    states: dict[str, object] = {}
    rank: dict[str, dict[int, int]] = {}
    for league, la in leagues.items():
        state = model.fit(la, np.ones(len(la.h), dtype=bool), cutoff)
        states[league] = state
        if hasattr(state, "attack"):
            current = {
                la.index[t]
                for t in set(window.loc[window["league"] == league, "home_id"])
                | set(window.loc[window["league"] == league, "away_id"])
            }
            in_league = sorted(
                {
                    int(i)
                    for i in np.unique(
                        np.concatenate([la.h[la.season == season], la.a[la.season == season]])
                    )
                }
                | current,
                key=lambda i: -(state.attack[i] + state.defense[i]),
            )
            rank[league] = {i: pos + 1 for pos, i in enumerate(in_league)}

    def team_block(tid: str, name: str, short: str, abbr: str, league: str, idx: int) -> dict:
        info = team_info.get(tid, {})
        state = states[league]
        return {
            "id": tid,
            "name": short or name,
            "full_name": name,
            "abbr": abbr,
            "color": info.get("color") or "",
            "alt_color": info.get("alt_color") or "",
            "record": records.get(tid),
            "rating_rank": rank.get(league, {}).get(idx),
            "attack": float(state.attack[idx]) if hasattr(state, "attack") else None,
            "defense": float(state.defense[idx]) if hasattr(state, "defense") else None,
            "games": int(state.games[idx]),
        }

    games = []
    for g in window.itertuples():
        la, state = leagues[g.league], states[g.league]
        h, a = la.index[g.home_id], la.index[g.away_id]
        base_game = {
            "id": g.match_id,
            "date": g.date.isoformat().replace("+00:00", "Z"),
            "status": g.status,
            "league": g.league,
            "league_name": config.LEAGUES[g.league],
            "on_my_list": False,
            "neutral": False,
            "home": team_block(g.home_id, g.home_team, g.home_short, g.home_abbr, g.league, h),
            "away": team_block(g.away_id, g.away_team, g.away_short, g.away_abbr, g.league, a),
            "notes": "",
        }
        note = notes_by_match.get((g.home_id, g.away_id))
        adj_h = note["home_goals_adj"] if note else 0.0
        adj_a = note["away_goals_adj"] if note else 0.0
        if note:
            base_game["notes"] = note["notes"]
        if g.status != "scheduled":
            reason = NO_DATA_REASONS.get(g.status, "no data from the API")
            games.append({**base_game, "no_data": True, "no_data_reason": reason})
            continue

        lam_h, lam_a = model.expected_goals(state, [h], [a])
        lam_h, lam_a = max(float(lam_h[0]) + adj_h, 0.05), max(float(lam_a[0]) + adj_a, 0.05)
        grid = model.grid(lam_h, lam_a)
        everything = np.ones(len(la.h), dtype=bool)
        alt_state = model.fit(
            la, drop_last_matches(la, everything, cutoff, (h, a)), cutoff, warm=state
        )
        alt_h, alt_a = model.expected_goals(alt_state, [h], [a])
        alt_h, alt_a = max(float(alt_h[0]) + adj_h, 0.05), max(float(alt_a[0]) + adj_a, 0.05)
        grid_alt = model.grid(alt_h, alt_a)
        quote = live_odds.get(g.match_id)
        min_games = int(min(state.games[h], state.games[a]))
        ev = evaluate_match(
            grid=grid, grid_alt=grid_alt, odds=quote, rules=rules, min_games_played=min_games
        )

        p_home, p_draw, p_away = outcome_probs(grid)
        alt_probs = outcome_probs(grid_alt)
        alerts: list[str] = []
        swing = max(abs(p_home - alt_probs[0]), abs(p_away - alt_probs[2]))
        if swing >= 0.06:
            alerts.append(
                f"unstable model: the win chance moves {swing * 100:.0f} points"
                " without each team's last 2 matches"
            )
        if min_games < rules.min_games:
            alerts.append(
                f"thin data: a team has only {min_games} league matches in the model's window"
            )
        fair, fair_open = ev["market_probs"], ev["market_probs_open"]
        price_notes: list[str] = []  # line movement: worth knowing, but not a red flag
        if fair and fair_open:
            names = (g.home_short, "the draw", g.away_short)
            i = int(np.argmax(np.abs(np.array(fair) - np.array(fair_open))))
            if abs(fair[i] - fair_open[i]) >= MOVE_PROB:
                price_notes.append(
                    f"Since the line opened, the market's chance for {names[i]} went from"
                    f" {fair_open[i] * 100:.0f}% to {fair[i] * 100:.0f}%"
                )
        if quote and quote.get("total") is not None and quote.get("total_open") is not None:
            move = quote["total"] - quote["total_open"]
            if abs(move) >= MOVE_TOTAL:
                price_notes.append(
                    f"The goals line {'rose' if move > 0 else 'dropped'} from"
                    f" {quote['total_open']:g} to {quote['total']:g} since it opened"
                )

        score = most_likely_score(grid)
        winner = ("home", "draw", "away")[int(np.argmax([p_home, p_draw, p_away]))]
        espn_ref = None
        if quote:
            espn_ref = {
                "provider": quote.get("provider"),
                "ml_home": quote.get("ml_home"),
                "ml_draw": quote.get("ml_draw"),
                "ml_away": quote.get("ml_away"),
                "total": quote.get("total"),
                "spread_home": quote.get("spread_home"),
                "total_open": quote.get("total_open"),
            }
        games.append(
            {
                **base_game,
                "no_data": False,
                "projection": {
                    "home": lam_h,
                    "away": lam_a,
                    "margin": lam_h - lam_a,
                    "total": lam_h + lam_a,
                    "p_home": p_home,
                    "p_draw": p_draw,
                    "p_away": p_away,
                    "winner": winner,
                    "likely_score": list(score),
                    "fair_ml_home": prob_to_american(p_home),
                    "fair_ml_draw": prob_to_american(p_draw),
                    "fair_ml_away": prob_to_american(p_away),
                    "p_home_without_last2": alt_probs[0],
                    "p_away_without_last2": alt_probs[2],
                    "total_without_last2": alt_h + alt_a,
                    "home_goals_adj": adj_h,
                    "away_goals_adj": adj_a,
                },
                "market": {
                    "p_home": fair[0] if fair else None,
                    "p_draw": fair[1] if fair else None,
                    "p_away": fair[2] if fair else None,
                    "p_home_open": fair_open[0] if fair_open else None,
                    "p_draw_open": fair_open[1] if fair_open else None,
                    "p_away_open": fair_open[2] if fair_open else None,
                    "espn": espn_ref,
                },
                "ml": ev["ml"],
                "total": ev["total"],
                "spread": ev["spread"],
                "price_notes": price_notes,
                "alerts": alerts,
            }
        )

    m = context["metrics"]
    counts = (
        pd.Series([g["league"] for g in games if not g.get("no_data")]).value_counts().to_dict()
    )
    # A scoring level that runs away from the model shows up as a bias in total goals.
    bias, n = m.get("bias_total"), m.get("n") or 0
    if bias is not None and n >= 150 and abs(bias) >= 0.12:
        direction = "fewer" if bias < 0 else "more"
        warnings.append(
            f"This season the model has projected {abs(bias):.2f} {direction} goals per match than"
            f" were scored ({n} matches). Treat its {'unders' if bias < 0 else 'overs'} with extra care."
        )
    return {
        "sport": "soccer",
        "generated": now.isoformat(timespec="seconds"),
        "season": season,
        "season_label": f"{season}-{str(season + 1)[-2:]}",
        "week_label": f"Next {HORIZON_DAYS} days",
        "leagues": [
            {"code": code, "name": name, "games": int(counts.get(code, 0))}
            for code, name in config.LEAGUES.items()
        ],
        "model": {
            "name": model.name,
            "label": model.label,
            "accuracy": m.get("accuracy"),
            "rps": m.get("rps"),
            "log_loss": m.get("log_loss"),
            "mae_margin": m.get("mae_margin"),
            "mae_total": m.get("mae_total"),
            "bias_total": m.get("bias_total"),
            "n_backtest": m.get("n"),
            "calibration_error": context["cal_error"],
            "training_games": int(sum(int(la.completed.sum()) for la in leagues.values())),
            "settings": model.describe(),
        },
        "rules": rules.as_dict(),
        "evidence": {"seasons": context["odds_seasons"], **context["evidence"]},
        "warnings": warnings,
        "games": games,
        "parlays": [],
    }


# ----------------------------------------------------------------------- CLI
def run(
    season: int, *, skip_build: bool = False, offline: bool = False, log=print
) -> tuple[dict, dict]:
    """Build data, backtest and picks. Returns (picks, backtest)."""
    rules = SoccerRules()
    history = [y for y in config.HISTORY_SEASONS if y < season]
    if skip_build and config.MATCHES_PATH.exists():
        # Odds for upcoming matches still come from the (cached) scoreboard.
        matches, live_odds = build(season, history, offline=True, log=lambda *_: None)
    else:
        matches, live_odds = build(season, history, offline=offline, log=log)
        if not offline:
            fetch_odds_history(log=log)
    teams = load_teams()
    backtest, context = build_backtest(matches, season, rules, log=log)
    picks = build_picks(matches, teams, season, live_odds, context, rules, notes=load_notes())
    return picks, backtest


def export(picks: dict, backtest: dict) -> None:
    base.SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    for name, payload in (("soccer_picks.json", picks), ("soccer_backtest.json", backtest)):
        text = json.dumps(_clean(payload), ensure_ascii=False, indent=1) + "\n"
        (base.SITE_DATA_DIR / name).write_text(text, encoding="utf-8")


def print_summary(picks: dict) -> None:
    games = [g for g in picks["games"] if not g.get("no_data")]
    by_league = ", ".join(f"{lg['name']} {lg['games']}" for lg in picks["leagues"] if lg["games"])
    print(f"\nSoccer, {picks['week_label'].lower()}: {len(games)} matches ({by_league})")
    found = []
    for g in games:
        for market, keys in (
            ("ml", ("home", "draw", "away")),
            ("total", ("over", "under")),
            ("spread", ("home", "away")),
        ):
            for key in keys:
                side = g[market].get(key)
                if side and side["label"] in ("PLAY", "AVOID"):
                    found.append((side["label"], side["edge"], market, key, g))
    for label in ("PLAY", "AVOID"):
        rows = sorted((f for f in found if f[0] == label), key=lambda f: -f[1])
        print(f"\n{label} ({len(rows)}):")
        for _, edge, market, key, g in rows[:15]:
            side = g[market][key]
            if market == "total":
                what = f"{'Over' if key == 'over' else 'Under'} {g['total']['line']:g}"
            elif market == "spread":
                line = g["spread"]["home_line"]
                what = f"{g[key]['name']} {(line if key == 'home' else -line):+g}"
            else:
                what = "Draw" if key == "draw" else g[key]["name"]
            why = f" · {side['reasons'][0]}" if side["reasons"] else ""
            print(
                f"  {market:6s} {what} ({side['odds']:+d}) · model {side['p_model']:.0%}"
                f" vs market {side['p_market']:.0%} · edge {edge:+.1%}"
                f" · {g['away']['abbr']} @ {g['home']['abbr']}{why}"
            )
    out_dir = base.SITE_DATA_DIR.relative_to(base.ROOT)
    print(f"\nExported: {out_dir}/soccer_picks.json and {out_dir}/soccer_backtest.json")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate soccer picks and the site's JSON")
    parser.add_argument("--season", type=int, default=config.current_season())
    parser.add_argument("--skip-build", action="store_true", help="do not download again")
    parser.add_argument("--offline", action="store_true", help="use the cache only")
    args = parser.parse_args(argv)
    try:
        picks, backtest = run(args.season, skip_build=args.skip_build, offline=args.offline)
    except EspnError as exc:
        print(f"ERROR: the ESPN API failed and no data is made up: {exc}", file=sys.stderr)
        return 1
    export(picks, backtest)
    print_summary(picks)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
