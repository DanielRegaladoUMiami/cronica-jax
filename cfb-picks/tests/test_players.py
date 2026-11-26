"""NFL player projections: box score parsing, the model, the probabilities and the market side."""

import itertools
import json
import math

import numpy as np
import pandas as pd
import pytest
from src.data import kalshi
from src.players import backtest as bt
from src.players import dist, fetch, markets, week
from src.players.model import (
    Params,
    abs_week,
    build_priors,
    fit_team_effects,
    player_estimates,
    prepare,
    project_team,
    split_shares,
)

TEAMS = ["A", "B", "C", "D"]
# name, position, share of targets, share of carries
ROSTER = [
    ("QB", "QB", 0.0, 0.10),
    ("RB1", "RB", 0.10, 0.60),
    ("RB2", "RB", 0.05, 0.30),
    ("WR1", "WR", 0.40, 0.0),
    ("WR2", "WR", 0.30, 0.0),
    ("TE1", "TE", 0.15, 0.0),
]
SOFT_DEFENSE = "D"  # every offense gains 25% more against D


def make_league(weeks: int = 8, season: int = 2026, sit: dict | None = None):
    """A tiny league with fixed roles. `sit` = {(team, week): [players who do not play]}."""
    sit = sit or {}
    pairings = [(("A", "B"), ("C", "D")), (("A", "C"), ("B", "D")), (("A", "D"), ("B", "C"))]
    players, teams = [], []
    for wk in range(1, weeks + 1):
        for g, (home, away) in enumerate(pairings[(wk - 1) % 3]):
            gid = f"g{wk}{g}"
            for team, opp, is_home in ((home, away, True), (away, home, False)):
                boost = 1.25 if opp == SOFT_DEFENSE else 1.0
                pass_att, rush_att = 30.0, 25.0
                out = set(sit.get((team, wk), []))
                active = [r for r in ROSTER if r[0] not in out]
                t_total = sum(r[2] for r in active)
                c_total = sum(r[3] for r in active)
                base = {"game_id": gid, "season": season, "week": wk, "week_index": wk,
                        "date": pd.Timestamp("2026-09-01", tz="UTC") + pd.Timedelta(weeks=wk),
                        "team_id": team, "opp_id": opp, "home": is_home}  # fmt: skip
                rows = []
                for name, _pos, tshare, cshare in active:
                    targets = round(pass_att * 0.95 * tshare / t_total, 3)
                    carries = round(rush_att * cshare / c_total, 3)
                    row = dict.fromkeys(fetch.PLAYER_STATS, 0.0)
                    row.update(targets=targets, rec=targets * 0.65, rec_yds=targets * 8.0 * boost,
                               rec_td=targets * 0.05, rush_att=carries, rush_yds=carries * 4.0 * boost,
                               rush_td=carries * 0.03)  # fmt: skip
                    if name == "QB":
                        row.update(pass_att=pass_att, pass_cmp=pass_att * 0.65,
                                   pass_yds=pass_att * 0.95 * 8.0 * boost, pass_td=1.4, pass_int=0.7)  # fmt: skip
                    rows.append(
                        {**base, "athlete_id": f"{team}-{name}", "name": f"{team} {name}", **row}
                    )
                players.extend(rows)
                totals = {c: sum(r[c] for r in rows) for c in fetch.PLAYER_STATS}
                teams.append({**base, **totals, "plays": pass_att + rush_att, "points": 21.0 * boost,
                              "points_allowed": 21.0})  # fmt: skip
    positions = {f"{t}-{name}": pos for t in TEAMS for name, pos, _, _ in ROSTER}
    p, t = prepare(pd.DataFrame(players), pd.DataFrame(teams))
    return p, t, positions


def state(players, teams, positions, week_index, season=2026, params=None):
    params = params or Params()
    now = int(abs_week(season, week_index))
    priors = build_priors(players[players["t"] < now], positions)
    effects = fit_team_effects(teams, now, params)
    est = player_estimates(players, now, params, positions, priors, teams)
    return now, priors, effects, est, params


# ------------------------------------------------------------------ box scores
def box(team_id, passer, runner, catcher):
    def cat(name, keys, athletes):
        return {"name": name, "keys": keys, "athletes": [
            {"athlete": {"id": aid, "displayName": aid.title()}, "stats": stats} for aid, stats in athletes]}  # fmt: skip

    return {"team": {"id": team_id}, "statistics": [
        cat("passing", ["completions/passingAttempts", "passingYards", "yardsPerPassAttempt", "passingTouchdowns",
                        "interceptions", "sacks-sackYardsLost", "adjQBR", "QBRating"],
            [(passer, ["12/27", "144", "5.3", "0", "1", "2-15", "14.0", "45.9"])]),
        cat("rushing", ["rushingAttempts", "rushingYards", "yardsPerRushAttempt", "rushingTouchdowns", "longRushing"],
            [(runner, ["11", "21", "1.9", "1", "6"]), (passer, ["1", "-3", "-3.0", "0", "0"])]),
        cat("receiving", ["receptions", "receivingYards", "yardsPerReception", "receivingTouchdowns",
                          "longReception", "receivingTargets"],
            [(catcher, ["4", "48", "12.0", "0", "31", "7"]), (runner, ["2", "6", "3.0", "0", "5", "3"])]),
        cat("kicking", ["fieldGoalsMade/fieldGoalAttempts"], [("kicker", ["1/1"])]),
    ]}  # fmt: skip


GAME = {"game_id": "900", "season": 2026, "week": 3, "week_index": 3, "date": pd.Timestamp("2026-09-20", tz="UTC"),
        "home_id": "1", "away_id": "2", "home_points": 24, "away_points": 17}  # fmt: skip


def test_boxscore_is_trimmed_and_parsed():
    payload = {"boxscore": {"players": [box("1", "qb1", "rb1", "wr1"), box("2", "qb2", "rb2", "wr2")],
                            "teams": [{"team": {"id": "1"}, "statistics": [{"name": "totalOffensivePlays", "displayValue": "61"}]},
                                      {"team": {"id": "2"}, "statistics": []}]}, "news": ["dropped"]}  # fmt: skip
    trimmed = fetch.trim_boxscore(payload)
    assert set(trimmed) == {"players", "teams"}
    assert [c["name"] for c in trimmed["players"][0]["statistics"]] == [
        "passing",
        "rushing",
        "receiving",
    ]
    players, teams = fetch.parse_boxscore(trimmed, GAME)
    by = {(r["team_id"], r["athlete_id"]): r for r in players}
    qb = by[("1", "qb1")]
    assert (qb["pass_cmp"], qb["pass_att"], qb["pass_yds"], qb["pass_int"], qb["sacks"]) == (
        12,
        27,
        144,
        1,
        2,
    )
    assert qb["rush_yds"] == -3 and qb["home"] is True and qb["opp_id"] == "2"
    rb = by[("2", "rb2")]
    assert (rb["rush_att"], rb["rush_td"], rb["targets"], rb["rec"]) == (11, 1, 3, 2) and rb[
        "home"
    ] is False
    home = next(t for t in teams if t["team_id"] == "1")
    assert (
        home["targets"] == 10
        and home["plays"] == 61
        and home["points"] == 24
        and home["points_allowed"] == 17
    )


def test_one_sided_boxscore_is_not_used():
    trimmed = fetch.trim_boxscore({"boxscore": {"players": [box("1", "qb1", "rb1", "wr1")]}})
    assert fetch.parse_boxscore(trimmed, GAME) == ([], [])


def test_roster_marks_who_is_out():
    payload = {"athletes": [
        {"position": "offense", "items": [
            {"id": "1", "displayName": "Healthy", "position": {"abbreviation": "WR"}, "injuries": []},
            {"id": "2", "displayName": "Doubt", "position": {"abbreviation": "RB"}, "injuries": [{"status": "Questionable"}]}]},
        {"position": "injuredReserveOrOut", "items": [
            {"id": "3", "displayName": "Gone", "position": {"abbreviation": "TE"}, "injuries": [{"status": "Injured Reserve"}]},
            {"id": "4", "displayName": "Listed", "position": {"abbreviation": "QB"}, "injuries": []}]},
        {"position": "practiceSquad", "items": [{"id": "5", "displayName": "Squad", "position": {"abbreviation": "WR"}}]},
    ]}  # fmt: skip
    rows = {r["name"]: r for r in fetch.parse_roster(payload, "8")}
    assert (
        rows["Healthy"]["injury_status"] is None
        and rows["Doubt"]["injury_status"] == "Questionable"
    )
    assert (
        rows["Gone"]["injury_status"] in fetch.OUT_STATUSES
        and rows["Listed"]["injury_status"] == "Out"
    )
    assert rows["Squad"]["practice_squad"] and not rows["Healthy"]["practice_squad"]


# ----------------------------------------------------------------------- model
def test_shares_add_up_and_the_replacement_takes_the_vacancy():
    positions = {"star": "WR", "backup": "WR", "te": "TE", "gone": "WR"}
    shares = {"star": 0.30, "backup": 0.05, "te": 0.15, "gone": 0.30}
    active = ["star", "backup", "te"]
    plain = split_shares(shares, active, [], positions, 0.9, True)
    with_gone = split_shares(shares, active, ["gone"], positions, 0.9, True, 1.0, 1.0)
    assert sum(plain.values()) == pytest.approx(1.0) and sum(with_gone.values()) == pytest.approx(
        1.0
    )
    # Split by head inside the position: the backup gains far more, in proportion, than the star.
    assert with_gone["backup"] / plain["backup"] > 1.8
    assert with_gone["star"] / plain["star"] < 1.3
    assert with_gone["te"] < plain["te"]


def test_team_effects_find_the_soft_defense():
    players, teams, positions = make_league()
    _now, _priors, effects, _est, _params = state(players, teams, positions, 9)
    soft = effects.defense("ypa", SOFT_DEFENSE)
    assert soft > 0.3 and all(soft > effects.defense("ypa", t) for t in "ABC")
    assert effects.expect("ypa", "A", "D", True) > effects.expect("ypa", "A", "B", True)
    assert effects.expect("ypa", "A", "D", True, opponent=False) == pytest.approx(
        effects.mu["ypa"] + effects.offense("ypa", "A")
    )


def test_roles_are_recovered_and_players_add_up_to_the_team():
    players, teams, positions = make_league()
    _now, priors, effects, est, params = state(players, teams, positions, 9)
    # The raw share is the truth of this toy league; the one used is pulled toward a newcomer's.
    assert est.at["A-WR1", "raw_tshare"] == pytest.approx(0.40, abs=0.01)
    assert est.at["A-RB1", "raw_cshare"] == pytest.approx(0.60, abs=0.01)
    assert 0.30 < est.at["A-WR1", "tshare"] < 0.40 and 0.50 < est.at["A-RB1", "cshare"] < 0.60
    assert bool(est.at["A-WR1", "regular"]) and est.at["A-WR1", "weight"] == 1.0
    active = [f"A-{name}" for name, *_ in ROSTER]
    rows, ctx = project_team(
        "A", "B", True, effects, est, active, "A-QB", [], positions, priors, params
    )
    by = {r["athlete_id"]: r for r in rows}
    assert sum(r["rec_yds"] for r in rows) == pytest.approx(ctx["pass_yds"])
    assert sum(r["rush_att"] for r in rows) == pytest.approx(ctx["rush_att"])
    assert by["A-QB"]["pass_yds"] == pytest.approx(ctx["pass_yds"] * params.qb_share)
    assert by["A-WR1"]["rec_yds"] > by["A-WR2"]["rec_yds"] > by["A-TE1"]["rec_yds"]
    assert by["A-RB1"]["td"] == pytest.approx(by["A-RB1"]["rush_td"] + by["A-RB1"]["rec_td"])
    # Against the soft defense everyone on the offense is projected for more.
    soft, _ = project_team(
        "A", "D", True, effects, est, active, "A-QB", [], positions, priors, params
    )
    assert {r["athlete_id"]: r for r in soft}["A-WR1"]["rec_yds"] > by["A-WR1"]["rec_yds"]


def test_a_missing_runner_hands_his_carries_to_the_others():
    players, teams, positions = make_league()
    _now, priors, effects, est, params = state(players, teams, positions, 9)
    full = [f"A-{name}" for name, *_ in ROSTER]
    without = [a for a in full if a != "A-RB1"]
    base, _ = project_team(
        "A", "B", True, effects, est, full, "A-QB", [], positions, priors, params
    )
    short, ctx = project_team(
        "A", "B", True, effects, est, without, "A-QB", ["A-RB1"], positions, priors, params
    )
    rb2_before = {r["athlete_id"]: r for r in base}["A-RB2"]["rush_att"]
    rb2_after = {r["athlete_id"]: r for r in short}["A-RB2"]["rush_att"]
    assert rb2_after > 1.8 * rb2_before
    assert ctx["missing_carries"] == pytest.approx(0.60, abs=0.08)
    # The ablation "does not know who is out": RB2 keeps his usual workload.
    blind = Params(use_teammates=False)
    kept, _ = project_team(
        "A", "B", True, effects, est, without, "A-QB", ["A-RB1"], positions, priors, blind
    )
    assert {r["athlete_id"]: r for r in kept}["A-RB2"]["rush_att"] == pytest.approx(
        rb2_before, rel=0.05
    )
    assert "A-RB1" not in {r["athlete_id"] for r in kept}


def test_a_backup_who_rarely_gets_the_ball_is_discounted():
    sit = {("A", wk): ["RB2"] for wk in range(2, 8)}  # RB2 touches the ball in weeks 1 and 8 only
    players, teams, positions = make_league(sit=sit)
    _now, _priors, _effects, est, params = state(players, teams, positions, 9)
    assert est.at["A-RB2", "appear"] < 0.45
    assert est.at["A-RB1", "appear"] == pytest.approx(1.0)
    sit_deep = {("A", wk): ["TE1"] for wk in range(2, 8)}
    players, teams, positions = make_league(sit=sit_deep)
    _now, _priors, _effects, est, params = state(players, teams, positions, 9)
    te = est.loc["A-TE1"]
    if not te["regular"]:
        assert params.appear_floor <= te["weight"] < 1.0


def test_a_replacement_quarterback_throws_less():
    players, teams, positions = make_league()
    _now, priors, effects, est, params = state(players, teams, positions, 9)
    active = [f"A-{name}" for name, *_ in ROSTER]
    _, usual = project_team(
        "A", "B", True, effects, est, active, "A-QB", [], positions, priors, params, None, "A-QB"
    )
    swapped = [a for a in active if a != "A-QB"] + ["A-QB2"]
    positions["A-QB2"] = "QB"
    rows, change = project_team(
        "A", "B", True, effects, est, swapped, "A-QB2", [], positions, priors, params, None, "A-QB"
    )
    assert change["qb_change"] and not usual["qb_change"]
    assert change["pass_att"] == pytest.approx(usual["pass_att"] * params.qb_change_volume)
    assert change["rush_att"] > usual["rush_att"]
    assert {r["athlete_id"]: r for r in rows}["A-QB2"]["pass_att"] > 0


def test_the_betting_line_moves_touchdowns_more_than_yards():
    players, teams, positions = make_league()
    _now, priors, effects, est, params = state(players, teams, positions, 9)
    active = [f"A-{name}" for name, *_ in ROSTER]
    _, flat = project_team(
        "A", "B", True, effects, est, active, "A-QB", [], positions, priors, params
    )
    points = effects.expect("points", "A", "B", True) * 1.2
    _, up = project_team(
        "A",
        "B",
        True,
        effects,
        est,
        active,
        "A-QB",
        [],
        positions,
        priors,
        params,
        {"points": points},
    )
    assert up["market_factor"] == pytest.approx(1.2)
    assert up["pass_td"] / flat["pass_td"] > up["pass_yds"] / flat["pass_yds"] > 1.0


def test_walk_forward_only_uses_the_past_and_counts_ghosts():
    sit = {("A", 8): ["RB2"]}
    players, teams, positions = make_league(sit=sit)
    back = bt.walk_forward(players, teams, positions, Params(), [2026])
    assert back["week"].min() == 2  # week 1 has nothing before it
    row = back[(back["athlete_id"] == "A-WR1") & (back["week"] == 8)].iloc[0]
    assert row["p_rec_yds"] == pytest.approx(row["a_rec_yds"], rel=0.25)
    # RB2 has a real role, so his absence in week 8 was "known": he is missing, not a ghost.
    assert back[(back["athlete_id"] == "A-RB2") & (back["week"] == 8)].empty
    assert row["missing_carries"] > 0.2
    table = bt.metrics(back)
    assert {"pass_yds", "rush_yds", "rec_yds"} <= set(table["stat"])
    assert (table["mae"] >= 0).all()


def test_announced_starter_is_last_weeks_quarterback_if_he_plays():
    grp = pd.DataFrame({"athlete_id": ["old", "new"], "pass_att": [3.0, 28.0]})
    assert bt.announced_starter(grp, "old") == "old"
    assert bt.announced_starter(grp, "someone else") == "new"
    assert bt.announced_starter(grp[grp["pass_att"] > 100], "old") is None


# --------------------------------------------------------------- probabilities
def test_negative_binomial_tail():
    assert dist.nb_sf(0, 2.0, 5.0) == 1.0
    assert dist.nb_sf(1, 0.5, 1e6) == pytest.approx(1 - math.exp(-0.5), abs=1e-4)  # Poisson limit
    tail = [dist.nb_sf(n, 4.0, 10.0) for n in range(12)]
    assert all(a >= b for a, b in itertools.pairwise(tail))
    assert dist.nb_sf(3, 4.0, 2.0) < dist.nb_sf(3, 4.0, 50.0) or dist.nb_sf(
        8, 4.0, 2.0
    ) > dist.nb_sf(8, 4.0, 50.0)


def test_distributions_are_calibrated_on_their_own_data():
    rng = np.random.default_rng(7)
    proj = rng.uniform(20, 90, 4000)
    actual = np.maximum(rng.gamma(shape=3.0, scale=proj / 3.0), 0).round()
    frame = pd.DataFrame(
        {
            "p_rec_yds": proj,
            "a_rec_yds": actual,
            "p_rec": proj / 12,
            "a_rec": rng.poisson(proj / 12),
        }
    )
    every = lambda df, stat: pd.Series(True, index=df.index)
    d = dist.fit(frame, every)
    assert "rec_yds" in d.ratios and d.dispersion["rec"] > 5
    # More yards asked for, less likely; a bigger projection, more likely.
    assert d.prob_at_least("rec_yds", 40, 50) > d.prob_at_least("rec_yds", 70, 50)
    assert d.prob_at_least("rec_yds", 60, 80) > d.prob_at_least("rec_yds", 60, 40)
    assert d.prob_at_least("rec_yds", 60, 2.0) is None  # too small a projection to say anything
    assert d.prob_over("rec_yds", 49.5, 50) == d.prob_at_least("rec_yds", 50, 50)
    table = dist.calibration(d, frame, every, "rec_yds", range(20, 121, 20))
    assert (table["predicted"] - table["observed"]).abs().max() < 0.05
    mean = week.mean_for_line(d, "rec_yds", 49.5)
    assert d.prob_over("rec_yds", 49.5, mean) == pytest.approx(0.5, abs=0.02)
    assert week.mean_for_line(d, "pass_td", 1.5) is None


# --------------------------------------------------------------------- markets
def test_names_and_kalshi_tickers():
    assert (
        markets.plain_name("Amon-Ra St. Brown")
        == markets.plain_name("Amon-Ra St. Brown Jr.")
        == "amonra st brown"
    )
    assert markets.plain_name("D'Andre Swift") == "dandre swift"
    known = {"CHI", "DET", "GB", "LAR", "JAX", "TEN", "WSH", "ARI", "NE", "LAC"}
    assert markets.event_teams("KXNFLPASSYDS-26NOV26CHIDET", known) == ("CHI", "DET")
    assert markets.event_teams("KXNFLGAME-26NOV25GBLAR", known) == ("GB", "LAR")
    assert markets.event_teams("KXNFLREC-26NOV29TENJAC", known) == ("TEN", "JAX")
    assert markets.event_teams("KXNFLREC-26NOV29WASARI", known) == ("WSH", "ARI")
    assert markets.event_teams("KXNFLREC-26NOV29NELAC", known) == ("NE", "LAC")
    assert markets.event_teams("KXNFLREC-26NOV29XXXYYY", known) is None


def test_draftkings_lines_are_read_from_espn():
    ref = "http://sports.core.api.espn.com/v2/sports/football/leagues/nfl/seasons/2026/athletes/3046779?lang=en"
    payload = {"count": 3, "pageCount": 1, "items": [
        {"athlete": {"$ref": ref}, "type": {"id": "8"}, "current": {"target": {"value": 279.5}}, "open": {"target": {"value": 275.5}}},
        {"athlete": {"$ref": ref}, "type": {"id": "31"}, "current": {}, "open": {}},
        {"type": {"id": "98"}, "current": {"target": {"value": 27.5}}},
    ]}  # fmt: skip
    assert markets.trim_props(payload)["rows"] == [["3046779", "pass_yds", 279.5, 275.5]]


def test_kalshi_fee_and_empty_book():
    assert kalshi.taker_fee(0.5) == pytest.approx(0.0175)
    assert kalshi.taker_fee(0.97) < 0.003
    assert kalshi.maker_fee(0.5, "quadratic") == 0.0
    assert kalshi.maker_fee(0.5, "quadratic_with_maker_fees") == pytest.approx(0.004375)
    live = kalshi.parse_market({"ticker": "T-250", "yes_bid_dollars": "0.6500", "yes_ask_dollars": "0.6600",
                                "no_bid_dollars": "0.3400", "no_ask_dollars": "0.3500", "volume_fp": "203.39",
                                "floor_strike": 249.5, "yes_sub_title": "Jared Goff: 250+"})  # fmt: skip
    assert (
        live["quoted"]
        and live["yes_ask"] == 0.66
        and live["strike"] == 249.5
        and live["volume"] == pytest.approx(203.39)
    )
    empty = kalshi.parse_market(
        {"ticker": "T", "yes_bid_dollars": "0.0000", "yes_ask_dollars": "1.0000"}
    )
    assert not empty["quoted"] and empty["yes_ask"] is None


# ------------------------------------------------------------------- the flags
def rung(t, p, yes_bid, yes_ask, volume=500, edge_dk=None, p_dk=None):
    no_bid, no_ask = round(1 - yes_ask, 2), round(1 - yes_bid, 2)
    row = {"t": t, "p": p, "p_dk": p_dk, "yes_bid": yes_bid, "yes_ask": yes_ask, "no_bid": no_bid, "no_ask": no_ask,
           "volume": volume, "ticker": f"T-{t}",
           "edge_yes": p - yes_ask - kalshi.taker_fee(yes_ask),
           "edge_no": (1 - p) - no_ask - kalshi.taker_fee(no_ask)}  # fmt: skip
    if edge_dk is not None:
        row["edge_yes_dk"], row["edge_no_dk"] = edge_dk, -edge_dk
    return row


def test_only_likely_contracts_backed_by_a_second_market_are_plays():
    calm = {}
    # Likely (70%), a real price (60 cents), and both the model and DraftKings say it is low.
    play = week.best_contract([rung(50, 0.70, 0.59, 0.60, edge_dk=0.05, p_dk=0.68)], calm)
    assert play["label"] == "PLAY" and play["side"] == "yes"
    assert play["edge"] == pytest.approx(0.70 - 0.60 - kalshi.taker_fee(0.60))
    assert play["p"] == pytest.approx(0.70) and play["p_dk"] == pytest.approx(0.68)
    # The model alone is only a lean: no DraftKings line, one that backs it in part, or a thin market.
    assert week.best_contract([rung(50, 0.70, 0.59, 0.60)], calm)["label"] == "LEAN"
    part = week.best_contract([rung(50, 0.70, 0.59, 0.60, edge_dk=0.02, p_dk=0.64)], calm)
    assert part["label"] == "LEAN" and "only in part" in part["reasons"][0]
    thin = week.best_contract([rung(50, 0.70, 0.59, 0.60, volume=3, edge_dk=0.05, p_dk=0.68)], calm)
    assert thin["label"] == "LEAN"
    unlikely = week.best_contract([rung(50, 0.70, 0.59, 0.60, edge_dk=0.05, p_dk=0.50)], calm)
    assert unlikely["label"] == "LEAN" and "less likely" in unlikely["reasons"][0]
    # The "No" side counts when it is the likely one.
    no = week.best_contract([rung(50, 0.22, 0.30, 0.31, edge_dk=-0.05, p_dk=0.24)], calm)
    assert no["label"] == "PLAY" and no["side"] == "no"
    assert (
        no["p"] == pytest.approx(0.78) and no["p_dk"] == pytest.approx(0.76) and no["price"] == 0.70
    )


def test_long_shots_are_never_flagged():
    calm = {}
    # The user's case: the market says 12%, the model says 16%. Four points of "edge", and no flag.
    assert week.best_contract([rung(50, 0.16, 0.11, 0.12, edge_dk=0.04, p_dk=0.16)], calm) is None
    # Twenty cents for something the model gives 30%: still a long shot.
    assert week.best_contract([rung(50, 0.30, 0.19, 0.20, edge_dk=0.08, p_dk=0.30)], calm) is None
    # A fair price for a coin flip the model does not make likely.
    assert week.best_contract([rung(50, 0.52, 0.44, 0.45, edge_dk=0.05, p_dk=0.52)], calm) is None
    # Too expensive to leave anything, a wide book, or no edge at all.
    assert week.best_contract([rung(50, 0.99, 0.92, 0.93, edge_dk=0.05, p_dk=0.99)], calm) is None
    assert week.best_contract([rung(50, 0.80, 0.50, 0.60, edge_dk=0.05, p_dk=0.80)], calm) is None
    assert week.best_contract([rung(50, 0.62, 0.59, 0.60, edge_dk=0.05, p_dk=0.66)], calm) is None


def test_reasons_to_stop_make_it_a_check():
    calm = {}
    far = week.best_contract([rung(50, 0.85, 0.59, 0.60, edge_dk=0.05, p_dk=0.70)], calm)
    assert far["label"] == "CHECK" and "far from the model" in far["reasons"][0]
    against = week.best_contract([rung(50, 0.70, 0.59, 0.60, edge_dk=-0.02, p_dk=0.59)], calm)
    assert against["label"] == "CHECK"
    for flag in ("questionable", "small_sample", "qb_change", "returning"):
        doubt = week.best_contract(
            [rung(50, 0.70, 0.59, 0.60, edge_dk=0.05, p_dk=0.68)], {flag: True}
        )
        assert doubt["label"] == "CHECK"


def test_team_picks_are_told_about_a_quarterback_change(tmp_path):
    side = lambda label: {"label": label, "stake": 0.01, "reasons": []}
    picks = {"season": 2026, "week": 12, "games": [
        {"id": "1", "away": {"abbr": "NE"}, "home": {"abbr": "LAC"}, "alerts": [],
         "ml": {"home": side("PLAY"), "away": side("PASS"), "pick": "home"},
         "spread": {"home": side("AVOID"), "away": side("PASS")}, "total": {"over": side("PLAY"), "under": side("PASS")}},
        {"id": "2", "alerts": [], "ml": {"home": side("PLAY"), "away": side("PASS")}, "spread": {}, "total": {}},
    ]}  # fmt: skip
    path = tmp_path / "nfl_picks.json"
    path.write_text(json.dumps(picks), encoding="utf-8")
    data = {"season": 2026, "week": 12, "games": [
        {"game_id": "1", "teams": [{"side": "away", "name": "Patriots", "qb": "Tommy DeVito", "usual_qb": "Drake Maye", "qb_change": True},
                                   {"side": "home", "name": "Chargers", "qb": "Justin Herbert", "usual_qb": None, "qb_change": False}]},
        {"game_id": "2", "teams": [{"side": "away", "qb_change": False}, {"side": "home", "qb_change": False}]},
    ]}  # fmt: skip
    data["kalshi_winners"] = {"NE@LAC": {"away": {"yes_ask": 0.52}, "home": {"yes_ask": 0.49}}}
    assert week.annotate_team_picks(data, path, log=lambda *_: None) == 2
    out = json.loads(path.read_text(encoding="utf-8"))
    first, second = out["games"]
    assert first["kalshi"]["away"]["yes_ask"] == 0.52 and "kalshi" not in second
    assert first["ml"]["home"]["label"] == "AVOID" and first["ml"]["home"]["stake"] == 0.0
    assert first["total"]["over"]["label"] == "AVOID" and first["ml"]["away"]["label"] == "PASS"
    assert (
        "Tommy DeVito" in first["alerts"][0] and "Drake Maye" in first["ml"]["home"]["reasons"][0]
    )
    assert second["ml"]["home"]["label"] == "PLAY" and "qb_changes" not in second
    # A file from another week is left alone.
    assert week.annotate_team_picks({**data, "week": 13}, path, log=lambda *_: None) == 0
