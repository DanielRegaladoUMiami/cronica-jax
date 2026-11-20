"""Soccer decision rules, bet settlement, the betting simulation and the picks export."""

import json
from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from soccer_league import synthetic_league
from src.model.prob import american_to_decimal, prob_to_american
from src.picks import _clean
from src.soccer.data import ODDS_COLUMNS, week_start
from src.soccer.model import PoissonModel, make_leagues, outcome_probs, score_matrix, total_probs
from src.soccer.picks import build_picks, load_notes, team_lookup
from src.soccer.rules import (
    SoccerRules,
    evaluate_match,
    fair_three_way,
    fair_two_way,
    is_main_line,
    is_quarter_line,
    settle,
)
from src.soccer.simulate import drop_last_matches, simulate_bets

RULES = SoccerRules()
GRID = score_matrix(1.9, 0.9, -0.09)  # a clear home favorite
P_HOME, P_DRAW, P_AWAY = outcome_probs(GRID)


def three_way(home: float, draw: float, away: float, vig: float = 1.05) -> dict:
    """Prices whose no-vig chances are (home, draw, away), up to rounding."""
    assert home + draw + away == pytest.approx(1.0)
    return {
        "ml_home": prob_to_american(home * vig),
        "ml_draw": prob_to_american(draw * vig),
        "ml_away": prob_to_american(away * vig),
    }


def shifted(edge_home: float, edge_away: float) -> dict:
    """A market that sits below the model by the given edges on home and away."""
    return three_way(P_HOME - edge_home, P_DRAW + edge_home + edge_away, P_AWAY - edge_away)


# ------------------------------------------------------------------ helpers
def test_no_vig_probabilities():
    fair = fair_three_way(-150, 280, 400)
    assert fair.sum() == pytest.approx(1.0)
    assert fair[0] > fair[2] > fair[1] or fair[0] > fair[1] > fair[2]
    assert fair_three_way(-150, None, 400) is None
    assert fair_two_way(-110, -110) == pytest.approx((0.5, 0.5))
    assert fair_two_way(None, -110) is None


def test_main_and_quarter_lines():
    assert is_main_line(-110, -110) and is_main_line(-160, 130)
    assert not is_main_line(950, -5000)  # an alternate 4.5 total
    assert is_main_line(None, None)  # nothing to judge by
    assert is_quarter_line(2.25) and is_quarter_line(-0.75) and is_quarter_line(2.3)
    assert not is_quarter_line(2.5) and not is_quarter_line(-1.0) and not is_quarter_line(0.0)


# -------------------------------------------------------------------- labels
def test_play_needs_edge_on_a_side_the_market_respects():
    assert P_HOME - 0.07 >= RULES.ml_min_market  # the test's premise
    out = evaluate_match(grid=GRID, grid_alt=GRID, odds=shifted(0.07, 0.0), rules=RULES)
    home = out["ml"]["home"]
    assert home["label"] == "PLAY" and home["reasons"] == []
    assert home["edge"] == pytest.approx(0.07, abs=0.01)
    assert 0 < home["stake"] <= RULES.stake_cap and home["ev"] > 0
    assert out["ml"]["pick"] == "home"
    assert out["ml"]["draw"]["label"] == "PASS"
    assert sum(out["market_probs"]) == pytest.approx(1.0)


def test_outsider_on_the_moneyline_is_avoided():
    out = evaluate_match(grid=GRID, grid_alt=GRID, odds=shifted(0.0, 0.07), rules=RULES)
    away = out["ml"]["away"]
    assert away["p_market"] < RULES.ml_min_market
    assert away["label"] == "AVOID" and away["reasons"][0].startswith("outsider")
    assert away["stake"] == 0.0


def test_an_edge_too_large_is_avoided():
    out = evaluate_match(grid=GRID, grid_alt=GRID, odds=shifted(0.14, 0.0), rules=RULES)
    home = out["ml"]["home"]
    assert home["edge"] >= RULES.gap_edge
    assert home["label"] == "AVOID" and home["reasons"][0].startswith("the market disagrees")


def test_thin_data_and_unstable_picks_are_avoided():
    odds = shifted(0.07, 0.0)
    thin = evaluate_match(grid=GRID, grid_alt=GRID, odds=odds, rules=RULES, min_games_played=3)
    assert thin["ml"]["home"]["label"] == "AVOID"
    assert thin["ml"]["home"]["reasons"][0].startswith("thin data")

    flipped = score_matrix(0.9, 1.9, -0.09)  # without the last 2 matches the away side is better
    unstable = evaluate_match(grid=GRID, grid_alt=flipped, odds=odds, rules=RULES)
    home = unstable["ml"]["home"]
    assert home["stable"] is False and home["label"] == "AVOID"
    assert home["reasons"][0].startswith("unstable")


def test_agreement_with_the_market_is_a_pass():
    out = evaluate_match(grid=GRID, grid_alt=GRID, odds=shifted(0.0, 0.0), rules=RULES)
    assert {out["ml"][k]["label"] for k in ("home", "draw", "away")} == {"PASS"}
    assert all(out["ml"][k]["stake"] == 0.0 for k in ("home", "draw", "away"))


def test_no_odds_means_no_line():
    out = evaluate_match(grid=GRID, grid_alt=None, odds=None, rules=RULES)
    assert out["ml"]["home"]["label"] == "NO LINE" and out["ml"]["home"]["edge"] is None
    assert out["ml"]["pick"] == "home"  # falls back to the most likely result
    assert out["total"]["pick"] is None and out["spread"]["pick"] is None
    assert out["market_probs"] is None


# -------------------------------------------------------------------- totals
def test_total_goals():
    odds = {"total": 2.5, "over_odds": -110, "under_odds": -110}
    out = evaluate_match(grid=GRID, grid_alt=GRID, odds=odds, rules=RULES)
    over, _, under = total_probs(GRID, 2.5)
    assert out["total"]["line"] == 2.5
    assert out["total"]["over"]["p_model"] == pytest.approx(over)
    assert out["total"]["under"]["p_model"] == pytest.approx(under)
    assert out["total"]["over"]["p_market"] == pytest.approx(0.5)
    assert out["total"]["pick"] == ("over" if over > under else "under")


def test_whole_number_total_leaves_pushes_out():
    odds = {"total": 3.0, "over_odds": 100, "under_odds": -120}
    out = evaluate_match(grid=GRID, grid_alt=GRID, odds=odds, rules=RULES)
    over, push, under = total_probs(GRID, 3.0)
    assert push > 0
    assert out["total"]["over"]["p_model"] == pytest.approx(over / (1 - push))
    assert out["total"]["under"]["p_model"] == pytest.approx(under / (1 - push))
    assert out["total"]["over"]["p_model"] + out["total"]["under"]["p_model"] == pytest.approx(1.0)


def test_alternate_and_quarter_lines_are_not_evaluated():
    alt = {"total": 4.5, "over_odds": 950, "under_odds": -5000}
    assert evaluate_match(grid=GRID, grid_alt=GRID, odds=alt, rules=RULES)["total"]["pick"] is None
    quarter = {"total": 2.25, "over_odds": -110, "under_odds": -110}
    assert (
        evaluate_match(grid=GRID, grid_alt=GRID, odds=quarter, rules=RULES)["total"]["line"] is None
    )
    asian = {"spread_home": -0.75, "spread_home_odds": -105, "spread_away_odds": -115}
    assert (
        evaluate_match(grid=GRID, grid_alt=GRID, odds=asian, rules=RULES)["spread"]["pick"] is None
    )


def test_handicap_uses_the_home_line():
    odds = {"spread_home": -0.5, "spread_home_odds": -110, "spread_away_odds": -110}
    out = evaluate_match(grid=GRID, grid_alt=GRID, odds=odds, rules=RULES)
    assert out["spread"]["home_line"] == -0.5
    assert out["spread"]["home"]["p_model"] == pytest.approx(P_HOME)
    assert out["spread"]["away"]["p_model"] == pytest.approx(P_DRAW + P_AWAY)
    # Handicaps are two-way bets: the outsider flag is only for the three-way result.
    assert not any(r.startswith("outsider") for r in out["spread"]["away"]["reasons"])


# ---------------------------------------------------------------- settlement
@pytest.mark.parametrize(
    ("market", "side", "line", "score", "expected"),
    [
        ("ml", "home", None, (2, 1), 1),
        ("ml", "draw", None, (1, 1), 1),
        ("ml", "away", None, (1, 1), -1),
        ("total", "over", 2.5, (2, 1), 1),
        ("total", "under", 2.5, (2, 1), -1),
        ("total", "over", 3.0, (2, 1), 0),
        ("total", "under", 3.0, (2, 0), 1),
        ("spread", "home", -1.5, (2, 1), -1),
        ("spread", "away", -1.5, (2, 1), 1),
        ("spread", "home", -1.0, (2, 1), 0),
        ("spread", "home", 0.5, (1, 1), 1),
        ("spread", "away", 0.5, (1, 1), -1),
        ("spread", "home", 0.0, (0, 0), 0),
    ],
)
def test_settle(market, side, line, score, expected):
    assert settle(market, side, line, *score) == expected


# ---------------------------------------------------------------- simulation
@pytest.fixture(scope="module")
def league():
    matches, _, _ = synthetic_league()
    return make_leagues(matches)["syn.1"]


def test_drop_last_matches(league):
    la = league
    everything = np.ones(len(la.h), dtype=bool)
    cutoff = float(np.median(la.day))
    mask = drop_last_matches(la, everything, cutoff, (0, 1), n=2)
    dropped = np.flatnonzero(~mask)
    assert 3 <= len(dropped) <= 4  # two each, minus one if they met in those matches
    assert (la.day[dropped] < cutoff).all()
    for team in (0, 1):
        played = np.flatnonzero(((la.h == team) | (la.a == team)) & (la.day < cutoff))
        assert set(played[-2:]) <= set(dropped)
    assert everything.all()  # the input mask is not modified


def test_simulation_settles_every_pick(league):
    la = league
    ids = la.match_id[la.season == 2025]
    quote = {
        "provider": "Test Book",
        "ml_home": 120,
        "ml_draw": 240,
        "ml_away": 230,
        "total": 2.5,
        "over_odds": -110,
        "under_odds": -110,
        "spread_home": -0.5,
        "spread_home_odds": 120,
        "spread_away_odds": -145,
    }
    odds = pd.DataFrame([{"match_id": m, "has_odds": True, **quote} for m in ids])
    odds = odds.reindex(columns=["match_id", "has_odds", *ODDS_COLUMNS])
    bets = simulate_bets(PoissonModel(), {"syn.1": la}, odds, RULES, [2025])

    assert set(bets["market"]) == {"ml", "total", "spread"}
    assert set(bets["label"]) <= {"PLAY", "AVOID", "PASS"}
    assert bets.groupby("match_id").size().max() <= 3  # one pick per market
    assert set(bets["match_id"]) <= set(ids)
    assert (bets.loc[bets["label"] != "AVOID", "flag"] == "").all()
    assert (bets.loc[bets["label"] == "AVOID", "flag"] != "").all()
    assert (bets.loc[bets["label"] != "PLAY", "stake"] == 0).all()
    for row in bets.itertuples():
        if row.result > 0:
            assert row.profit == pytest.approx(american_to_decimal(row.odds) - 1)
        else:
            assert row.profit == (0.0 if row.result == 0 else -1.0)


# --------------------------------------------------------------------- picks
def test_notes_file(tmp_path):
    path = tmp_path / "soccer_notes.csv"
    path.write_text(
        "home,away,home_goals_adj,away_goals_adj,notes\n"
        "Team 3,Team 4,0.5,,Striker back\n"
        ",Team 1,1,1,ignored: no home team\n"
        "Team 5,Team 6,abc,-0.3,\n",
        encoding="utf-8",
    )
    notes = load_notes(path)
    assert [n["home"] for n in notes] == ["Team 3", "Team 5"]
    assert notes[0] == {
        "home": "Team 3",
        "away": "Team 4",
        "home_goals_adj": 0.5,
        "away_goals_adj": 0.0,
        "notes": "Striker back",
    }
    assert notes[1]["home_goals_adj"] == 0.0 and notes[1]["away_goals_adj"] == -0.3
    assert load_notes(tmp_path / "missing.csv") == []


def _fixture(i: int, home: int, away: int, when, status: str = "scheduled") -> dict:
    return {
        "match_id": f"up-{i}",
        "league": "eng.1",
        "season": 2025,
        "date": when,
        "week_start": week_start(when),
        "status": status,
        "completed": False,
        "home_id": str(100 + home),
        "home_team": f"Team {home}",
        "home_short": f"Team {home}",
        "home_abbr": f"T{home}",
        "home_goals": None,
        "away_id": str(100 + away),
        "away_team": f"Team {away}",
        "away_short": f"Team {away}",
        "away_abbr": f"T{away}",
        "away_goals": None,
    }


def test_build_picks_end_to_end():
    played, _, _ = synthetic_league(league="eng.1")
    now = played["date"].max().to_pydatetime() + timedelta(days=2)
    upcoming = pd.DataFrame(
        [
            _fixture(1, 0, 7, now + timedelta(days=1)),
            _fixture(2, 3, 4, now + timedelta(days=3)),
            _fixture(3, 1, 2, now + timedelta(days=12)),  # beyond the 7-day window
            _fixture(4, 5, 6, now + timedelta(days=2), status="postponed"),
        ]
    )
    matches = pd.concat([played, upcoming], ignore_index=True)
    for col in ("date", "week_start"):
        matches[col] = pd.to_datetime(matches[col], utc=True)
    teams = pd.DataFrame(
        [
            {
                "team_id": str(100 + i),
                "team": f"Team {i}",
                "short": f"Team {i}",
                "abbr": f"T{i}",
                "color": "112233",
                "alt_color": "ffffff",
                "league": "eng.1",
                "season": 2025,
            }
            for i in range(8)
        ]
    )
    assert team_lookup(teams)["team 3"] == "103" and team_lookup(teams)["t3"] == "103"

    context = {
        "model": PoissonModel(),
        "leagues": make_leagues(matches),
        "metrics": {"n": 0},
        "cal_error": None,
        "evidence": {},
        "odds_seasons": [],
    }
    quote = {
        "provider": "DraftKings",
        "ml_home": -300,
        "ml_draw": 400,
        "ml_away": 750,
        "total": 2.5,
        "over_odds": -120,
        "under_odds": 100,
        "spread_home": -1.5,
        "spread_home_odds": 100,
        "spread_away_odds": -120,
        "ml_home_open": -220,
        "ml_draw_open": 340,
        "ml_away_open": 550,
        "total_open": 3.5,
    }
    notes = [
        {
            "home": "Team 3",
            "away": "team 4",
            "home_goals_adj": 0.5,
            "away_goals_adj": 0.0,
            "notes": "Striker back",
        },
        {
            "home": "Nobody FC",
            "away": "Team 1",
            "home_goals_adj": 0.0,
            "away_goals_adj": 0.0,
            "notes": "",
        },
    ]
    args = (matches, teams, 2025, {"up-1": quote}, context, RULES)
    picks = build_picks(*args, now=now, notes=notes)
    plain = build_picks(*args, now=now)
    games = {g["id"]: g for g in picks["games"]}

    assert set(games) == {"up-1", "up-2", "up-4"}
    assert games["up-4"]["no_data"] and games["up-4"]["no_data_reason"] == "postponed"
    assert picks["leagues"][0] == {"code": "eng.1", "name": "Premier League", "games": 2}
    assert any("Nobody FC" in w for w in picks["warnings"])

    priced = games["up-1"]  # the best team at home to the worst one
    pr = priced["projection"]
    assert pr["p_home"] + pr["p_draw"] + pr["p_away"] == pytest.approx(1.0)
    assert pr["winner"] == "home" and pr["home"] > pr["away"]
    assert len(pr["likely_score"]) == 2
    market = priced["market"]
    assert market["p_home"] + market["p_draw"] + market["p_away"] == pytest.approx(1.0)
    assert market["espn"]["provider"] == "DraftKings"
    assert priced["ml"]["home"]["edge"] == pytest.approx(pr["p_home"] - market["p_home"])
    assert priced["total"]["line"] == 2.5 and priced["spread"]["home_line"] == -1.5
    assert len(priced["price_notes"]) == 2  # the result price and the goals line both moved
    assert priced["home"]["record"].count("-") == 2 and priced["home"]["rating_rank"] == 1

    unpriced = games["up-2"]
    assert unpriced["ml"]["home"]["label"] == "NO LINE" and unpriced["total"]["pick"] is None
    assert unpriced["notes"] == "Striker back"
    assert unpriced["projection"]["home_goals_adj"] == 0.5
    before = next(g for g in plain["games"] if g["id"] == "up-2")["projection"]
    assert unpriced["projection"]["home"] == pytest.approx(before["home"] + 0.5)
    assert unpriced["projection"]["p_home"] > before["p_home"]

    text = json.dumps(_clean(picks), allow_nan=False)  # ready for the site: no NaN, no numpy
    assert '"sport": "soccer"' in text
