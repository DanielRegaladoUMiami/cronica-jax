"""What differs between college football and the NFL: files, team names, the postseason and ties."""

import copy
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from src.data.build import fetch_season, postseason_rounds
from src.data.odds import normalize_team, team_lookup
from src.data.parse import parse_calendar, parse_event, parse_event_odds, parse_team
from src.model.backtest import calibration, metrics
from src.model.rules import Rules
from src.model.simulate import settle
from src.picks import _records, next_open_week, week_for, week_label

from src import config

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def board():
    return json.loads((FIXTURES / "nfl_scoreboard.json").read_text())


def utc(*args) -> datetime:
    return datetime(*args, tzinfo=timezone.utc)


# --------------------------------------------------------------------- config
def test_each_league_has_its_own_files():
    cfb, nfl = config.CFB, config.NFL
    # College football keeps the original names, so nothing already published moves.
    assert cfb.games_path.name == "games.parquet"
    assert cfb.odds_manual_path.name == "odds_manual.csv"
    assert cfb.params_path.name == "params.json"
    assert cfb.site_json("picks.json").name == "picks.json"
    assert nfl.games_path.name == "nfl_games.parquet"
    assert nfl.odds_history_path.name == "nfl_odds_history.parquet"
    assert nfl.notes_path.name == "nfl_notes.csv"
    assert nfl.params_path.name == "nfl_params.json"
    assert nfl.site_json("backtest.json").name == "nfl_backtest.json"
    assert (cfb.cache_prefix, nfl.cache_prefix) == ("", "nfl/")
    assert cfb.site_url.endswith("/football/college-football")
    assert nfl.site_url.endswith("/football/nfl") and nfl.standings_url.endswith("/nfl/standings")
    assert set(config.LEAGUES) == {"cfb", "nfl"}


def test_nfl_red_flags_trip_earlier():
    college, pro = Rules(**config.CFB.rules), Rules(**config.NFL.rules)
    assert college == Rules()
    assert pro.gap_margin < college.gap_margin and pro.gap_total < college.gap_total
    assert pro.move_spread < college.move_spread and pro.move_total < college.move_total
    assert pro.edge_min == college.edge_min and pro.stake_cap == college.stake_cap
    assert config.NFL.big_favorite < config.CFB.big_favorite


# -------------------------------------------------------------------- parsing
def test_nfl_teams_go_by_nickname(board):
    game = board["events"][1]
    by_place = parse_event(game)
    by_nick = parse_event(game, nicknames=True)
    assert (by_place["away_team"], by_place["home_team"]) == ("Miami", "Buffalo")
    assert (by_nick["away_team"], by_nick["home_team"]) == ("Dolphins", "Bills")
    assert by_nick["home_abbr"] == "BUF" and by_nick["status"] == "scheduled"
    assert by_nick["home_rank"] is None  # ESPN sends 99 for "unranked"
    team = parse_team(game["competitions"][0]["competitors"][0], nicknames=True)
    assert (team["team"], team["location"], team["display_name"]) == (
        "Bills",
        "Buffalo",
        "Buffalo Bills",
    )


def test_nfl_final_and_neutral_site(board):
    final = parse_event(board["events"][0], nicknames=True)
    assert final["completed"] and (final["away_team"], final["home_team"]) == ("Colts", "Texans")
    assert (final["away_points"], final["home_points"]) == (3, 26)
    assert (final["season"], final["season_type"], final["week"]) == (2026, 2, 11)
    abroad = parse_event(board["events"][2], nicknames=True)
    assert abroad["neutral_site"] is True and final["neutral_site"] is False


def test_nfl_scoreboard_odds(board):
    odds = parse_event_odds(board["events"][1])
    assert odds["provider"] == "DraftKings"
    assert odds["spread_home"] == -16.5 and odds["total"] == 45.5
    assert odds["ml_home"] < -500 < 500 < odds["ml_away"]  # a 16.5-point favorite
    assert odds["spread_home_open"] is not None


# ----------------------------------------------------------------- postseason
def test_every_nfl_playoff_round_is_its_own_week(board):
    calendar = parse_calendar(board)
    rounds = postseason_rounds(calendar, config.NFL)
    assert [(r["label"], r["week_index"]) for r in rounds] == [
        ("Wild Card", 19),
        ("Divisional Round", 20),
        ("Conference Championships", 21),
        ("Super Bowl", 22),
    ]
    # College bowls all land in one bucket after the last regular week.
    assert {r["week_index"] for r in postseason_rounds(calendar, config.CFB)} == {19}


def test_pro_bowl_is_not_a_round(board):
    blocks = copy.deepcopy(board["leagues"][0]["calendar"])
    blocks[1]["entries"] = board["postseason_2025"]  # ESPN lists the Pro Bowl as week 4 of 5
    calendar = parse_calendar({"leagues": [{"calendar": blocks}]})
    assert "Pro Bowl" in [e["label"] for e in calendar["postseason"]]
    rounds = postseason_rounds(calendar, config.NFL)
    assert [r["label"] for r in rounds][-1] == "Super Bowl" and len(rounds) == 4
    assert rounds[-1]["week_index"] == 22 and rounds[-1]["week"] == 5  # still the fourth round


def test_current_week_and_its_label(board):
    calendar = parse_calendar(board)
    assert week_for(calendar, utc(2026, 8, 1), config.NFL) == 1  # before the season
    assert week_for(calendar, utc(2026, 9, 10), config.NFL) == 1
    assert week_for(calendar, utc(2027, 1, 15), config.NFL) == 19  # Wild Card weekend
    assert week_for(calendar, utc(2027, 1, 30), config.NFL) == 21
    assert week_for(calendar, utc(2027, 3, 1), config.NFL) == 22  # after the Super Bowl
    assert week_for(calendar, utc(2027, 1, 30), config.CFB) == 19  # the bowl bucket
    assert week_label(calendar, 2, config.NFL) == "Week 2"
    assert week_label(calendar, 20, config.NFL) == "Divisional Round"
    assert week_label(calendar, 19, config.CFB) == "Postseason"


def test_a_finished_week_rolls_to_the_next_one():
    games = pd.DataFrame(
        {
            "season": [2026] * 5,
            "week_index": [11, 11, 12, 12, 13],
            "completed": [True, True, True, False, False],
        }
    )
    assert next_open_week(games, 2026, 11) == 12  # every week-11 game is final
    assert next_open_week(games, 2026, 12) == 12  # one still to play
    assert next_open_week(games, 2026, 13) == 13  # nothing after it
    assert next_open_week(games, 2025, 11) == 11  # no games known for that season


# ------------------------------------------------------------- season download
class StubClient:
    """Stands in for EspnClient: returns canned weeks and fails if standings are asked for."""

    league = config.NFL

    def __init__(self, weeks: dict) -> None:
        self.weeks = weeks
        self.asked: list[tuple[int, int]] = []

    def scoreboard_week(self, season, week, seasontype=2, **_) -> dict:
        self.asked.append((seasontype, week))
        return self.weeks.get((seasontype, week), {"events": []})

    def standings(self, *args, **kwargs):
        raise AssertionError("the NFL has no lower division: standings are not needed")


def _game(board, gid, home, away, score, season_type, week, final=True) -> dict:
    """An event between two fixture teams (or two made-up all-star sides)."""
    teams = {
        c["team"]["abbreviation"]: c["team"]
        for e in board["events"]
        for c in e["competitions"][0]["competitors"]
    }
    stars = {
        "AFC": {
            "id": "31",
            "location": "AFC",
            "name": "AFC",
            "abbreviation": "AFC",
            "displayName": "AFC",
        },
        "NFC": {
            "id": "32",
            "location": "NFC",
            "name": "NFC",
            "abbreviation": "NFC",
            "displayName": "NFC",
        },
    }
    teams = {**teams, **stars}
    state = {"name": "STATUS_FINAL", "state": "post", "completed": True}
    if not final:
        state = {"name": "STATUS_SCHEDULED", "state": "pre", "completed": False}
    return {
        "id": gid,
        "date": f"2026-{9 + week // 3:02d}-{10 + week:02d}T18:00Z",
        "season": {"year": 2026, "type": season_type},
        "week": {"number": week},
        "status": {"type": state},
        "competitions": [
            {
                "neutralSite": False,
                "competitors": [
                    {"homeAway": "home", "score": str(score[0]), "team": teams[home]},
                    {"homeAway": "away", "score": str(score[1]), "team": teams[away]},
                ],
            }
        ],
    }


def test_nfl_season_download(board):
    calendar = [
        {
            "value": "2",
            "entries": [
                {
                    "label": "Week 1",
                    "value": "1",
                    "startDate": "2026-09-06T08:00Z",
                    "endDate": "2026-09-16T07:59Z",
                },
                {
                    "label": "Week 2",
                    "value": "2",
                    "startDate": "2026-09-16T08:00Z",
                    "endDate": "2026-09-23T07:59Z",
                },
            ],
        },
        {
            "value": "3",
            "entries": [
                {
                    "label": "Wild Card",
                    "value": "1",
                    "startDate": "2027-01-13T08:00Z",
                    "endDate": "2027-01-20T07:59Z",
                },
                {
                    "label": "Pro Bowl",
                    "value": "4",
                    "startDate": "2027-01-28T08:00Z",
                    "endDate": "2027-02-04T07:59Z",
                },
                {
                    "label": "Super Bowl",
                    "value": "5",
                    "startDate": "2027-02-04T08:00Z",
                    "endDate": "2027-02-12T07:59Z",
                },
            ],
        },
    ]
    weeks = {
        (2, 1): {
            "leagues": [{"calendar": calendar}],
            "events": [
                _game(board, "1", "BUF", "MIA", (27, 20), 2, 1),
                _game(board, "2", "HOU", "IND", (17, 17), 2, 1),
            ],
        },
        (2, 2): {
            "events": [
                _game(board, "3", "SF", "MIN", (24, 30), 2, 2),
                _game(board, "4", "MIA", "HOU", (10, 13), 2, 2),
            ]
        },
        (3, 1): {"events": [_game(board, "5", "BUF", "MIN", (31, 14), 3, 1)]},
        (3, 4): {"events": [_game(board, "6", "AFC", "NFC", (60, 50), 3, 4)]},
        (3, 5): {"events": [_game(board, "7", "BUF", "SF", (0, 0), 3, 5, final=False)]},
    }
    client = StubClient(weeks)
    games, teams, _ = fetch_season(client, 2026, log=lambda *_: None)

    assert (3, 4) not in client.asked  # the Pro Bowl week is never requested
    assert sorted(games["game_id"]) == ["1", "2", "3", "4", "5", "7"]  # no all-star game
    index = dict(zip(games["game_id"], games["week_index"], strict=True))
    assert (index["1"], index["4"], index["5"], index["7"]) == (
        1,
        2,
        3,
        4,
    )  # Wild Card, then the Super Bowl
    assert games["home_fbs"].all() and games["away_fbs"].all() and not games["fcs_game"].any()
    assert sorted(teams["team"]) == ["49ers", "Bills", "Colts", "Dolphins", "Texans", "Vikings"]
    tie = games[games["game_id"] == "2"].iloc[0]
    assert tie["completed"] and tie["home_points"] == tie["away_points"] == 17


# ----------------------------------------------------------------------- ties
def test_a_tie_has_no_winner_to_pick():
    df = pd.DataFrame(
        {
            "pred_margin": [2.0, 5.0, -1.0],
            "margin": [3.0, 0.0, -7.0],
            "pred_total": [44.0, 41.0, 47.0],
            "total": [40.0, 40.0, 51.0],
        }
    )
    m = metrics(df, sigma_margin=13.0, big=7.0)
    assert m["n"] == 3 and m["accuracy"] == 1.0  # two decided games, both called right
    assert (m["correct"], m["wrong"], m["ties"]) == (2, 0, 1)
    assert m["n_big"] == 0 and m["accuracy_big"] is None and m["big_margin"] == 7.0
    assert 0 < m["brier"] < 0.25
    bins = calibration(df, sigma=13.0, min_n=1)
    assert sum(b["n"] for b in bins) == 2
    assert calibration(df, sigma=13.0) == []  # three games are too few for a percentage


def test_ties_and_pushes_are_refunded():
    assert settle("ml", "home", None, 0.0, 40.0) == 0
    assert settle("ml", "away", None, 0.0, 40.0) == 0
    assert settle("spread", "home", -3.0, 3.0, 40.0) == 0  # the favorite wins by exactly the spread
    assert settle("spread", "away", -3.0, 3.0, 40.0) == 0
    assert settle("total", "over", 40.0, 3.0, 40.0) == 0
    assert settle("spread", "home", -3.0, 4.0, 40.0) == 1


def test_records_show_ties():
    games = pd.DataFrame(
        {
            "season": [2026, 2026, 2026, 2025],
            "completed": [True, True, False, True],
            "home_id": ["1", "2", "1", "1"],
            "away_id": ["2", "1", "2", "2"],
            "home_points": [27, 20, None, 3],
            "away_points": [20, 20, None, 30],
        }
    )
    assert _records(games, 2026) == {"1": "1-0-1", "2": "0-1-1"}
    assert _records(games, 2025) == {"1": "0-1", "2": "1-0"}


# ---------------------------------------------------------------- team lookup
def test_team_lookup_takes_a_city_only_when_it_is_one_team():
    teams = pd.DataFrame(
        {
            "team_id": ["12", "19", "20", "24"],
            "team": ["Chiefs", "Giants", "Jets", "Chargers"],
            "abbr": ["KC", "NYG", "NYJ", "LAC"],
            "display_name": [
                "Kansas City Chiefs",
                "New York Giants",
                "New York Jets",
                "Los Angeles Chargers",
            ],
            "location": ["Kansas City", "New York", "New York", "Los Angeles"],
        }
    )
    lookup = team_lookup(teams)
    for name in ("Chiefs", "kansas city chiefs", "KC", "Kansas City"):
        assert lookup[normalize_team(name)] == "12"
    assert lookup[normalize_team("Jets")] == "20" and lookup[normalize_team("NYG")] == "19"
    assert normalize_team("New York") not in lookup  # two teams: it would be a guess
    assert lookup[normalize_team("Los Angeles")] == "24"  # only one here
    # College files written before the location column existed still load.
    assert team_lookup(teams.drop(columns="location"))[normalize_team("Chargers")] == "24"
