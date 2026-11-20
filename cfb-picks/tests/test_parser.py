"""ESPN API parser. The fixtures are trimmed real responses
(tests/fixtures), plus three edge cases built from them."""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from src.data.espn import scoreboard_is_settled
from src.data.parse import (
    parse_american,
    parse_calendar,
    parse_event,
    parse_event_odds,
    parse_line,
    parse_odds,
    parse_odds_entry,
    parse_standings_fbs,
    parse_status,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def scoreboard():
    return json.loads((FIXTURES / "scoreboard.json").read_text())


@pytest.fixture(scope="module")
def events(scoreboard):
    return {e["id"]: e for e in scoreboard["events"]}


# ------------------------------------------------------------------ numbers
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("+185", 185),
        ("-110", -110),
        ("−225", -225),
        (455, 455),
        (-105.0, -105),
        ("EVEN", 100),
        (" +100 ", 100),
        (None, None),
        ("", None),
        ("-", None),
        ("OFF", None),
        (0, None),
        (50, None),
        ("abc", None),
        (True, None),
        (float("nan"), None),
    ],
)
def test_parse_american(raw, expected):
    assert parse_american(raw) == expected


@pytest.mark.parametrize(
    "raw, expected",
    [
        ("+5.5", 5.5),
        ("-14", -14.0),
        ("o46.5", 46.5),
        ("u46.5", 46.5),
        ("PK", 0.0),
        (5.5, 5.5),
        (-7, -7.0),
        (None, None),
        ("", None),
        ("OFF", None),
        ("n/a", None),
    ],
)
def test_parse_line(raw, expected):
    assert parse_line(raw) == expected


# --------------------------------------------------------------------- games
def test_scheduled_game(events):
    row = parse_event(events["401856866"])
    assert row["status"] == "scheduled" and row["completed"] is False
    assert (row["away_team"], row["home_team"]) == ("BYU", "Kansas")
    assert (row["away_abbr"], row["home_abbr"]) == ("BYU", "KU")
    assert (
        row["home_points"] is None and row["away_points"] is None
    )  # a game not yet played has no score
    assert row["away_rank"] == 10 and row["home_rank"] is None  # 99 in the API = unranked
    assert (row["season"], row["week"], row["neutral_site"]) == (2026, 12, False)
    assert row["date"] == datetime(2026, 11, 21, 17, 0, tzinfo=timezone.utc)


def test_final_game_has_score_and_sides(events):
    row = parse_event(events["401866458"])
    assert row["status"] == "final" and row["completed"] is True
    assert (row["away_team"], row["away_points"]) == ("Western Michigan", 42)
    assert (row["home_team"], row["home_points"]) == ("Akron", 14)


def test_neutral_site_flag(events):
    assert parse_event(events["401856766"])["neutral_site"] is True


def test_undefined_team_is_dropped(events):
    assert parse_event(events["900000001"]) is None


def test_canceled_game_is_not_completed(events):
    row = parse_event(events["900000002"])
    assert row["status"] == "canceled" and row["completed"] is False and row["home_points"] is None


def test_final_without_real_score_is_not_trusted(events):
    row = parse_event(events["900000003"])  # "final" 0-0: impossible, not used for training
    assert row["status"] == "unknown" and row["completed"] is False


def test_malformed_events_do_not_crash():
    assert parse_event({}) is None
    assert parse_event({"competitions": []}) is None
    assert (
        parse_event(
            {
                "competitions": [
                    {"competitors": [{"homeAway": "home", "team": {"id": "1", "location": "A"}}]}
                ]
            }
        )
        is None
    )
    assert parse_status({}) == "unknown"


# ---------------------------------------------------------------------- odds
def test_scoreboard_odds_use_home_line_convention(events):
    odds = parse_event_odds(events["401856866"])  # ESPN: "BYU -5.5", BYU is the away team
    assert odds["provider"] == "DraftKings"
    assert odds["spread_home"] == 5.5  # HOME team's line: Kansas +5.5
    assert (odds["spread_home_odds"], odds["spread_away_odds"]) == (-108, -112)
    assert (odds["total"], odds["over_odds"], odds["under_odds"]) == (46.5, -115, -105)
    assert (odds["ml_home"], odds["ml_away"]) == (185, -225)
    assert (odds["spread_home_open"], odds["total_open"]) == (7.0, 50.5)


def test_completed_game_has_no_scoreboard_odds(events):
    assert parse_event_odds(events["401866458"]) is None


def test_pickcenter_closing_lines():
    payload = json.loads((FIXTURES / "pickcenter.json").read_text())
    odds = parse_odds(payload["pickcenter"])
    assert odds["spread_home"] == 2.5 and odds["total"] == 54.5
    assert odds["spread_home_open"] == 1.5 and odds["total_open"] == 52.5
    assert odds["ml_home"] is None  # this game had no moneyline: left empty, not made up


def test_odds_fallbacks():
    legacy = {
        "provider": {"name": "consensus"},
        "details": "BYU -8",
        "overUnder": 65.5,
        "spread": 8.0,
        "homeTeamOdds": {"moneyLine": 246, "team": {"abbreviation": "USU"}},
        "awayTeamOdds": {"moneyLine": -311, "team": {"abbreviation": "BYU"}},
    }
    odds = parse_odds_entry(legacy)
    assert odds["spread_home"] == 8.0 and odds["total"] == 65.5
    assert (odds["ml_home"], odds["ml_away"]) == (246, -311)
    home_fav = parse_odds_entry(
        {
            "details": "USU -3",
            "homeTeamOdds": {"team": {"abbreviation": "USU"}},
            "awayTeamOdds": {"team": {"abbreviation": "BYU"}},
        }
    )
    assert home_fav["spread_home"] == -3.0


def test_no_lines_means_none():
    assert parse_odds([]) is None
    assert parse_odds(None) is None
    assert parse_odds([{"provider": {"name": "x"}, "details": None}]) is None


def test_prefers_draftkings_when_several_providers():
    entries = [
        {"provider": {"name": "consensus"}, "spread": -3.0, "overUnder": 50},
        {"provider": {"name": "DraftKings"}, "spread": -3.5, "overUnder": 49.5},
    ]
    assert parse_odds(entries)["provider"] == "DraftKings"


# ------------------------------------------------------ standings and calendar
def test_standings_give_fbs_membership():
    fbs = parse_standings_fbs(json.loads((FIXTURES / "standings.json").read_text()))
    assert fbs["252"] == "Big 12 Conference"  # BYU
    assert fbs["2390"] == "Atlantic Coast Conference"  # Miami
    assert len(fbs) == 9


def test_calendar(scoreboard):
    cal = parse_calendar(scoreboard)
    assert len(cal["regular"]) == 15
    week12 = next(w for w in cal["regular"] if w["week"] == 12)
    assert week12["start"] < datetime(2026, 11, 21, tzinfo=timezone.utc) < week12["end"]
    assert [w["label"] for w in cal["postseason"]] == ["Bowls", "CFP"]


def test_settled_week_detection(events):
    final = events["401866458"]
    after = datetime(2026, 11, 20, tzinfo=timezone.utc)
    assert scoreboard_is_settled({"events": [final]}, now=after) is True
    assert (
        scoreboard_is_settled(
            {"events": [final]}, now=datetime(2026, 11, 11, 12, tzinfo=timezone.utc)
        )
        is False
    )  # too recent
    assert (
        scoreboard_is_settled(
            {"events": [final, events["401856866"]]}, now=after + timedelta(days=30)
        )
        is False
    )  # one game is still unplayed
    assert scoreboard_is_settled({"events": []}) is False
