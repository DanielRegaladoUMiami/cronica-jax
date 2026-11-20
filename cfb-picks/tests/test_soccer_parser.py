"""ESPN soccer payloads -> flat rows. The fixtures are real responses, trimmed."""

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pytest
from src.soccer import config
from src.soccer.data import (
    drop_unreliable_totals,
    is_plausible,
    parse_match,
    parse_match_teams,
    parse_odds,
    parse_odds_entry,
    parse_status,
    parse_summary_odds,
    three_way_overround,
    week_start,
)

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def board():
    return json.loads((FIXTURES / "soccer_scoreboard.json").read_text())["events"]


@pytest.fixture(scope="module")
def summary():
    return json.loads((FIXTURES / "soccer_summary.json").read_text())


# ------------------------------------------------------------------ matches
def test_finished_match(board):
    row = parse_match(board[0], "eng.1")
    assert row["match_id"] == "401879246"
    assert (row["league"], row["season"]) == ("eng.1", 2026)
    assert (row["home_team"], row["away_team"]) == ("Liverpool", "Arsenal")
    assert (row["home_goals"], row["away_goals"]) == (0, 1)
    assert row["status"] == "final" and row["completed"] is True
    assert row["date"] == datetime(2026, 11, 1, 16, 30, tzinfo=timezone.utc)


def test_scheduled_match_has_no_goals(board):
    row = parse_match(board[1], "eng.1")
    assert row["status"] == "scheduled" and row["completed"] is False
    assert row["home_goals"] is None and row["away_goals"] is None
    assert (row["home_abbr"], row["away_abbr"]) == ("MNC", "FUL")


def test_postponed_and_abandoned_are_not_results(board):
    assert parse_match(board[2], "eng.1")["status"] == "postponed"
    abandoned = parse_match(board[3], "eng.1")
    assert abandoned["status"] == "canceled" and abandoned["completed"] is False
    assert abandoned["home_goals"] is None  # the score at the time it was abandoned is not a result


def test_unknown_team_is_dropped(board):
    assert parse_match(board[4], "eng.1") is None
    assert [t["abbr"] for t in parse_match_teams(board[4])] == ["MNC"]


def test_malformed_events_do_not_raise():
    assert parse_match({}, "eng.1") is None
    assert parse_match({"competitions": [{"competitors": []}]}, "eng.1") is None
    assert parse_status({}) == "unknown"


def test_final_without_a_score_is_unknown(board):
    event = json.loads(json.dumps(board[0]))
    event["competitions"][0]["competitors"][0]["score"] = None
    row = parse_match(event, "eng.1")
    assert row["status"] == "unknown" and row["completed"] is False


def test_week_starts_on_tuesday():
    # A weekend round (Friday to Monday) belongs to one week.
    friday = datetime(2026, 11, 20, 19, 0, tzinfo=timezone.utc)
    monday = datetime(2026, 11, 23, 20, 0, tzinfo=timezone.utc)
    tuesday = datetime(2026, 11, 24, 19, 45, tzinfo=timezone.utc)
    assert week_start(friday) == week_start(monday) == datetime(2026, 11, 17, tzinfo=timezone.utc)
    assert week_start(tuesday) == datetime(2026, 11, 24, tzinfo=timezone.utc)


def test_season_runs_from_august_to_june():
    assert config.season_months(2026)[0] == "202608"
    assert config.season_months(2026)[-1] == "202706"
    assert config.current_season(datetime(2026, 11, 20, tzinfo=timezone.utc)) == 2026
    assert config.current_season(datetime(2027, 3, 1, tzinfo=timezone.utc)) == 2026
    assert config.current_season(datetime(2027, 8, 1, tzinfo=timezone.utc)) == 2027


# --------------------------------------------------------------------- odds
def test_scoreboard_odds(board):
    odds = parse_odds_entry(board[1]["competitions"][0]["odds"][0])
    assert odds["provider"] == "DraftKings"
    assert (odds["ml_home"], odds["ml_draw"], odds["ml_away"]) == (-255, 425, 600)
    assert (odds["total"], odds["over_odds"], odds["under_odds"]) == (3.5, 110, -140)
    assert (odds["spread_home"], odds["spread_home_odds"], odds["spread_away_odds"]) == (
        -1.5,
        -105,
        -135,
    )
    assert (odds["ml_home_open"], odds["ml_draw_open"], odds["ml_away_open"]) == (-250, 400, 550)
    assert odds["total_open"] == 3.5


def test_summary_pickcenter_odds(summary):
    odds = parse_summary_odds(summary["new"])
    assert odds["provider"] == "DraftKings"
    assert (odds["ml_home"], odds["ml_draw"], odds["ml_away"]) == (-125, 320, 300)
    assert (odds["total"], odds["over_odds"], odds["under_odds"]) == (3.5, -105, -120)
    assert (odds["spread_home"], odds["spread_home_odds"], odds["spread_away_odds"]) == (
        -0.5,
        -135,
        100,
    )
    assert odds["ml_home_open"] == -145


def test_summary_falls_back_to_the_older_odds_list(summary):
    odds = parse_summary_odds(summary["old"])
    # DraftKings is preferred over Bet365; the "Bet 365" block has no moneyline at all.
    assert odds["provider"] == "DraftKings (old)"
    assert (odds["ml_home"], odds["ml_draw"], odds["ml_away"]) == (140, 235, 200)
    assert (odds["total"], odds["over_odds"], odds["under_odds"]) == (2.5, -110, -120)
    assert (odds["spread_home"], odds["spread_home_odds"], odds["spread_away_odds"]) == (
        0.25,
        -185,
        155,
    )
    assert odds["ml_home_open"] is None and odds["total_open"] is None


def test_live_feeds_and_broken_quotes_are_skipped(summary):
    entry = json.loads(json.dumps(summary["new"]["pickcenter"][0]))
    live = {**entry, "provider": {"name": "DraftKings Live"}}
    assert parse_odds([live]) is None
    assert parse_odds([None]) is None  # what the scoreboard sends for a finished match
    assert parse_odds(None) is None
    assert parse_odds([{"provider": {"name": "Bet 365"}, "drawOdds": {"value": 10.0}}]) is None
    assert parse_odds([live, entry])["provider"] == "DraftKings"


def test_overround_filter():
    fair = {"ml_home": 100, "ml_draw": 300, "ml_away": 300}  # 50% + 25% + 25%
    assert three_way_overround(fair) == pytest.approx(1.0)
    assert is_plausible(fair)
    assert not is_plausible({"ml_home": -1000, "ml_draw": -1000, "ml_away": -1000})
    assert not is_plausible({"ml_home": 900, "ml_draw": 900, "ml_away": 900})  # sums to 30%
    assert not is_plausible({"ml_home": -150, "ml_draw": None, "ml_away": 320})


def test_unreliable_totals_are_blanked():
    odds = pd.DataFrame(
        {
            "match_id": ["a", "b", "c"],
            "provider": ["DraftKings (old)", "DraftKings (old)", "ESPN BET"],
            "season": [2023, 2022, 2023],
            "ml_home": [-150, -150, -150],
            "total": [4.5, 2.5, 2.5],
            "over_odds": [-141, -110, -110],
            "under_odds": [-105, -110, -110],
            "total_open": [None, None, 2.5],
        }
    )
    clean = drop_unreliable_totals(odds)
    assert pd.isna(clean.loc[0, "total"]) and pd.isna(clean.loc[0, "over_odds"])
    assert clean.loc[0, "ml_home"] == -150  # the moneyline of that feed is kept
    assert clean.loc[1, "total"] == 2.5 and clean.loc[2, "total"] == 2.5
    assert odds.loc[0, "total"] == 4.5  # the input is not modified
