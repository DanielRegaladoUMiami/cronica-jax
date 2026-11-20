"""Paths and constants."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
CACHE_DIR = DATA_DIR / "cache"
SITE_DIR = ROOT / "site"
SITE_DATA_DIR = SITE_DIR / "data"

GAMES_PATH = DATA_DIR / "games.parquet"
TEAMS_PATH = DATA_DIR / "teams.parquet"
ODDS_HISTORY_PATH = DATA_DIR / "odds_history.parquet"
ODDS_MANUAL_PATH = DATA_DIR / "odds_manual.csv"
NOTES_PATH = DATA_DIR / "notes.csv"
PARLAYS_PATH = DATA_DIR / "parlays.csv"
PARAMS_PATH = ROOT / "src" / "model" / "params.json"

# ESPN's public API (no key needed).
ESPN_SITE = "https://site.api.espn.com/apis/site/v2/sports/football/college-football"
ESPN_STANDINGS = "https://site.api.espn.com/apis/v2/sports/football/college-football/standings"
FBS_GROUP = 80
SCOREBOARD_LIMIT = 300

# Past seasons are only used to tune hyperparameters and as a preseason prior.
# The current season is always evaluated out of sample.
HISTORY_SEASONS = [2021, 2022, 2023, 2024, 2025]


def current_season(now: datetime | None = None) -> int:
    """A season is named after the year it starts in (January bowls belong to the previous year)."""
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 8 else now.year - 1
