"""Leagues, seasons and paths for the soccer side of the project."""

from __future__ import annotations

from datetime import datetime, timezone

from .. import config as base

# ESPN league slug -> display name. Adding a league is one line here.
LEAGUES = {
    "eng.1": "Premier League",
    "esp.1": "LaLiga",
    "ita.1": "Serie A",
    "ger.1": "Bundesliga",
    "fra.1": "Ligue 1",
}

ESPN_SOCCER = "https://site.api.espn.com/apis/site/v2/sports/soccer"

MATCHES_PATH = base.DATA_DIR / "soccer_matches.parquet"
TEAMS_PATH = base.DATA_DIR / "soccer_teams.parquet"
ODDS_HISTORY_PATH = base.DATA_DIR / "soccer_odds_history.parquet"
PARAMS_PATH = base.ROOT / "src" / "soccer" / "params.json"

# Earlier seasons are used to tune the model and as the starting point for ratings.
# A season is named after the year it starts in: 2026 means 2026-27.
HISTORY_SEASONS = [2021, 2022, 2023, 2024, 2025]
SEASON_MONTHS = (8, 9, 10, 11, 12, 1, 2, 3, 4, 5, 6)  # August through June


def current_season(now: datetime | None = None) -> int:
    now = now or datetime.now(timezone.utc)
    return now.year if now.month >= 7 else now.year - 1


def season_months(season: int) -> list[str]:
    """The calendar months of a season as YYYYMM strings (ESPN's scoreboard takes a month)."""
    return [f"{season if m >= 7 else season + 1}{m:02d}" for m in SEASON_MONTHS]
